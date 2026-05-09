"""
runtime/monitor.py — Live Command Monitor
Zero external libraries. Pure Python stdlib only.

Terminal 1: python main.py monitor   (leave running)
Terminal 2: type commands normally   (detected automatically)
"""

import os
import sys
import time
import json
import subprocess
from typing import Optional, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.normalizer      import normalize_command
from core.model           import load_model, predict
from runtime.user_manager import load_user_profile

MIN_WINDOW   = 5
BUFFER_MAX   = 25
SESSION_GAP  = 45.0
LOG_PATH     = os.path.join("reports", "live_log.jsonl")

SAFE_BASES = {
    "ls", "pwd", "cd", "echo", "clear", "man",
    "help", "date", "cat", "less", "more"
}

CONTEXT_WEIGHTS = {
    "recon_seen":   15,
    "network_seen": 20,
    "priv_seen":    15,
    "exfil_seen":   35,
    "cleanup_seen": 30,
    "payload_seen": 40,
}

CONTEXT_TRIGGERS = {
    "recon_seen":   {"whoami", "id", "uname", "hostname", "ps",
                     "netstat", "ss", "lsof", "w", "who"},
    "network_seen": {"nmap", "ssh", "scp", "sftp", "rsync",
                     "nc", "ncat", "ping", "arp", "dig"},
    "priv_seen":    {"sudo", "su", "pkexec", "doas"},
    "exfil_seen":   {"scp", "rsync", "sftp", "ftp"},
    "payload_seen": set(),
    "cleanup_seen": set(),
}


class SessionContext:
    def __init__(self):
        self.flags            = {k: False for k in CONTEXT_WEIGHTS}
        self.session_risk_floor = 0.0
        self.cmd_count        = 0

    def update(self, cmd, base):
        self.cmd_count += 1
        c = cmd.lower()
        for flag, triggers in CONTEXT_TRIGGERS.items():
            if base in triggers and not self.flags[flag]:
                self.flags[flag] = True
        if "history -c" in c or "unset histfile" in c or "export histsize=0" in c:
            self.flags["cleanup_seen"] = True
        if "|bash" in c.replace(" ", "") or "|sh" in c.replace(" ", ""):
            self.flags["payload_seen"] = True
        floor = sum(w for k, w in CONTEXT_WEIGHTS.items() if self.flags[k])
        self.session_risk_floor = min(floor, 60.0)

    def context_bonus(self, base):
        bonus = 0.0
        if base in SAFE_BASES:
            if self.flags["recon_seen"]:   bonus += 8
            if self.flags["network_seen"]: bonus += 12
            if self.flags["exfil_seen"]:   bonus += 20
        return bonus

    def active_flags(self):
        return [k for k, v in self.flags.items() if v]

    def session_label(self):
        f = self.active_flags()
        if not f:
            return "clean"
        return "+".join(f).replace("_seen", "")

    def reset(self):
        self.__init__()


def instant_classify(cmd, ctx):
    c      = cmd.strip().lower()
    tokens = c.split()
    base   = tokens[0] if tokens else ""

    if base in {"scp", "rsync", "sftp"} and ("@" in c or "://" in c):
        return {"status": "ANOMALOUS", "score": 82, "intent": "Data Exfiltration"}

    if base in {"curl", "wget"} and ("|bash" in c.replace(" ", "") or "|sh" in c.replace(" ", "")):
        return {"status": "ANOMALOUS", "score": 90, "intent": "Payload Execution"}

    if "history -c" in c or "unset histfile" in c or "export histsize=0" in c:
        return {"status": "ANOMALOUS", "score": 75, "intent": "Anti-Forensics"}

    if base == "nmap" and "-" in c:
        score = 55 if ctx.flags["priv_seen"] else 40
        return {
            "status": "ANOMALOUS" if score >= 50 else "SUSPICIOUS",
            "score":  score,
            "intent": "Network Reconnaissance",
        }

    if base == "ssh" and "@" in c:
        score = 65 if ctx.flags["recon_seen"] else 42
        return {
            "status": "ANOMALOUS" if score >= 50 else "SUSPICIOUS",
            "score":  score,
            "intent": "Lateral Movement",
        }

    if base in {"sudo", "su"} and len(tokens) > 1:
        if tokens[1] in {"bash", "sh", "su", "-i", "-s"} or "/etc/shadow" in c:
            score = 65 if ctx.flags["recon_seen"] else 48
            return {
                "status": "ANOMALOUS" if score >= 50 else "SUSPICIOUS",
                "score":  score,
                "intent": "Privilege Escalation",
            }

    if base in {"whoami", "id"} and ctx.flags["network_seen"]:
        return {"status": "SUSPICIOUS", "score": 38, "intent": "Post-Exploit Recon"}

    if base == "find" and ("perm" in c or "writable" in c):
        return {"status": "SUSPICIOUS", "score": 45, "intent": "Privilege Discovery"}

    return None


def print_event(user, cmd, status, score, intent, reason="", ctx_label=""):
    ts    = time.strftime("%H:%M:%S")
    icon  = (
        "[!!!]" if score >= 65 else
        "[!! ]" if score >= 48 else
        "[!  ]" if score >= 30 else
        "[ OK]"
    )
    level = (
        "CRITICAL" if score >= 65 else
        "HIGH    " if score >= 48 else
        "MEDIUM  " if score >= 30 else
        "NORMAL  "
    )
    intent_str = (intent or "Normal")[:28]
    ctx_str    = "  ctx=[{}]".format(ctx_label) if ctx_label and ctx_label != "clean" else ""

    print("")
    print("[{}] [{}] $ {}".format(ts, user, cmd))
    print("  {} {} | Score: {:5.1f}/100 | Intent: {}{}".format(
        icon, level, score, intent_str, ctx_str
    ))
    if reason:
        print("  Reason: {}".format(reason))


def log_event(user, cmd, status, score, intent):
    os.makedirs("reports", exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({
            "ts":     round(time.time(), 2),
            "ts_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user":   user,
            "cmd":    cmd,
            "status": status,
            "score":  round(score, 1),
            "intent": intent or "Normal",
        }) + "\n")


def _ensure_prompt_command():
    """
    Add PROMPT_COMMAND to ~/.bashrc so bash flushes history
    after every command. This is what makes real-time detection work.
    Uses double-quotes carefully to avoid syntax errors.
    """
    bashrc = os.path.expanduser("~/.bashrc")
    try:
        existing = open(bashrc).read() if os.path.exists(bashrc) else ""
        if "history -a" not in existing:
            with open(bashrc, "a") as f:
                f.write("\n# DNA IDS: flush history after every command\n")
                # Write the line using double-quote string — no nested quote issue
                f.write("PROMPT_COMMAND=" + '"' + "history -a" + '"' + "\n")
            print("  [OK] Added PROMPT_COMMAND to ~/.bashrc")
            print("  [!] Run this in your OTHER terminal:  source ~/.bashrc")
            print("")
        else:
            print("  [OK] PROMPT_COMMAND already set -- real-time detection active")
            print("")
    except Exception as e:
        print("  [!] Could not write ~/.bashrc: {}".format(e))
        print("")


def stream_history(history_path):
    """
    Use tail -f to watch ~/.bash_history in real time.
    Each new line = one command just executed in the other terminal.
    """
    _ensure_prompt_command()
    proc = subprocess.Popen(
        ["tail", "-f", "-n", "0", history_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    for line in proc.stdout:
        cmd = line.strip()
        if cmd and not cmd.startswith("#"):
            yield cmd


def stream_stdin():
    """Manual mode: type commands directly into this terminal."""
    print("[Monitor] Type commands below (Enter to submit, Ctrl+D to quit):")
    print("")
    while True:
        try:
            line = input()
            if line.strip():
                yield line.strip()
        except EOFError:
            break


class Smoother:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self._v    = 0.0

    def update(self, v):
        self._v = self.alpha * v + (1 - self.alpha) * self._v
        return round(self._v, 1)

    def reset(self):
        self._v = 0.0


def run_stream(use_stdin=False):
    iso, scaler, global_profile, clf = load_model()
    user         = (os.getenv("USER") or os.getenv("USERNAME") or "unknown").strip()
    user_profile = load_user_profile(user) or global_profile

    history_path = os.path.expanduser("~/.bash_history")
    has_history  = os.path.exists(history_path) and not use_stdin

    print("+" + "-" * 68 + "+")
    print("|     DNA FINGERPRINTING SYSTEM -- BEHAVIORAL IDS                    |")
    print("+" + "-" * 68 + "+")
    print("|  Monitoring user : {:<49}|".format(user))
    print("|  Mode            : {:<49}|".format(
        "bash_history (auto)" if has_history else "stdin (manual)"
    ))
    print("|  Profile         : {:<49}|".format(
        "per-user" if user_profile is not global_profile else "global baseline"
    ))
    print("+" + "-" * 68 + "+")
    if has_history:
        print("|  STEP 1: Keep THIS terminal open (monitoring)                     |")
        print("|  STEP 2: Open a NEW terminal                                      |")
        print("|  STEP 3: In the new terminal run:  source ~/.bashrc               |")
        print("|  STEP 4: Type commands normally -- alerts appear here             |")
    else:
        print("|  Type commands below (one per line):                              |")
    print("+" + "-" * 68 + "+")
    print("")

    buffer   = []
    last_ts  = time.time()
    smoother = Smoother()
    ctx      = SessionContext()

    stream = stream_history(history_path) if has_history else stream_stdin()

    for raw_cmd in stream:
        now = time.time()

        if now - last_ts > SESSION_GAP and buffer:
            print("\n  [~] New session (gap {:.0f}s) -- context reset".format(
                now - last_ts
            ))
            buffer = []
            smoother.reset()
            ctx.reset()
        last_ts = now

        norm   = normalize_command(raw_cmd)
        tokens = raw_cmd.strip().lower().split()
        base   = tokens[0] if tokens else ""

        ctx.update(raw_cmd, base)
        instant = instant_classify(raw_cmd, ctx)

        buffer.append(norm)
        if len(buffer) > BUFFER_MAX:
            buffer = buffer[-BUFFER_MAX:]

        if len(buffer) >= MIN_WINDOW:
            result         = predict(
                "\n".join(buffer[-MIN_WINDOW:]),
                iso, scaler, user_profile, clf
            )
            session_status = result["status"]
            session_score  = result["risk_score"]
            session_intent = result.get("intent", "Normal")
        else:
            session_status = "NORMAL"
            session_score  = 10.0
            session_intent = "Normal"

        ctx_bonus     = ctx.context_bonus(base)
        session_score = max(session_score, ctx.session_risk_floor) + ctx_bonus

        if instant and instant["status"] == "ANOMALOUS":
            final_status = "ANOMALOUS"
            final_score  = max(instant["score"], session_score)
            final_intent = instant["intent"]
            reason       = "Rule: {}".format(final_intent)
        elif instant and instant["status"] == "SUSPICIOUS":
            final_status = "SUSPICIOUS" if session_status != "ANOMALOUS" else "ANOMALOUS"
            final_score  = max(instant["score"], session_score)
            final_intent = instant["intent"]
            reason       = "Pattern: {}".format(final_intent)
        else:
            final_status = session_status
            final_score  = session_score
            final_intent = session_intent
            reason       = ""

        if final_score >= 65:
            final_status = "ANOMALOUS"
        elif final_score >= 34:
            final_status = "SUSPICIOUS"
        else:
            final_status = "NORMAL"

        if final_status == "ANOMALOUS" and final_score >= 65:
            smoothed = final_score
            buffer   = []
            smoother.reset()
        else:
            smoothed = smoother.update(final_score)

        log_event(user, raw_cmd, final_status, smoothed, final_intent)
        print_event(
            user, raw_cmd, final_status, smoothed,
            final_intent, reason, ctx.session_label()
        )
