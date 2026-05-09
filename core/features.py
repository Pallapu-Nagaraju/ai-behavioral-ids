"""
core/features.py — Feature Engineering (Pure Python)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extracts 20 behavioral signals from raw bash history.
Zero external dependencies — stdlib only (re, math, collections).

Changes from previous version:
  - Added _mean() and _std() helpers (removed numpy dependency)
  - All math done with stdlib math module
  - sequence_attack_score unchanged (was already pure Python)
"""

import re
import math
from collections import Counter
from typing import List, Dict, Tuple

MIN_COMMANDS  = 10
SHORT_THRESH  = 25
MEDIUM_THRESH = 60

PRIVILEGE_CMDS = {"sudo", "su", "doas", "pkexec", "visudo"}
NETWORK_CMDS   = {"ssh", "scp", "sftp", "curl", "wget", "ftp",
                  "netstat", "ss", "nmap", "ping", "traceroute",
                  "nc", "ncat", "telnet", "arp", "dig", "nslookup"}
FILE_CMDS      = {"ls", "cd", "cp", "mv", "rm", "mkdir", "rmdir",
                  "touch", "find", "ln", "chmod", "chown", "stat",
                  "tar", "zip", "unzip", "gzip", "cat", "less",
                  "more", "head", "tail", "diff", "file", "wc"}
DEV_CMDS       = {"git", "python", "python3", "pip", "pip3", "npm",
                  "node", "gcc", "make", "cmake", "cargo", "go",
                  "java", "mvn", "gradle", "docker", "kubectl",
                  "vim", "nano", "emacs", "code", "pytest", "tox"}
RECON_CMDS     = {"whoami", "id", "hostname", "uname", "ps", "top",
                  "htop", "w", "who", "last", "lastlog", "history",
                  "env", "printenv", "set", "uptime", "lsof", "mount"}
EXFIL_CMDS     = {"scp", "rsync", "sftp", "ftp"}
CLEANUP_CMDS   = {"shred", "wipe"}

KNOWN_ATTACK_BIGRAMS = {
    ("whoami", "sudo"), ("sudo", "su"),   ("id", "sudo"),
    ("sudo",   "cat"),  ("wget", "chmod"),("curl", "bash"),
    ("chmod",  "bash"), ("wget", "bash"), ("nmap", "ssh"),
    ("ssh",    "scp"),  ("cat",  "scp"),
}

KNOWN_ATTACK_TRIGRAMS = {
    ("whoami", "id",   "sudo"),
    ("id",     "sudo", "cat"),
    ("sudo",   "cat",  "scp"),
    ("wget",   "chmod","bash"),
    ("nmap",   "ssh",  "scp"),
}

# ── Pure Python stats helpers ─────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _std(values: List[float], mean: float = None) -> float:
    if len(values) < 2:
        return 0.0
    m = mean if mean is not None else _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))

# ── Parsing ───────────────────────────────────────────────────────────────────

def extract_base(raw: str) -> str:
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        return ""
    return raw.split()[0].lower()


def parse_history(text: str) -> Tuple[List[str], List[str]]:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    bases = [extract_base(l) for l in lines]
    bases = [b for b in bases if b]
    return bases, lines


def bucket(n: int) -> str:
    if n < MIN_COMMANDS:   return "tiny"
    if n < SHORT_THRESH:   return "short"
    if n < MEDIUM_THRESH:  return "medium"
    return "full"


def entropy_norm(freq: Counter) -> float:
    total = sum(freq.values())
    if total == 0:
        return 0.0
    ent = -sum((c / total) * math.log2(c / total) for c in freq.values() if c > 0)
    return ent / math.log2(len(freq)) if len(freq) > 1 else 0.0


def sequence_attack_score(bases: List[str]) -> float:
    """
    Ordered sliding-window pattern matching.
    Detects attacker workflows, not just keywords.
    Returns normalized score [0, 1].
    """
    if len(bases) < 3:
        return 0.0

    score = 0.0
    n     = len(bases)

    for i in range(n - 1):
        a, b = bases[i], bases[i + 1]
        if a in {"whoami", "id", "uname", "hostname"} and b in {"sudo", "su"}:
            score += 25
        if a in {"sudo", "su"} and b == "cat":
            score += 20
        if a == "nmap" and b == "ssh":
            score += 25
        if a == "ssh" and b in {"scp", "rsync", "sftp"}:
            score += 30
        if a in {"wget", "curl"} and b in {"chmod", "bash", "sh"}:
            score += 30
        if a in {"curl", "wget"} and b in {"curl", "wget"}:
            score += 10
        if b == "unset" and a == "history":
            score += 15
        if (a, b) in KNOWN_ATTACK_BIGRAMS:
            score += 10

    for i in range(n - 2):
        t = (bases[i], bases[i + 1], bases[i + 2])
        if t in KNOWN_ATTACK_TRIGRAMS:
            score += 15

    score = min(score, 200.0)
    return round(score / 200.0, 4)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(text: str) -> Dict:
    bases, full_cmds = parse_history(text)
    n = len(bases)

    zero = {k: 0.0 for k in FEATURE_NAMES}
    zero["_n"] = 0
    zero["_bucket"] = "tiny"
    if n == 0:
        return zero

    freq    = Counter(bases)
    priv_r  = sum(1 for c in bases if c in PRIVILEGE_CMDS) / n
    net_r   = sum(1 for c in bases if c in NETWORK_CMDS)   / n
    file_r  = sum(1 for c in bases if c in FILE_CMDS)      / n
    dev_r   = sum(1 for c in bases if c in DEV_CMDS)        / n
    recon_r = sum(1 for c in bases if c in RECON_CMDS)      / n
    exfil_r = sum(1 for c in bases if c in EXFIL_CMDS)      / n
    clean_r = sum(1 for c in bases if c in CLEANUP_CMDS)    / n

    has_histclean = float(any(
        "history -c" in l.lower() or "unset histfile" in l.lower()
        or "export histsize=0" in l.lower() or "shred" in l.lower()
        for l in full_cmds
    ))

    burst   = sum(1 for a, b in zip(bases, bases[1:]) if a == b) / max(1, n - 1)
    pipe_r  = sum(1 for c in full_cmds if "|"              in c) / n
    redir_r = sum(1 for c in full_cmds if re.search(r"[<>]", c)) / n
    top5    = sum(v for _, v in freq.most_common(5)) / n
    avg_len = sum(len(c) for c in full_cmds) / n
    seq_sc  = sequence_attack_score(bases)

    priv_net   = (priv_r + net_r) / 2.0
    recon_priv = recon_r * priv_r * 100

    return {
        "total_commands":        float(n),
        "unique_ratio":          len(freq) / n,
        "entropy_norm":          entropy_norm(freq),
        "sudo_rate":             priv_r,
        "network_rate":          net_r,
        "file_rate":             file_r,
        "dev_rate":              dev_r,
        "recon_rate":            recon_r,
        "exfil_rate":            exfil_r,
        "cleanup_rate":          clean_r + has_histclean * 0.1,
        "avg_cmd_length":        avg_len,
        "burst_score":           burst,
        "pipe_rate":             pipe_r,
        "redirect_rate":         redir_r,
        "top5_coverage":         top5,
        "priv_net_combo":        priv_net,
        "recon_priv_combo":      recon_priv,
        "session_size_log":      math.log1p(n),
        "sequence_attack_score": seq_sc,
        "cmd_diversity_index":   len(freq) / max(1, math.log1p(n)),
        "_n":                    n,
        "_bucket":               bucket(n),
    }


FEATURE_NAMES = [
    "total_commands",    "unique_ratio",       "entropy_norm",
    "sudo_rate",         "network_rate",        "file_rate",
    "dev_rate",          "recon_rate",          "exfil_rate",
    "cleanup_rate",      "avg_cmd_length",      "burst_score",
    "pipe_rate",         "redirect_rate",       "top5_coverage",
    "priv_net_combo",    "recon_priv_combo",    "session_size_log",
    "sequence_attack_score", "cmd_diversity_index",
]


def to_vector(features: Dict) -> List[float]:
    return [features.get(k, 0.0) for k in FEATURE_NAMES]
