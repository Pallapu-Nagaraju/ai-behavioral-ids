"""
core/explainer.py — Human-Readable Alert Generation + Optional AI SOC Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detection : 100% pure Python statistical engine.
Explainer : Converts z-score deviations → structured security report.
AI (10%)  : Optional SOC narrative via Ollama local LLM (--ai flag only).
            Uses urllib only. Ollama must be running: ollama serve
            Model: llama3 (or whatever you have installed)
            Never used for scoring or detection.
"""

import json
import urllib.request
import urllib.error
from typing import Dict, List

FEATURE_LABELS = {
    "sudo_rate":             ("Privilege Escalation Rate",   "rate"),
    "network_rate":          ("Network Command Rate",        "rate"),
    "recon_rate":            ("Reconnaissance Rate",         "rate"),
    "exfil_rate":            ("Data Exfiltration Rate",      "rate"),
    "cleanup_rate":          ("Anti-Forensics Rate",         "rate"),
    "priv_net_combo":        ("Priv + Network Combo",        "combined"),
    "recon_priv_combo":      ("Recon x Privilege Signal",    "signal"),
    "entropy_norm":          ("Command Entropy",             "norm"),
    "unique_ratio":          ("Command Diversity",           "ratio"),
    "dev_rate":              ("Dev Activity Rate",           "rate"),
    "burst_score":           ("Burst / Automation Score",    "score"),
    "sequence_attack_score": ("Sequence Attack Pattern",     "score"),
    "pipe_rate":             ("Pipe Usage Rate",             "rate"),
    "redirect_rate":         ("I/O Redirect Rate",           "rate"),
    "top5_coverage":         ("Workflow Concentration",      "ratio"),
    "cmd_diversity_index":   ("Command Richness Index",      "index"),
}

NARRATIVES = {
    "sudo_rate":
        "Privilege escalation above baseline — credential abuse risk.",
    "network_rate":
        "Elevated network commands — possible lateral movement or C2 beacon.",
    "recon_rate":
        "System enumeration spike — classic initial-access attacker fingerprint.",
    "exfil_rate":
        "Exfiltration tools active (scp/rsync/sftp) — data theft risk.",
    "cleanup_rate":
        "Anti-forensics commands detected (history -c / shred) — evidence destruction.",
    "priv_net_combo":
        "Simultaneous privilege + network activity — high-confidence exfiltration signal.",
    "recon_priv_combo":
        "Recon combined with privilege escalation — credential abuse pattern.",
    "sequence_attack_score":
        "Known attack command sequences detected by ordered pattern analysis.",
    "entropy_norm":
        "Unusually diverse command set — attacker exploration behavior.",
    "unique_ratio":
        "High variety relative to session length — exploratory attacker pattern.",
    "dev_rate":
        "Developer tooling absent — atypical for this user profile.",
    "burst_score":
        "Repetitive command bursting — automated script or tool execution.",
}

ATTACK_SIGS = {
    "Privilege Escalation":  lambda d: d.get("sudo_rate", 0) > 2.5,
    "Data Exfiltration":     lambda d: (d.get("exfil_rate", 0) > 2.0
                                         or d.get("network_rate", 0) > 2.5),
    "System Reconnaissance": lambda d: d.get("recon_rate", 0) > 2.5,
    "Lateral Movement":      lambda d: d.get("priv_net_combo", 0) > 2.5,
    "Anti-Forensics":        lambda d: d.get("cleanup_rate", 0) > 2.0,
    "Credential Abuse":      lambda d: d.get("recon_priv_combo", 0) > 2.5,
    "Scripted Attack":       lambda d: d.get("sequence_attack_score", 0) > 1.5,
    "Developer Absence":     lambda d: d.get("dev_rate", 0) < -2.5,
}

COMPTIA = {
    "Privilege Escalation":  ["Security+ Domain 1.3 (Threats, Attacks)",
                               "CySA+ Domain 1.2 (Threat Intelligence)"],
    "Data Exfiltration":     ["Network+ Domain 2.1 (Protocols)",
                               "CySA+ Domain 2.1 (Threat Detection)"],
    "System Reconnaissance": ["Security+ Domain 1.1 (Recon)"],
    "Lateral Movement":      ["Security+ Domain 1.3", "CySA+ Domain 2.1"],
    "Anti-Forensics":        ["CySA+ Domain 3.1 (Incident Response)"],
    "Credential Abuse":      ["Security+ Domain 1.3"],
    "Scripted Attack":       ["CySA+ Domain 1.2"],
    "Developer Absence":     ["Security+ Domain 4.3 (Monitoring)"],
}

# ── Ollama configuration ──────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"   # change to "mistral" or "phi3" if that's what you have


# ── Core explain function ─────────────────────────────────────────────────────

def explain(result: Dict, profile: Dict) -> Dict:
    if result["status"] == "INSUFFICIENT_DATA":
        return {
            "status":         "INSUFFICIENT_DATA",
            "risk_level":     "UNKNOWN",
            "risk_score":     0.0,
            "summary":        result.get("message", "Session too short to analyze."),
            "indicators":     [],
            "attack_types":   [],
            "intent":         None,
            "intent_probs":   {},
            "comptia_refs":   ["Security+ Domain 4.3 (Logging & Monitoring)"],
            "recommendation": "Collect at least 10 commands before analysis.",
            "layers":         {},
            "bucket":         result.get("bucket", "tiny"),
            "n":              result.get("n", 0),
        }

    devs         = result["deviations"]
    features     = result["features"]
    status       = result["status"]
    risk_score   = result["risk_score"]
    bkt          = result.get("bucket", "full")
    n            = result.get("n", 0)
    intent       = result.get("intent")
    intent_probs = result.get("intent_probs", {})

    indicators   = _build_indicators(devs, features, profile, bkt)
    attack_types = [name for name, rule in ATTACK_SIGS.items() if rule(devs)]

    if status == "NORMAL":
        summary = (f"Behavior consistent with user profile "
                   f"({n} commands, {bkt}). Risk: {risk_score:.1f}/100.")
    else:
        top = ", ".join(i["label"] for i in indicators[:3]) or "behavioral deviation"
        summary = (f"Session deviates from baseline ({n} commands, {bkt}). "
                   f"Concerns: {top}. Risk: {risk_score:.1f}/100.")
        if intent and intent != "Normal":
            summary += f" Detected intent: {intent}."

    refs = set()
    for at in attack_types:
        for r in COMPTIA.get(at, []):
            refs.add(r)
    if not refs:
        refs.add("Security+ Domain 4.3 (Logging & Monitoring)")

    layers = {
        "stat_anomaly":  {"score": result.get("if_risk", 0.0),
                          "label": result.get("if_label", 1)},
        "sequence_ngram":{"score": result.get("seq_risk", 0.0),
                          "raw":   result.get("seq_score", 0.0)},
        "intent_engine": {"score": result.get("intent_risk", 0.0),
                          "intent": intent},
        "zscore":        {"score": result.get("z_risk", 0.0),
                          "flags": len(result.get("sig_devs", {}))},
    }

    return {
        "status":         status,
        "risk_level":     result["risk_level"],
        "risk_score":     risk_score,
        "summary":        summary,
        "indicators":     indicators,
        "attack_types":   attack_types,
        "intent":         intent,
        "intent_probs":   intent_probs,
        "comptia_refs":   sorted(refs),
        "recommendation": _recommend(result["risk_level"], attack_types),
        "layers":         layers,
        "bucket":         bkt,
        "n":              n,
    }


def _build_indicators(devs, features, profile, bkt):
    thresh = {"short": 3.5, "medium": 3.0}.get(bkt, 2.5)
    inds   = []
    for feat, z in devs.items():
        if abs(z) < thresh or feat not in FEATURE_LABELS:
            continue
        label, unit = FEATURE_LABELS[feat]
        actual   = features.get(feat, 0.0)
        baseline = profile.get(feat, {}).get("mean", 0.0)
        pct      = ((actual - baseline) / max(abs(baseline), 0.001)) * 100
        pct      = max(-9999.0, min(9999.0, pct))
        inds.append({
            "feature":   feat,
            "label":     label,
            "direction": "high" if z > 0 else "low",
            "z_score":   round(z, 2),
            "actual":    round(actual, 4),
            "baseline":  round(baseline, 4),
            "pct_change":round(pct, 1),
            "unit":      unit,
            "narrative": NARRATIVES.get(feat),
            "severity":  ("CRITICAL" if abs(z) > 4 else
                          "HIGH"     if abs(z) > 3 else "MEDIUM"),
        })
    return sorted(inds, key=lambda x: abs(x["z_score"]), reverse=True)


def _recommend(risk_level, attack_types):
    if risk_level in ("LOW", "UNKNOWN"):
        return "No action required. Continue routine monitoring."
    if risk_level == "MEDIUM":
        return "Review session logs manually. Verify user identity via secondary channel."
    if risk_level == "HIGH":
        rec = "Immediate review recommended."
        if "Privilege Escalation"  in attack_types: rec += " Audit sudo/su activity."
        if "Data Exfiltration"     in attack_types: rec += " Check outbound transfers."
        if "System Reconnaissance" in attack_types: rec += " Review enumeration commands."
        if "Anti-Forensics"        in attack_types: rec += " Preserve logs immediately."
        if "Lateral Movement"      in attack_types: rec += " Isolate host, check network."
        return rec.strip()
    return ("URGENT: Suspend session. Alert security team. "
            "Initiate incident response. Preserve all logs.")


# ── AI SOC Report via Ollama — 10% feature, local LLM, no API key needed ─────

def _check_ollama() -> str:
    """
    Check which models are available in Ollama.
    Returns the best available model name, or empty string if Ollama is down.
    """
    try:
        req  = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data   = json.loads(resp.read().decode())
            models = [m["name"].split(":")[0] for m in data.get("models", [])]
            # Priority: llama3 > mistral > phi3 > gemma > whatever is there
            for preferred in ["llama3", "mistral", "phi3", "gemma", "llama2"]:
                if preferred in models:
                    return preferred
            return models[0] if models else ""
    except Exception:
        return ""


def generate_soc_report(exp: Dict) -> str:
    """
    Calls local Ollama LLM to generate a SOC analyst narrative.

    Rules:
      - Local only — no internet, no API key, no cost
      - Called ONLY when user passes --ai flag
      - Uses urllib only — zero SDK dependencies
      - Detection already happened — this only explains in plain English
      - Fails gracefully if Ollama is not running

    Setup (one time):
        ollama serve          # start Ollama
        ollama pull llama3    # download model (~4GB, once)
    """
    # Auto-detect available model
    model = _check_ollama()
    if not model:
        return (
            "[AI SOC report unavailable]\n"
            "Ollama is not running. Start it with:\n"
            "  ollama serve\n"
            "  ollama pull llama3   (first time only)\n"
            "Then re-run with --ai flag."
        )

    findings = {
        "status":       exp["status"],
        "risk_score":   exp["risk_score"],
        "risk_level":   exp["risk_level"],
        "attack_types": exp["attack_types"],
        "intent":       exp.get("intent"),
        "top_indicators": [
            {
                "feature":   i["label"],
                "direction": i["direction"],
                "z_score":   i["z_score"],
                "narrative": i["narrative"],
            }
            for i in exp.get("indicators", [])[:5]
        ],
        "n_commands": exp.get("n"),
        "bucket":     exp.get("bucket"),
    }

    prompt = (
        "You are a SOC (Security Operations Center) analyst reviewing a behavioral anomaly alert "
        "from a command-line intrusion detection system.\n\n"
        "The detection engine has computed these findings:\n"
        + json.dumps(findings, indent=2)
        + "\n\nWrite a concise professional SOC analyst report (4-6 sentences) that:\n"
        "1. States whether this is a real threat or likely false positive\n"
        "2. Explains what the behavioral evidence means in plain English\n"
        "3. Recommends the single most important next action\n\n"
        "Be direct. No bullet points. Write as a senior analyst in an incident ticket."
    )

    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            text = data.get("response", "").strip()
            if text:
                return f"[Model: {model}]\n{text}"
            return "[AI SOC report: empty response from Ollama]"

    except urllib.error.URLError:
        return (
            "[AI SOC report unavailable]\n"
            "Cannot connect to Ollama. Make sure it is running:\n"
            "  ollama serve"
        )
    except Exception as e:
        return f"[AI SOC report error: {e}]"


# ── CLI report formatter ──────────────────────────────────────────────────────

def format_report(exp: Dict, verbose: bool = True,
                  ai_report: str = None) -> str:
    lines = []
    sep   = "-" * 70
    icons = {
        "NORMAL":            "[OK]",
        "SUSPICIOUS":        "[!!]",
        "ANOMALOUS":         "[!!]",
        "INSUFFICIENT_DATA": "[--]",
    }
    icon = icons.get(exp["status"], "[?]")

    lines += [
        "", sep,
        f"  {icon}  SESSION ANALYSIS REPORT", sep,
        f"  Status      : {exp['status']}",
        f"  Risk Level  : {exp['risk_level']}",
        f"  Risk Score  : {exp['risk_score']:.1f} / 100",
        f"  Session     : {exp.get('n','?')} commands ({exp.get('bucket','?')})",
    ]

    layers = exp.get("layers", {})
    if layers and verbose:
        lines += ["", "  Detection Layer Scores:"]
        layer_defs = [
            ("stat_anomaly",  "Statistical Anomaly  "),
            ("sequence_ngram","Sequence Patterns    "),
            ("intent_engine", "Intent Engine        "),
            ("zscore",        "Z-Score Deviations   "),
        ]
        for key, name in layer_defs:
            lyr   = layers.get(key, {})
            score = lyr.get("score", 0.0)
            bars  = int(score / 5)
            bar   = "#" * bars + "." * (20 - bars)
            extra = ""
            if key == "intent_engine" and lyr.get("intent"):
                extra = f" -> {lyr['intent']}"
            lines.append(f"    {name} [{bar}] {score:5.1f}/100{extra}")

    lines += ["", "  Summary:", f"    {exp['summary']}"]

    if exp.get("attack_types") and verbose:
        lines += ["", "  Detected Attack Patterns:"]
        for at in exp["attack_types"]:
            lines.append(f"    >> {at}")

    if exp.get("intent") and exp["intent"] != "Normal" and verbose:
        lines.append(f"\n  Detected Intent: {exp['intent']}")

    if exp.get("indicators") and verbose:
        lines += ["", "  Behavioral Indicators (z-score deviations from baseline):"]
        for ind in exp["indicators"]:
            arrow = "^" if ind["direction"] == "high" else "v"
            lines.append(
                f"    [{ind['severity']:8s}] {ind['label']:35s} "
                f"{arrow} {abs(ind['pct_change']):.0f}% (z={ind['z_score']:+.1f})"
            )
            if ind["narrative"]:
                lines.append(f"               -> {ind['narrative']}")

    if exp.get("comptia_refs") and verbose:
        lines += ["", "  CompTIA References:"]
        for ref in exp["comptia_refs"]:
            lines.append(f"    [Ref] {ref}")

    lines += ["", "  Recommendation:", f"    {exp['recommendation']}"]

    # AI SOC report — Ollama local LLM (10% feature)
    if ai_report:
        lines += [
            "",
            "  " + "-" * 66,
            "  AI SOC ANALYST REPORT (Ollama Local LLM)",
            "  " + "-" * 66,
        ]
        for line in ai_report.split("\n"):
            if line.strip():
                lines.append(f"  {line.strip()}")

    lines += [sep, ""]
    return "\n".join(lines)
