<div align="center">

# 🧬 AI Behavioral IDS
## Command-Line DNA Fingerprinting System

**Continuous behavioral identity verification for Linux shell sessions**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Kali%20%7C%20WSL-0D1117?style=for-the-badge&logo=linux&logoColor=white)](https://kali.org)
[![AI](https://img.shields.io/badge/AI-Behavioral%20Anomaly%20%2B%20Intent%20Detection-FF6B35?style=for-the-badge)]()
[![Pure Python ML](https://img.shields.io/badge/No%20External%20ML%20Libraries-Pure%20Python-22C55E?style=for-the-badge)]()

> *"Traditional IDS checks if you have a valid key. This system checks if you walk like the person who owns that key."*

</div>

---

## 📌 The Problem

Authentication ends at login. After that, the system blindly trusts whoever is at the keyboard — valid credentials, valid session, zero behavioral validation.

This system solves it by building a **behavioral DNA fingerprint** per user and continuously asking:
> *"Does this command sequence match how this specific user normally behaves?"*

| What it detects | How |
|---|---|
| Stolen credentials in use | Behavioral mismatch from learned profile |
| Insider threat / misuse | Deviation from personal baseline |
| Living-off-the-land attacks | Sequence scoring + z-score deviation |
| Reconnaissance activity | Intent classifier + recon feature rate |

---

## 🤖 The AI Engine

This is a **4-component hybrid scoring system**. Each component contributes a scored signal that is fused into a final risk score.

---

### Layer 1 — Statistical Anomaly Detection

**What it does:** Learns the statistical shape of a user's normal behavior in 20-dimensional feature space. Sessions that fall outside that shape score as anomalous.

**Why this approach:**
- Works unsupervised — no labeled attack data needed to train
- Handles high-dimensional sparse behavioral data well
- Fast at inference time
- Not dominated by the majority class (unlike kNN or SVM)

**How it works:**

The anomaly engine compares each session's behavioral vector against the user's learned normal baseline using statistical deviation scoring.

Sessions with unusual command frequency, elevated reconnaissance activity, privilege usage, or abnormal sequence patterns receive higher anomaly scores. The raw score is remapped to a 0–100 risk contribution:

```
if_risk = max(0, min(100, (raw_score - 0.4) * 250))
```

**Implementation:** Built entirely in pure Python — no external ML libraries required.

---

### Layer 2 — N-gram Sequence Engine (Pattern Matching)

**What it does:** Detects known attack command sequences using bigram and trigram matching.

**Why N-grams:**
- Statistical models treat each command independently — they miss *order*
- Attacks follow predictable workflows: `whoami → sudo → scp`
- N-grams capture that sequential structure
- No training data needed — uses a fixed attack sequence library

**How it works:**
```python
# Bigrams extracted from session commands
session  = ["whoami", "sudo", "scp"]
bigrams  = [("whoami", "sudo"), ("sudo", "scp")]

# Matched against known attack patterns:
KNOWN_ATTACK_BIGRAMS = {
    ("whoami", "sudo"), ("wget", "chmod"),
    ("curl", "bash"), ("nmap", "ssh"), ...
}

sequence_attack_score = matched_bigrams / normalization_factor
seq_risk = min(100, sequence_attack_score * 100)
```

**Hard floor:** If `seq_score >= 0.30`, risk is floored at 55 regardless of other layers.

---

### Layer 3 — Intent Classifier

**What it does:** Classifies a session into one of 7 intent categories — not just "anomalous" but *what kind* of behavior it represents.

**Intent categories:**
```
0 → Normal
1 → Privilege Escalation
2 → Data Exfiltration
3 → System Reconnaissance
4 → Lateral Movement
5 → Persistence / Backdoor
6 → Anti-Forensics
```

**Why this approach:**
- Outputs class probabilities via a softmax function implemented in pure Python
- Interpretable: class weights explain what drove the prediction

- Implementable in pure Python without external libraries

**Training data:**
```python
# Per-intent synthetic sessions from curated command pools
# Each intent class: ~14 labeled sessions
# Plus 10 insider threat sessions (mixed normal + attack)
# Total: ~100 labeled training samples across 7 classes
```

**Intent risk signal:**
```python
intent_risk = (1.0 - P(Normal)) * 100
# P(Normal) = 0.3  →  intent_risk = 70
# P(Normal) = 0.9  →  intent_risk = 10
```

---

### Layer 4 — Z-Score Profile Deviation

**What it does:** Measures how far each behavioral feature deviates from the user's personal learned baseline using z-scores.

**Two-tier feature separation:**
```python
# CRITICAL — strong security signals, flagged at standard threshold
CRITICAL_FEATURES = [
    "sudo_rate", "network_rate", "recon_rate",
    "exfil_rate", "cleanup_rate",
    "priv_net_combo", "recon_priv_combo"
]

# BEHAVIORAL — noisier signals, require z_thresh + 1.0 to flag
BEHAVIORAL_FEATURES = [
    "entropy_norm", "unique_ratio", "cmd_diversity_index"
]

# Entropy dominance guard:
# If entropy_norm is the only flagged feature → clear it
# Single entropy spike alone does not trigger an alert
```

**Z-score formula:**
```
z = (observed_value - profile_mean) / profile_std

Threshold by session size:
  full session   → z_thresh = 2.5
  medium session → z_thresh = 3.0
  short session  → z_thresh = 3.5
```

Profile baselines (mean and std per feature) are generated from the user's normal command history and stored in a per-user JSON profile.

---

### Risk Fusion Formula

All four layers combine into a single risk score. The weighting adapts based on how many critical features are flagged:

```python
# When 2 or more critical features are flagged (strong evidence):
risk = 0.30*if_risk + 0.25*seq_risk + 0.20*intent_risk + 0.25*z_risk

# When fewer than 2 features are flagged (weak/no z-score evidence):
risk = 0.45*if_risk + 0.25*seq_risk + 0.20*intent_risk + 0.10*z_risk
```

**Signal overrides (applied after fusion):**
```python
# Strong attack indicators override low fusion scores
if "exfil_rate" in sig_devs or "cleanup_rate" in sig_devs:
    risk = max(risk, 62.0)

if seq_score >= 0.30:
    risk = max(risk, 55.0)

# Normal override: IF says normal + no z-flags + low seq score
if if_label == 1 and not sig_devs and if_raw < 0.5 and seq_score < 0.05:
    risk = min(risk, 22.0)

# Final clamp
risk = round(min(100.0, max(0.0, risk)), 1)
```

**Status thresholds:**
```
risk < 34   →  NORMAL     (LOW)
34 – 50     →  SUSPICIOUS (MEDIUM)
≥ 50        →  ANOMALOUS  (HIGH)
```

---

## 🔬 Feature Engineering — 20 Behavioral Dimensions

Every session is converted to a 20-dimensional behavioral vector. All features are **rate-based** (count divided by total commands) so short and long sessions are directly comparable.

| # | Feature | Formula | What It Captures |
|---|---------|---------|-----------------|
| 1 | `sudo_rate` | `sudo_cmds / total` | Privilege escalation frequency |
| 2 | `network_rate` | `net_cmds / total` | Network tool usage |
| 3 | `recon_rate` | `recon_cmds / total` | System enumeration activity |
| 4 | `exfil_rate` | `exfil_cmds / total` | Data movement tools |
| 5 | `cleanup_rate` | `cleanup_cmds / total` | Evidence destruction |
| 6 | `file_rate` | `file_cmds / total` | Filesystem navigation |
| 7 | `dev_rate` | `dev_cmds / total` | Development activity |
| 8 | `priv_net_combo` | `(sudo_r + net_r) / 2` | Privilege + network combined |
| 9 | `recon_priv_combo` | `recon_r × sudo_r × 100` | Recon–privilege interaction |
| 10 | `sequence_attack_score` | `attack_bigrams / norm` | Known attack sequences |
| 11 | `entropy_norm` | `H(freq) / log2(vocab)` | Command vocabulary diversity |
| 12 | `unique_ratio` | `unique / total` | Command repetition pattern |
| 13 | `cmd_diversity_index` | `unique / log1p(total)` | Richness relative to size |
| 14 | `burst_score` | `consecutive_repeats / (n-1)` | Automated/scripted behavior |
| 15 | `pipe_rate` | `pipe_cmds / total` | Command chaining |
| 16 | `redirect_rate` | `redirect_cmds / total` | I/O redirection |
| 17 | `top5_coverage` | `top5_count / total` | Workflow concentration |
| 18 | `avg_cmd_length` | `sum(len) / total` | Command complexity |
| 19 | `session_size_log` | `log1p(n)` | Session volume (outlier-safe) |
| 20 | `total_commands` | raw count | Session size |

**Entropy formula:**
```
H = -Σ (p_i × log2(p_i))        for each unique command
entropy_norm = H / log2(vocab)   bounded to [0, 1]
```

---

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                ENDPOINT (Linux / WSL / Kali)              │
│                                                           │
│  ~/.bashrc → PROMPT_COMMAND hook → monitor.py             │
│                      ↓                                    │
│          session buffer (rolling N commands)              │
│                      ↓                                    │
│          core/features.py → 20-dim vector                 │
└──────────────────────┬────────────────────────────────────┘
                       │
       ┌───────────────▼───────────────┐
       │         core/model.py          │
       │                               │
       │  Statistical Anomaly Layer [ANOM]│
       │  N-gram Sequences      [seq]  │
       │  Intent Classifier     [int]  │
       │  Z-Score Deviation     [z]    │
       │                               │
       │  Risk Fusion → score (0–100)  │
       └───────────────┬───────────────┘
                       │
       ┌───────────────▼───────────────┐
       │  core/risk.py                 │
       │  Attack score + Insider score │
       └───────────────┬───────────────┘
                       │
       ┌───────────────▼───────────────┐
       │  core/explainer.py            │
       │  Indicators + Alert + Report  │
       └───────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Component | Technology | Why This Choice |
|-----------|-----------|----------------|
| Anomaly detection | Statistical anomaly detection (pure Python) | Unsupervised, no labeled attack data needed |
| Intent classification | Intent classifier (pure Python) | Confidence-based intent scoring, interpretable weights |
| Sequence matching | N-gram engine (custom) | Captures ordered command workflows |
| Baseline profiling | Z-score per-feature deviation | Per-user behavioral baseline comparison |
| Shell monitoring | `PROMPT_COMMAND` hook | Zero-latency, no polling — behavioral analysis runs during the active session, not after |
| SOC reports | Ollama (local LLM) | On-device, no data sent externally |
| Storage | JSON + Pickle | No database dependency |

> **Zero external ML dependencies.** All AI components — anomaly detection, intent classification, sequence scoring, and behavioral profiling — are implemented from scratch using only Python stdlib: `math`, `random`, `json`, `pickle`, `os`.

---

## 📁 Project Structure

```
dna_ids/
├── main.py                  # CLI entry point
├── monitor.py               # Real-time PROMPT_COMMAND hook
│
├── core/
│   ├── features.py          # 20-feature behavioral extractor
│   ├── model.py             # 4-layer AI engine + risk fusion
│   ├── risk.py              # Attack/insider heuristic scorer
│   ├── explainer.py         # Alert explainer + CompTIA mapping
│   ├── data_gen.py          # Synthetic session generator
│   └── realtime.py          # Session buffer manager
│
├── data/
│   ├── normal/              # Training sessions (real bash history)
│   └── attacks/             # Attack session test files
│
├── models/                  # Trained .pkl model files
├── profiles/                # Per-user JSON behavioral profiles
└── reports/                 # Generated JSON incident reports
```

---

## 🚀 Quick Start

```bash
# 1. No external ML dependencies required — pure Python only
# (Python 3.10+ standard library is sufficient)

# 2. Use your real shell history as training data
cp ~/.bash_history data/normal/normal_session_01.txt

# 3. Train all AI models
python main.py train

# 4. Start real-time monitoring
python main.py monitor

# 5. Analyze a specific session file
python main.py analyze data/attacks/attacker_session_01.txt

# 6. Hook into your shell permanently
echo 'PROMPT_COMMAND="history -a; tail -n1 ~/.bash_history | python /path/to/main.py analyze --stdin"' >> ~/.bashrc
source ~/.bashrc
```

**Training output:**
```
  [+] normal_session_01.txt    n= 312   bucket=full
  [✓] Anomaly model      → models/anomaly_model.pkl
  [✓] Intent Classifier  → models/intent_clf.pkl    (100 samples / 7 classes)
  [✓] Profile            → profiles/profile_default.json
  [✓] 3 sessions / 20 features
```

---

## 📊 Live Output Examples

**Normal session:**
```
[09:37:35] [dashakanta] $ ls
  -> NORMAL       Score:  4.2/100  Intent: Normal             [session-model]

[09:37:39] [dashakanta] $ git status
  -> NORMAL       Score:  3.8/100  Intent: Normal             [session-model]
```

**Reconnaissance detected:**
```
[09:41:12] [dashakanta] $ nmap -sV 10.0.0.1
  -> SUSPICIOUS   Score: 36.0/100  Intent: System Reconnaissance  [session-model]
```

**Full attack session:**
```
[09:45:03] [dashakanta] $ cat /etc/shadow
  -> ANOMALOUS    Score: 74.2/100  Intent: Privilege Escalation  [session-model]

  ⚠  Behavioral Indicators:
     [CRITICAL] Privilege Escalation Rate  ↑ 720% from baseline  (z=+5.1)
     [CRITICAL] Data Exfiltration Rate     ↑ 952% from baseline  (z=+6.1)
     [HIGH]     Sequence Attack Pattern    ↑ 532% from baseline  (z=+3.7)

  📘 Security+ Domain 1.3 | CySA+ Domain 1.2 | Network+ Domain 2.1
  🔴 URGENT: Suspend session. Alert security team. Initiate IR.
```

---

## 🎯 Detection Capabilities

| Behavior | Expected Score | Status |
|----------|---------------|--------|
| Normal dev commands (ls, git, pip) | 2 – 15 | `NORMAL` |
| Occasional sudo | 10 – 30 | `NORMAL` |
| nmap scan | 20 – 55 | `SUSPICIOUS` / `ANOMALOUS` |
| `sudo -l` reconnaissance | 25 – 55 | `SUSPICIOUS` |
| `/etc/shadow` access + exfil | 62 – 90 | `ANOMALOUS` |
| `history -c` + sequence attack | 55 – 100 | `ANOMALOUS` |
| Insider threat (mixed session) | 25 – 45 | `SUSPICIOUS` |

> Score varies by user profile — a Kali pentester who uses `nmap` daily scores lower than a developer who never does. That is the adaptive profiling working correctly.

---

## 👥 Multi-User Profiles

```bash
python main.py train --user alice --data data/alice/
python main.py train --user bob   --data data/bob/
python main.py analyze session.txt --user alice
```

Each user gets independent model files:
```
profiles/profile_alice.json   ← alice's behavioral baseline
models/model_alice.pkl        ← alice's anomaly model
models/intent_alice.pkl       ← alice's Intent Classifier
```

---

## 🧠 Why This Is AI

This project applies AI concepts through behavioral learning, anomaly detection, statistical profiling, sequence analysis, and intent classification.

Instead of relying on fixed attack signatures alone, the system **learns behavioral patterns** from historical command usage and dynamically compares future activity against that learned baseline.

The project combines:

- **Behavioral profiling** — builds a personal behavioral fingerprint per user from real command history
- **Statistical anomaly detection** — scores sessions against the learned normal distribution
- **Intent classification** — predicts *what category* of behavior a session represents, not just *whether* it is anomalous
- **Pattern sequence analysis** — detects ordered attack workflows using bigram and trigram matching
- **Adaptive per-user scoring** — the same command scores differently depending on who runs it and what their baseline looks like

This allows the system to detect suspicious behavior even when no exact attack signature exists — the core limitation of rule-based and signature-based IDS systems.

---

## ⚠️ Honest Limitations

| Limitation | Impact |
|-----------|--------|
| Trained on synthetic / small real data | Model may not generalize to all users |
| bash history only, no kernel visibility | Direct syscalls are invisible |
| No temporal modeling across sessions | Slow multi-day attacks not tracked |
| Behavioral overlap with insider threats | Careful insiders can score low |
| No network log correlation | Cannot cross-reference firewall or DNS |

> **Academic positioning:** This is a research prototype demonstrating continuous behavioral authentication. It is a complement to signature-based detection, not a replacement. The goal is explainable behavioral profiling, not perfect detection accuracy.

---

## 🔮 Future Work

| Upgrade | What It Adds |
|---------|-------------|
| LSTM / RNN sequence modeling | Long-range attack pattern memory across sessions |
| Reinforcement learning | Model self-updates as user behavior evolves |
| MITRE ATT&CK mapping | Intent label → standard technique ID |
| SIEM integration (Splunk / Elastic) | Enterprise deployment via webhook |
| auditd backend | Kernel-level monitoring, harder to bypass |
| Live web dashboard | Real-time SOC visualization |

---

## 📚 References

- Chandola, V., Banerjee, A., Kumar, V. (2009). **Anomaly Detection: A Survey.** *ACM Computing Surveys.*
- Yampolskiy, R.V., Govindaraju, V. (2008). **Behavioural biometrics: a survey and classification.**
- Gartner (2015). **Market Guide for User and Entity Behavior Analytics (UEBA).**
- MITRE ATT&CK Framework — [attack.mitre.org](https://attack.mitre.org)
- CompTIA CySA+ CS0-003 — Domain 1 (Threat Detection), Domain 2 (Vulnerability Management)

---

## 👤 Author

**Dashakanta** · Final Year Cybersecurity Project

> *Built to show that behavioral AI can verify identity continuously — not just at login, but on every command.*

---

<div align="center">

*Pure Python · Behavioral Profiling · 4-Layer Hybrid AI · Real-Time Shell Monitoring*

</div>
