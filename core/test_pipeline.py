from core.normalizer import normalize_command
from core.features import extract_features
from core.risk import compute_risk, classify_risk

def load_file(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# 🔹 NORMAL SESSION
normal = load_file("data/normal/normal_session_01.txt")
normal_norm = [normalize_command(c) for c in normal]
normal_feat = extract_features(normal_norm)


# 🔹 ATTACK SESSION
attack = load_file("data/attacks/attacker_session_01.txt")
attack_norm = [normalize_command(c) for c in attack]
attack_feat = extract_features(attack_norm)


# 🔹 PRINT RESULTS
print("\n====== NORMAL FEATURES ======")
for k, v in normal_feat.items():
    if not k.startswith("_"):
        print(f"{k}: {round(v, 3)}")

print("\n====== ATTACK FEATURES ======")
for k, v in attack_feat.items():
    if not k.startswith("_"):
        print(f"{k}: {round(v, 3)}")



normal_score = compute_risk(normal_feat)
attack_score = compute_risk(attack_feat)

print("\n====== RISK SCORES ======")
print("Normal Score:", normal_score, "→", classify_risk(normal_score))
print("Attack Score:", attack_score, "→", classify_risk(attack_score))
