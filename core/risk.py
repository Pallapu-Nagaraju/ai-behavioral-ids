def compute_scores(f):
    # -------------------
    # ATTACK SCORE
    # -------------------
    attack = 0
    attack += f["network_rate"] * 50
    attack += f["recon_rate"] * 40
    attack += f["exfil_rate"] * 40
    attack += f["sequence_attack_score"] * 50
    attack += f["sudo_rate"] * 20

    # Strong attack patterns
    if f["network_rate"] > 0.2 and f["recon_rate"] > 0.1:
        attack += 30

    if f["exfil_rate"] > 0.1 and f["sequence_attack_score"] > 0.2:
        attack += 30

    # -------------------
    # INSIDER SCORE
    # -------------------
    insider = 0

    # File-heavy abnormal usage
    if f["file_rate"] > 0.15 and f["dev_rate"] < 0.2:
        insider += 25

    # High diversity but low network
    if f["cmd_diversity_index"] > 5 and f["network_rate"] < 0.1:
        insider += 20

    # Suspicious sequence without network
    if f["sequence_attack_score"] > 0.1 and f["network_rate"] < 0.1:
        insider += 20

    # Moderate recon without strong attack
    if 0.05 < f["network_rate"] < 0.2 and f["recon_rate"] > 0.08:
        insider += 20

    # -------------------
    # NORMAL REDUCTION
    # -------------------
    # Only reduce if no strong attack
    if attack < 40:
        if f["dev_rate"] > 0.4:
            attack -= 20
            insider -= 10

        if f["entropy_norm"] > 0.8:
            attack -= 10
            insider -= 5

    return max(0, round(attack, 2)), max(0, round(insider, 2))


def classify(f):
    attack_score, insider_score = compute_scores(f)

    # 🔥 Priority-based decision (THIS FIXES YOUR PROBLEM)
    if attack_score >= 50:
        return "ATTACK", attack_score

    if insider_score >= 30:
        return "SUSPICIOUS", insider_score

    return "NORMAL", max(attack_score, insider_score)
