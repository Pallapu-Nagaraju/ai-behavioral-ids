"""
core/evaluate.py — Pure Python Evaluation Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zero external dependencies.
Metrics computed with stdlib only: no numpy, no sklearn metrics.

Reports:
  - Per-session prediction table
  - Confusion matrix
  - Accuracy / FPR / FNR / Precision / Recall / F1
  - Honest diagnosis

Usage:
  python main.py evaluate
  PYTHONPATH=. python core/evaluate.py
"""

import os
import sys
import time
from typing import Dict, List, Tuple
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import load_model, predict

# Which predictions count as "correct" for each true label
CORRECT_PREDS = {
    "NORMAL":  {"NORMAL"},
    "ATTACK":  {"ANOMALOUS", "SUSPICIOUS"},
    "INSIDER": {"SUSPICIOUS", "ANOMALOUS"},
}


def load_folder(folder: str):
    """Yield (filename, raw_text) for every .txt file in folder."""
    if not os.path.exists(folder):
        return
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(folder, fname)
        with open(path) as f:
            raw = f.read()
        yield fname, raw


def run_evaluation(data_dir: str = "data", verbose: bool = True) -> Dict:
    """
    Load profile, run against all labeled sessions, return metrics.
    All metrics computed in pure Python — no numpy/sklearn.
    """
    iso, scaler, profile, clf = load_model()

    rows     = []    # (filename, true_label, predicted_status, risk_score, latency_ms)
    folders  = {
        "NORMAL":  os.path.join(data_dir, "normal"),
        "ATTACK":  os.path.join(data_dir, "attacks"),
        "INSIDER": os.path.join(data_dir, "insider"),
    }

    for true_label, folder in folders.items():
        for fname, raw in load_folder(folder):
            t0     = time.perf_counter()
            result = predict(raw, iso, scaler, profile, clf)
            ms     = (time.perf_counter() - t0) * 1000
            rows.append((fname, true_label, result["status"],
                          result["risk_score"], ms))

    if not rows:
        print("[!] No evaluation data. Run: python main.py train")
        return {}

    # ── Per-session table ─────────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 80)
        print("  EVALUATION RESULTS")
        print("=" * 80)
        print(f"  {'File':<35}  {'True':^10}  {'Predicted':^14}  {'Score':>6}  {'ms':>6}  OK?")
        print("  " + "-" * 78)

    correct = 0
    for fname, tl, pred, score, ms in rows:
        hit  = pred in CORRECT_PREDS.get(tl, set())
        if hit:
            correct += 1
        mark = "OK" if hit else "MISS"
        if verbose:
            print(f"  {fname:<35}  {tl:^10}  {pred:^14}  {score:>6.1f}"
                  f"  {ms:>5.1f}  {mark}")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    accuracy = correct / len(rows) if rows else 0.0
    if verbose:
        print(f"\n  Accuracy: {correct}/{len(rows)} = {accuracy * 100:.1f}%")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    true_classes = ["NORMAL", "ATTACK", "INSIDER"]
    pred_classes = ["NORMAL", "SUSPICIOUS", "ANOMALOUS", "INSUFFICIENT_DATA"]
    cm = {t: {p: 0 for p in pred_classes} for t in true_classes}
    for _, tl, pred, _, _ in rows:
        cm[tl][pred] = cm[tl].get(pred, 0) + 1

    if verbose:
        print("\n  CONFUSION MATRIX")
        print(f"  {'':14}", end="")
        for p in pred_classes:
            print(f"  {p[:12]:^13}", end="")
        print()
        print("  " + "-" * 68)
        for tl in true_classes:
            print(f"  {tl:<14}", end="")
            for p in pred_classes:
                print(f"  {cm[tl].get(p, 0):^13}", end="")
            print()

    # ── FPR / FNR / Precision / Recall / F1 — pure Python ────────────────────
    normal_rows = [r for r in rows if r[1] == "NORMAL"]
    attack_rows = [r for r in rows if r[1] in ("ATTACK", "INSIDER")]

    tp  = sum(1 for _, tl, pred, _, _ in attack_rows if pred in {"SUSPICIOUS", "ANOMALOUS"})
    tn  = sum(1 for _, tl, pred, _, _ in normal_rows  if pred == "NORMAL")
    fp  = sum(1 for _, tl, pred, _, _ in normal_rows  if pred in {"SUSPICIOUS", "ANOMALOUS"})
    fn  = sum(1 for _, tl, pred, _, _ in attack_rows  if pred == "NORMAL")

    fpr       = fp / max(len(normal_rows), 1)
    fnr       = fn / max(len(attack_rows), 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = (2 * precision * recall / max(precision + recall, 1e-9))

    latencies  = [r[4] for r in rows]
    avg_ms     = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_lat = sorted(latencies)
    p99_ms     = sorted_lat[int(0.99 * len(sorted_lat))] if sorted_lat else 0.0

    if verbose:
        print(f"\n  False Positive Rate : {fp}/{len(normal_rows)} = {fpr * 100:.1f}%"
              f"  (alerts on normal sessions)")
        print(f"  False Negative Rate : {fn}/{len(attack_rows)} = {fnr * 100:.1f}%"
              f"  (missed attacks)")
        print(f"  Precision           : {precision * 100:.1f}%"
              f"  (of alerts, how many were real)")
        print(f"  Recall / TPR        : {recall * 100:.1f}%"
              f"  (of attacks, how many caught)")
        print(f"  F1 Score            : {f1:.3f}")
        print(f"\n  Avg latency         : {avg_ms:.2f} ms/session")
        print(f"  P99 latency         : {p99_ms:.2f} ms/session")

        # Honest diagnosis
        print("\n  Diagnosis:")
        if recall >= 0.9 and fpr <= 0.1:
            print("  [OK] Strong performance. Ready for real testing.")
        elif recall >= 0.7:
            print("  [!!] Decent detection. High FPR — add more training data.")
        else:
            print("  [!!] Low detection. Model undertrained. Add more sessions.")
        if fpr > 0.3:
            print("  [!!] High false alarm rate. Analysts will ignore alerts.")
        print()

    return {
        "accuracy":       round(accuracy, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "false_pos_rate": round(fpr, 4),
        "false_neg_rate": round(fnr, 4),
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "f1":             round(f1, 4),
        "avg_ms":         round(avg_ms, 3),
        "p99_ms":         round(p99_ms, 3),
        "n_total":        len(rows),
        "confusion":      cm,
    }


if __name__ == "__main__":
    run_evaluation()
