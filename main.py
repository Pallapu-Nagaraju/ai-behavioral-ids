#!/usr/bin/env python3
"""
main.py — AI Command-Line DNA Intrusion Detection System v3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3-Layer Pure Python Engine: Statistical Anomaly + Sequence Scoring + Intent Heuristics
Optional 10%: Claude API SOC report (--ai flag only)

Usage:
  python main.py train                  Generate synthetic data + build profile
  python main.py train-self             Train on YOUR real ~/.bash_history
  python main.py analyze <file>         Analyze a session file
  python main.py analyze <file> --ai    Analyze + generate AI SOC report
  python main.py analyze --stdin        Read commands from stdin / pipe
  python main.py demo                   Full end-to-end demonstration
  python main.py profile                Show behavioral DNA profile
  python main.py batch <dir>            Batch analyze all .txt files
  python main.py evaluate               Run evaluation with confusion matrix
  python main.py monitor                Live monitoring via ~/.bash_history
  python main.py monitor --stdin        Live monitoring from stdin pipe
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_gen  import generate_all
from core.features  import extract_features, MIN_COMMANDS
from core.model     import train, predict, load_model, MODEL_DIR, PROFILE_PATH
from core.explainer import explain, format_report

BANNER = """
  DNA FINGERPRINT IDS v3 — Pure Python Behavioral Detection
  ──────────────────────────────────────────────────────────
  Engine  : Statistical Anomaly + Sequence Scoring + Intent Heuristics
  AI (10%): Claude API SOC report (--ai flag only)
  Deps    : stdlib only (json, math, re, collections)
"""
DIV = "=" * 65


def _banner():
    print(BANNER)
    print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {DIV}\n")


def _check_model():
    if not os.path.exists(PROFILE_PATH):
        print("[!] No profile found. Run: python main.py train")
        sys.exit(1)


def _save_report(result, explanation, prefix="report"):
    os.makedirs("reports", exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"reports/{prefix}_{ts}.json"
    with open(path, "w") as f:
        json.dump({"result": result, "explanation": explanation},
                  f, indent=2, default=str)
    return path


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_train(args):
    _banner()
    print("[*] Generating synthetic sessions (normal / attack / insider)...\n")
    sessions = generate_all(output_dir="data")
    print(f"\n  Generated {len(sessions)} sessions")
    print(f"\n{DIV}")
    print("[*] Building behavioral profile (pure Python statistics)...\n")
    stats = train(data_dir="data")
    print(f"\n[OK] Done — {stats['n_sessions']} sessions / {stats['features']} features")
    print(f"     Zero external dependencies used.")


def cmd_train_self(args):
    _banner()
    print("[*] Training on YOUR real bash history...\n")
    from core.train_self import train_on_history
    path = getattr(args, "history", None)
    stats = train_on_history(history_path=path, verbose=True)
    print(f"\n[OK] Trained on {stats.get('history_commands', 0)} real commands.")
    print("     Model now knows YOUR normal behavior.")
    print("     Run: python main.py monitor")


def cmd_analyze(args):
    _banner()
    _check_model()
    iso, scaler, profile, clf = load_model()

    use_ai = getattr(args, "ai", False)

    if getattr(args, "stdin", False):
        print("[*] Reading from stdin (Ctrl+D when done)...\n")
        raw = sys.stdin.read()
    else:
        if not getattr(args, "file", None):
            print("[!] Provide a file path or use --stdin")
            sys.exit(1)
        with open(args.file) as f:
            raw = f.read()
        print(f"[*] Analyzing: {args.file}\n")

    result = predict(raw, iso, scaler, profile, clf)
    exp    = explain(result, profile)

    ai_report = None
    if use_ai:
        print("[*] Requesting AI SOC report from Claude API...")
        from core.explainer import generate_soc_report
        ai_report = generate_soc_report(exp)

    print(format_report(exp, ai_report=ai_report))
    path = _save_report(result, exp)
    print(f"  [+] Report saved -> {path}\n")


def cmd_demo(args):
    _banner()
    print("[*] Running full demonstration...\n")

    generate_all(output_dir="data")
    print()
    train(data_dir="data")
    iso, scaler, profile, clf = load_model()

    TINY = "cd Documents\nls\ngit status\nnano file.txt\ncat file.txt"

    tests = [
        ("Normal Dev (large)",  "data/normal/normal_session_01.txt"),
        ("Normal Dev (medium)", "data/normal/normal_session_06.txt"),
        ("Normal Dev (short)",  "data/normal/normal_session_11.txt"),
        ("Tiny (5 cmds)",        None),
        ("Attacker Session #1", "data/attacks/attacker_session_01.txt"),
        ("Attacker Session #2", "data/attacks/attacker_session_02.txt"),
        ("Insider Threat",      "data/insider/insider_session_01.txt"),
    ]

    summary = []
    for name, path in tests:
        raw    = TINY if path is None else (open(path).read() if os.path.exists(path) else None)
        if raw is None:
            continue
        result = predict(raw, iso, scaler, profile, clf)
        exp    = explain(result, profile)
        print(f"\n{'-' * 65}")
        print(f"  TEST: {name}")
        print(format_report(exp, verbose=True))
        summary.append((name, result["n"], result["status"],
                         result["risk_level"], result["risk_score"],
                         result.get("intent", "-")))

    print(f"\n{'=' * 65}")
    print("  DEMO SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Session':<32} {'N':>5}  {'Status':<14} {'Level':<10}  {'Intent':<25}  Score")
    print(f"  {'-' * 90}")
    for name, n, status, rl, rs, intent in summary:
        tag = "OK" if status == "NORMAL" else ("--" if status == "INSUFFICIENT_DATA" else "!!")
        print(f"  [{tag}] {name:<30} {n:>5}  {status:<14} {rl:<10}  "
              f"{(intent or '-')[:24]:<25}  {rs:.1f}/100")
    print(f"{'=' * 65}\n")


def cmd_profile(args):
    _banner()
    if not os.path.exists(PROFILE_PATH):
        print("[!] No profile. Run: python main.py train")
        sys.exit(1)
    with open(PROFILE_PATH) as f:
        profile = json.load(f)
    meta = profile.pop("_meta", {})
    print(f"  Behavioral DNA Profile — "
          f"{meta.get('n_training_sessions', '?')} sessions\n")
    print(f"  {'Feature':<28} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-' * 68}")
    for feat, stats in profile.items():
        print(f"  {feat:<28} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
              f"{stats['min']:>10.4f} {stats['max']:>10.4f}")
    print()


def cmd_batch(args):
    _banner()
    _check_model()
    iso, scaler, profile, clf = load_model()
    import glob
    files = sorted(glob.glob(os.path.join(args.dir, "*.txt")))
    if not files:
        print(f"[!] No .txt files in {args.dir}")
        sys.exit(1)
    print(f"  Batch: {len(files)} files in {args.dir}/\n")
    print(f"  {'File':<38} {'N':>5}  {'Status':<14} {'Level':<10}  {'Intent':<25}  Score")
    print(f"  {'-' * 95}")
    for path in files:
        with open(path) as f:
            raw = f.read()
        result = predict(raw, iso, scaler, profile, clf)
        tag    = "OK" if result["status"] == "NORMAL" else ("--" if result["status"] == "INSUFFICIENT_DATA" else "!!")
        intent = (result.get("intent") or "-")[:24]
        print(f"  [{tag}] {os.path.basename(path):<36} {result['n']:>5}  "
              f"{result['status']:<14} {result['risk_level']:<10}  "
              f"{intent:<25}  {result['risk_score']:.1f}/100")
    print()


def cmd_evaluate(args):
    _banner()
    _check_model()
    from core.evaluate import run_evaluation
    run_evaluation()


def cmd_monitor(args):
    _banner()
    _check_model()
    from runtime.monitor import run_stream
    use_stdin = getattr(args, "stdin", False)
    run_stream(use_stdin=use_stdin)


# ── Argument parser ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DNA Fingerprint IDS v3 — Pure Python Behavioral Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("train",   help="Generate synthetic data + build profile")
    ts = sub.add_parser("train-self", help="Train on YOUR ~/.bash_history (recommended)")
    ts.add_argument("--history", default=None,
                    help="Path to history file (default: ~/.bash_history)")
    sub.add_parser("demo",    help="Full end-to-end demonstration")
    sub.add_parser("profile", help="Show behavioral DNA profile")
    sub.add_parser("evaluate",help="Run evaluation with confusion matrix")

    ap = sub.add_parser("analyze", help="Analyze a session file")
    ap.add_argument("file",    nargs="?",          help="Path to session file")
    ap.add_argument("--stdin", action="store_true", help="Read from stdin")
    ap.add_argument("--ai",    action="store_true",
                    help="Generate AI SOC report via Claude API (10%% feature)")

    bp = sub.add_parser("batch", help="Batch analyze a directory")
    bp.add_argument("dir", help="Directory of .txt session files")

    mp = sub.add_parser("monitor",
                        help="Live command monitoring (reads ~/.bash_history by default)")
    mp.add_argument("--stdin", action="store_true",
                    help="Read from stdin pipe instead of ~/.bash_history")

    args = parser.parse_args()

    dispatch = {
        "train":      cmd_train,
        "train-self": cmd_train_self,
        "demo":       cmd_demo,
        "analyze":    cmd_analyze,
        "profile":    cmd_profile,
        "batch":      cmd_batch,
        "evaluate":   cmd_evaluate,
        "monitor":    cmd_monitor,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        _banner()
        parser.print_help()


if __name__ == "__main__":
    main()
