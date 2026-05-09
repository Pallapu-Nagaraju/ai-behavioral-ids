"""
core/train_self.py — Train on YOUR Real Command History
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zero external libraries. Pure Python stdlib only.
Reads from: ~/.bash_history, ~/.zsh_history, fish, Windows PowerShell.
"""

import os, sys, json, math, subprocess
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.features      import extract_features, FEATURE_NAMES, MIN_COMMANDS, to_vector
from runtime.user_manager import save_user_profile

HISTORY_CHUNK_SIZE = 50


def _get_windows_username() -> Optional[str]:
    try:
        r = subprocess.run(["cmd.exe","/c","echo %USERNAME%"],
                           capture_output=True, text=True, timeout=3)
        n = r.stdout.strip()
        if n and "%" not in n:
            return n
    except Exception:
        pass
    try:
        users_dir = "/mnt/c/Users"
        if os.path.exists(users_dir):
            skip = {"Public","Default","Default User","All Users"}
            best, best_size = None, 0
            for d in os.listdir(users_dir):
                if d in skip: continue
                p = (f"/mnt/c/Users/{d}/AppData/Roaming/Microsoft/"
                     f"Windows/PowerShell/PSReadLine/ConsoleHost_history.txt")
                if os.path.exists(p):
                    s = os.path.getsize(p)
                    if s > best_size:
                        best_size, best = s, d
            return best
    except Exception:
        pass
    return None


def find_history_files() -> List[Dict]:
    candidates = [
        {"path": os.path.expanduser("~/.bash_history"),  "source": "bash"},
        {"path": os.path.expanduser("~/.zsh_history"),   "source": "zsh"},
        {"path": os.path.expanduser("~/.local/share/fish/fish_history"), "source": "fish"},
    ]
    win_user = _get_windows_username()
    if win_user:
        candidates.append({
            "path": (f"/mnt/c/Users/{win_user}/AppData/Roaming/Microsoft/"
                     f"Windows/PowerShell/PSReadLine/ConsoleHost_history.txt"),
            "source": "powershell"
        })
    appdata = os.getenv("APPDATA")
    if appdata:
        candidates.append({
            "path": os.path.join(appdata,"Microsoft","Windows","PowerShell",
                                 "PSReadLine","ConsoleHost_history.txt"),
            "source": "powershell"
        })
    seen, out = set(), []
    for c in candidates:
        p = c["path"]
        if p in seen or not os.path.exists(p): continue
        seen.add(p)
        c["size_kb"] = round(os.path.getsize(p)/1024, 1)
        out.append(c)
    return out


def read_history_file(path: str, source: str) -> List[str]:
    try:
        with open(path, errors="ignore", encoding="utf-8") as f:
            raw = f.read()
    except Exception:
        return []
    cmds = []
    for line in raw.splitlines():
        line = line.strip()
        if not line: continue
        if source == "bash"       and line.startswith("#"): continue
        if source == "zsh"        and line.startswith(": ") and ";" in line:
            line = line.split(";",1)[-1].strip()
        if source == "fish":
            if line.startswith("- cmd:"): line = line[6:].strip()
            elif line.startswith("when:") or line.startswith("paths:"): continue
        if source == "powershell" and line.startswith("#"): continue
        if line: cmds.append(line)
    return cmds


def chunk_history(cmds: List[str], chunk_size: int = HISTORY_CHUNK_SIZE) -> List[str]:
    if not cmds: return []
    chunks, step = [], max(1, chunk_size // 2)
    for i in range(0, len(cmds) - chunk_size + 1, step):
        chunks.append("\n".join(cmds[i:i+chunk_size]))
    remainder = cmds[-(chunk_size//2):]
    if len(remainder) >= MIN_COMMANDS:
        chunks.append("\n".join(remainder))
    return chunks


def train_on_history(history_path: str = None, verbose: bool = True) -> Dict:
    """
    Pure Python training on real shell history.
    No numpy. No sklearn. Uses core/model.py pure implementations.
    """
    import pickle
    from core.model import (RobustScaler, IsolationForest,
                             _train_intent_clf, _MIN_STD,
                             MODEL_DIR, IF_PATH, SCALER_PATH,
                             PROFILE_PATH, INTENT_PATH)
    from core.features import _mean, _std  # pure python helpers

    user = (os.getenv("USER") or os.getenv("USERNAME") or "user").strip()
    all_cmds: List[str] = []
    sources_used: List[str] = []

    if history_path:
        src  = "powershell" if "ConsoleHost" in history_path else "bash"
        cmds = read_history_file(history_path, src)
        if cmds:
            all_cmds.extend(cmds)
            sources_used.append(f"{src} ({len(cmds)} cmds)")
            if verbose:
                print(f"  [+] {src:20s} -> {len(cmds):>5} commands")
    else:
        hfiles = find_history_files()
        if verbose: print("  [*] Scanning for history files...\n")
        for hf in hfiles:
            cmds = read_history_file(hf["path"], hf["source"])
            if cmds:
                all_cmds.extend(cmds)
                sources_used.append(f"{hf['source']} ({len(cmds)} cmds)")
                if verbose:
                    print(f"  [+] {hf['source']:20s} -> {len(cmds):>5} commands"
                          f"  [{hf['size_kb']} KB]  {hf['path']}")
            elif verbose:
                print(f"  [-] {hf['source']:20s} -> not found / empty")

    if verbose:
        print(f"\n  [*] Total commands collected: {len(all_cmds)}")

    if len(all_cmds) < MIN_COMMANDS * 3:
        if verbose:
            print("\n  [!] Not enough real history. Falling back to synthetic.")
        from core.data_gen import generate_all
        generate_all("data")
        from core.model import train
        return train("data")

    chunks = chunk_history(all_cmds, HISTORY_CHUNK_SIZE)
    if verbose:
        print(f"  [*] Split into {len(chunks)} session chunks\n")

    valid_feats = []
    for i, chunk in enumerate(chunks):
        f = extract_features(chunk)
        if f["_n"] >= MIN_COMMANDS:
            valid_feats.append(f)
            if verbose and (i % max(1, len(chunks)//6) == 0):
                print(f"  [+] Chunk {i+1:>3}/{len(chunks)}  "
                      f"n={f['_n']:>4}  bucket={f['_bucket']}")

    if len(valid_feats) < 3:
        if verbose: print("  [!] Not enough valid chunks. Falling back to synthetic.")
        from core.data_gen import generate_all
        generate_all("data")
        from core.model import train
        return train("data")

    if verbose:
        print(f"\n  [✓] Training on {len(valid_feats)} real session chunks...")

    X_raw  = [to_vector(f) for f in valid_feats]
    scaler = RobustScaler()
    X_sc   = scaler.fit_transform(X_raw)
    iso    = IsolationForest(n_estimators=200,
                              max_samples=min(256, len(X_sc)),
                              contamination=0.05, random_state=42)
    iso.fit(X_sc)

    # Build behavioral profile — pure Python stats
    profile: Dict = {}
    for j, name in enumerate(FEATURE_NAMES):
        col = [row[j] for row in X_raw]
        mn  = _mean(col)
        sd  = max(_std(col, mn), _MIN_STD.get(name, 1e-9))
        profile[name] = {
            "mean": round(mn, 8), "std": round(sd, 8),
            "min":  round(min(col), 8), "max": round(max(col), 8),
        }
    profile["_meta"] = {
        "n_training_sessions": len(valid_feats),
        "feature_names":       FEATURE_NAMES,
        "trained_from":        "real_history",
        "user":                user,
        "history_commands":    len(all_cmds),
        "sources":             sources_used,
    }

    if verbose: print("  [*] Training Intent Classifier...")
    clf = _train_intent_clf(valid_feats)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(IF_PATH,     "wb") as f: pickle.dump(iso,    f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    with open(PROFILE_PATH,"w")  as f: json.dump(profile,  f, indent=2)
    with open(INTENT_PATH, "wb") as f: pickle.dump(clf,    f)
    save_user_profile(user, profile)

    if verbose:
        print(f"\n  [✓] Models saved to models/")
        print(f"  [✓] User profile -> profiles/{user}.json")
        print(f"\n  Sources used:")
        for s in sources_used: print(f"    * {s}")
        print(f"\n  Total: {len(all_cmds)} real commands -> model knows YOUR behavior.")

    return {"n_sessions": len(valid_feats), "features": len(FEATURE_NAMES),
            "history_commands": len(all_cmds), "sources": sources_used, "user": user}
