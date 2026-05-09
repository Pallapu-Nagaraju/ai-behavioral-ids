"""
runtime/user_manager.py — Per-User Profile Management
Saves and loads per-user behavioral profiles as JSON files.
Falls back to global profile when user-specific profile does not exist.
"""

import os
import json
from typing import Dict, Optional

PROFILES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiles")


def _path(user: str) -> str:
    return os.path.join(PROFILES_DIR, f"{user}.json")


def load_user_profile(user: str) -> Optional[Dict]:
    """Return user profile dict or None if not found."""
    p = _path(user)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def save_user_profile(user: str, profile: Dict) -> None:
    """Persist a user profile to disk."""
    os.makedirs(PROFILES_DIR, exist_ok=True)
    with open(_path(user), "w") as f:
        json.dump(profile, f, indent=2)


def list_users() -> list:
    """Return list of users who have saved profiles."""
    if not os.path.exists(PROFILES_DIR):
        return []
    return [
        f[:-5] for f in os.listdir(PROFILES_DIR)
        if f.endswith(".json")
    ]


def build_user_profile_from_sessions(user: str, session_texts: list,
                                     global_profile: Dict) -> Dict:
    """
    Build a per-user profile from a list of raw session strings.
    Merges with global profile structure (keeps same feature keys).
    """
    from core.features import extract_features, FEATURE_NAMES
    import math

    feature_rows = []
    for raw in session_texts:
        feats = extract_features(raw)
        if feats["_n"] >= 10:
            feature_rows.append(feats)

    if not feature_rows:
        return global_profile   # fall back

    profile = {}
    for name in FEATURE_NAMES:
        col  = [f.get(name, 0.0) for f in feature_rows]
        mean = sum(col) / len(col)
        std  = math.sqrt(sum((v - mean) ** 2 for v in col) / max(len(col), 1))
        std  = max(std, 1e-9)
        profile[name] = {
            "mean": round(mean, 6),
            "std":  round(std,  6),
            "min":  round(min(col), 6),
            "max":  round(max(col), 6),
        }

    profile["_meta"] = {
        "user":              user,
        "n_sessions":        len(feature_rows),
        "feature_names":     FEATURE_NAMES,
    }

    save_user_profile(user, profile)
    return profile
