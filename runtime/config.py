import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROFILES_DIR = os.path.join(BASE_DIR, "runtime", "profiles")

NORMAL_THR = 25
SUSP_THR = 35
