"""
Loads the deployment artifacts (model / vocab / config) produced by the
training notebook's "Save the Best Model" step. Point ARTIFACTS_DIR at
wherever you copied that deployment_artifacts/ folder to.
"""

import json
import os
from pathlib import Path

# Directory containing this file (ocr/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Allow override via environment variable
ARTIFACTS_DIR = Path(
    os.environ.get(
        "ARTIFACTS_DIR",
        BASE_DIR / "deployment_artifacts"
    )
)

MODEL_PATH = ARTIFACTS_DIR / "crnn_ocr_model.keras"
VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"


def _require(path, hint):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{hint} not found at '{path}'. "
            f"Copy the deployment_artifacts folder into the project "
            f"or set ARTIFACTS_DIR."
        )
    return str(path)


def load_config() -> dict:
    _require(CONFIG_PATH, "config.json")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vocab() -> list:
    _require(VOCAB_PATH, "vocab.json")
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
