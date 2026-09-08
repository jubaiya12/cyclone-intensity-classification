"""Shared configuration for the cyclone intensity classification project."""

import os
from pathlib import Path

# Project root is the parent of this file's directory (src/../ = project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths (adjust via CYCLONE_DATA_ROOT env var if you download the dataset elsewhere)
DATA_ROOT = os.environ.get("CYCLONE_DATA_ROOT", str(PROJECT_ROOT / "data"))
DATA_DIR = os.path.join(DATA_ROOT, "insat3d_ir_cyclone_ds", "CYCLONE_DATASET_INFRARED")
LABELS_CSV = os.path.join(DATA_ROOT, "insat_3d_ds - Sheet.csv")

RESULTS_DIR = str(PROJECT_ROOT / "results")

KAGGLE_DATASET = "sshubam/insat3d-infrared-raw-cyclone-images-20132021"

# Model / training
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

CLASSES = [
    "Depression",
    "Cyclonic Storm",
    "Severe CS",
    "Very Severe CS",
    "Extremely Severe CS",
]

MODEL_OUT_PATH = str(PROJECT_ROOT / "src" / "best_model.pth")
TEST_SPLIT_CSV = str(PROJECT_ROOT / "src" / "test_split.csv")


def knots_to_category(knots: int) -> str:
    """Map a wind speed in knots to its IMD intensity category."""
    if knots < 34:
        return "Depression"
    elif knots < 48:
        return "Cyclonic Storm"
    elif knots < 64:
        return "Severe CS"
    elif knots < 90:
        return "Very Severe CS"
    else:
        return "Extremely Severe CS"
