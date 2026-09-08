"""Download the INSAT3D cyclone dataset from Kaggle.

Requires a Kaggle API token (kaggle.json) to be present at ~/.kaggle/kaggle.json
Get one from: https://www.kaggle.com/settings -> API -> Create New Token

Usage:
    python download_data.py
"""

import os
import subprocess
import sys

from config import DATA_ROOT, KAGGLE_DATASET, DATA_DIR, LABELS_CSV


def check_kaggle_credentials():
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("ERROR: No Kaggle API token found at", kaggle_json)
        print("Get one from https://www.kaggle.com/settings -> API -> Create New Token")
        print("Then place kaggle.json at ~/.kaggle/kaggle.json (chmod 600 on Linux/Mac)")
        sys.exit(1)


def download():
    check_kaggle_credentials()
    os.makedirs(DATA_ROOT, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} into {DATA_ROOT} ...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", DATA_ROOT,
            "--unzip",
        ],
        check=True,
    )
    print("Download complete.")
    print("Images dir exists:", os.path.exists(DATA_DIR))
    print("Labels CSV exists:", os.path.exists(LABELS_CSV))


if __name__ == "__main__":
    download()
