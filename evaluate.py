"""Evaluate the trained model on the held-out test set.

Usage:
    python evaluate.py
(Run train.py first -- this expects best_model.pth and test_split.csv to exist.)
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from config import CLASSES, MODEL_OUT_PATH, BATCH_SIZE, RESULTS_DIR, TEST_SPLIT_CSV
from dataset import CycloneDataset, EVAL_TRANSFORMS
from model import build_model

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_df = pd.read_csv(TEST_SPLIT_CSV)
    test_loader = DataLoader(CycloneDataset(test_df, EVAL_TRANSFORMS), batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(device)
    model.load_state_dict(torch.load(MODEL_OUT_PATH, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    present_labels = sorted(set(all_labels) | set(all_preds))
    present_names = [CLASSES[i] for i in present_labels]

    print(classification_report(all_labels, all_preds, labels=present_labels, target_names=present_names))
    print("Macro F1:", f1_score(all_labels, all_preds, average="macro"))

    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=present_names, yticklabels=present_names, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print("Saved", cm_path)


if __name__ == "__main__":
    main()
