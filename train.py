"""Train the cyclone intensity classifier.

Usage:
    python train.py
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_SEED, MODEL_OUT_PATH, CLASSES,
    RESULTS_DIR, TEST_SPLIT_CSV,
)
from dataset import build_labeled_dataframe, CycloneDataset, TRAIN_TRANSFORMS, EVAL_TRANSFORMS
from model import build_model, compute_class_weights

torch.manual_seed(RANDOM_SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cpu":
        print("WARNING: no GPU detected -- training will be much slower on CPU.")

    df = build_labeled_dataframe()
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    df["label"] = df["category"].map(class_to_idx)

    counts = df["label"].value_counts()
    df = df[df["label"].isin(counts[counts >= 2].index)].reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=RANDOM_SEED)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    test_df.to_csv(TEST_SPLIT_CSV, index=False)  # so evaluate.py/gradcam.py use the same split

    train_loader = DataLoader(CycloneDataset(train_df, TRAIN_TRANSFORMS), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CycloneDataset(val_df, EVAL_TRANSFORMS), batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(device)
    class_weights = compute_class_weights(train_df["label"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT_PATH)
            print("  -> Saved new best model")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    print("Saved", curves_path)


if __name__ == "__main__":
    main()
