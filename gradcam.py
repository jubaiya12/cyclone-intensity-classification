"""Generate Grad-CAM visualizations for one test example per category.

Usage:
    python gradcam.py
(Run train.py first -- this expects best_model.pth and test_split.csv to exist.)
"""

import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import CLASSES, MODEL_OUT_PATH, IMG_SIZE, RESULTS_DIR, TEST_SPLIT_CSV
from dataset import EVAL_TRANSFORMS
from model import build_model

os.makedirs(RESULTS_DIR, exist_ok=True)


def visualize_gradcam(model, cam, sample_row, device):
    img = Image.open(sample_row["path"]).convert("RGB")
    input_tensor = EVAL_TRANSFORMS(img).unsqueeze(0).to(device)

    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    pred_idx = outputs.argmax(dim=1).item()
    confidence = probs[pred_idx].item()

    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]
    rgb_img = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    # IR images are already rainbow/jet-colored, same scheme Grad-CAM uses --
    # overlaying jet-on-jet hides the heatmap. Use grayscale base for contrast.
    gray_img = np.array(img.convert("L").resize((IMG_SIZE, IMG_SIZE))) / 255.0
    gray_img_3ch = np.stack([gray_img] * 3, axis=-1)
    visualization = show_cam_on_image(gray_img_3ch, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title(f'Original (true: {sample_row["category"]}, {sample_row["knots"]}kt)')
    axes[0].axis("off")
    axes[1].imshow(visualization)
    axes[1].set_title(f"Grad-CAM: {CLASSES[pred_idx]} ({confidence*100:.1f}%)")
    axes[1].axis("off")
    plt.tight_layout()
    out_name = os.path.join(RESULTS_DIR, f'gradcam_{sample_row["category"].replace(" ", "_")}.png')
    plt.savefig(out_name, dpi=150)
    print("Saved", out_name)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_df = pd.read_csv(TEST_SPLIT_CSV)

    model = build_model(device)
    model.load_state_dict(torch.load(MODEL_OUT_PATH, map_location=device))
    model.eval()

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    for cat in test_df["category"].unique():
        subset = test_df[test_df["category"] == cat]
        if len(subset) > 0:
            print(f"--- True category: {cat} ---")
            visualize_gradcam(model, cam, subset.iloc[0], device)


if __name__ == "__main__":
    main()
