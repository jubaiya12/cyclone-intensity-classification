# Tropical Cyclone Intensity Classification from Satellite Imagery using Deep Learning

CIA 3 Project — Computer Vision (Deep Learning-based approach)

## Overview

This project classifies tropical cyclone intensity into 5 categories (based on the IMD scale) directly from INSAT-3D infrared satellite imagery, using a transfer-learning CNN (ResNet18). Grad-CAM is used to visualize which regions of each image the model focused on when making its prediction.

**Pipeline:** Satellite image → ResNet18 (transfer learning) → intensity category → Grad-CAM explainability

## Dataset

- **Source:** [INSAT3D Infrared & Raw Cyclone Imagery (2013–2021)](https://www.kaggle.com/datasets/sshubam/insat3d-infrared-raw-cyclone-images-20132021) (Kaggle)
- **Size:** 418 infrared satellite images
- **Labels:** Wind speed in knots, sourced from the dataset's official label CSV, bucketed into 5 IMD intensity categories:

| Category | Wind Speed (knots) |
|---|---|
| Depression | < 34 |
| Cyclonic Storm | 34–47 |
| Severe Cyclonic Storm | 48–63 |
| Very Severe Cyclonic Storm | 64–89 |
| Extremely Severe Cyclonic Storm | 90+ |

## Method

- **Model:** ResNet18 pretrained on ImageNet, fine-tuned for 5-class classification
- **Training:** Class-weighted loss (to handle imbalance toward rarer, high-intensity storms), heavy data augmentation (rotation, flip, colour jitter) given the small dataset size
- **Explainability:** Grad-CAM heatmaps generated for every prediction, overlaid on a grayscale version of the base image for visual clarity

## Results

- **Test accuracy:** ~29–48% across runs (small test set of 21 images leads to noticeable run-to-run variance)
- **Grad-CAM findings:** for higher-intensity storms in particular, the model consistently focuses on the cyclone's eye/core region — consistent with meteorological expectations
- Misclassifications cluster near category boundaries (e.g. a 33kt storm mistaken for the neighbouring 34kt+ category), which is expected given how close these values sit to the IMD scale's cutoffs

**Known limitation:** the dataset is small (418 images across 5 classes), which limits generalization and causes accuracy to vary between training runs. This is discussed further in the accompanying literature review.

## Why Google Colab

Training was done in Google Colab rather than locally, since a GPU (T4) is needed for practical training times — CNN training on CPU alone would be significantly slower. This notebook was developed and run in Colab, then exported here with all outputs (training logs, confusion matrix, Grad-CAM images) preserved.

## Files

```
cyclone-intensity-classification/
├── src/                      # all pipeline code
│   ├── config.py             # shared paths, classes, hyperparameters
│   ├── download_data.py      # downloads the dataset from Kaggle
│   ├── dataset.py            # builds labeled dataframe + PyTorch Dataset
│   ├── model.py              # ResNet18 model definition
│   ├── train.py              # training loop -> saves best_model.pth
│   ├── evaluate.py           # evaluation on test set -> confusion matrix
│   └── gradcam.py            # Grad-CAM visualizations per category
├── notebooks/
│   └── Cyclone_Intensity_Classification_v2.ipynb   # original Colab run, with saved outputs
├── results/                  # generated output images (created automatically)
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── gradcam_*.png (one per intensity category)
├── requirements.txt
├── .gitignore
└── README.md
```

The `.ipynb` file is the original Colab run, kept as-is because GitHub renders its saved outputs (training log, confusion matrix, Grad-CAM images) inline. The `src/` scripts are the same pipeline broken into clean, reusable modules that anyone can run from the command line, GPU or not.

## How to Run

```bash
pip install -r requirements.txt
cd src

# 1. Place your Kaggle API token at ~/.kaggle/kaggle.json
#    (get one from https://www.kaggle.com/settings -> API -> Create New Token)

# 2. Download the dataset (saved to ../data/, not tracked in git)
python download_data.py

# 3. Train (saves best_model.pth here in src/, and a plot to ../results/)
python train.py

# 4. Evaluate (reads best_model.pth, saves confusion matrix to ../results/)
python evaluate.py

# 5. Generate Grad-CAM visualizations (saved to ../results/)
python gradcam.py
```

**Note:** training on CPU will be significantly slower than on a GPU. This project was originally trained on a Google Colab T4 GPU; `notebooks/Cyclone_Intensity_Classification_v2.ipynb` captures that run's full output. Running `train.py` locally without a GPU still works end-to-end (tested), just more slowly.

## Author

Umma Jubaiya — B.Tech CSE (AI & ML), CHRIST (Deemed to be University), Bengaluru
