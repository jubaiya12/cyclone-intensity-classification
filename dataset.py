"""Data loading: builds the labeled dataframe and PyTorch Dataset for cyclone images."""

import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import DATA_DIR, LABELS_CSV, IMG_SIZE, knots_to_category


def build_labeled_dataframe() -> pd.DataFrame:
    """Read the official labels CSV and match each row to a real image file.

    Returns a dataframe with columns: path, knots, category
    """
    labels_df = pd.read_csv(LABELS_CSV)

    records = []
    for _, row in labels_df.iterrows():
        img_path = os.path.join(DATA_DIR, row["img_name"])
        if os.path.exists(img_path):
            knots = int(row["label"])
            records.append({
                "path": img_path,
                "knots": knots,
                "category": knots_to_category(knots),
            })

    df = pd.DataFrame(records)
    print(f"Matched {len(df)} of {len(labels_df)} CSV rows to real image files")
    print(df["category"].value_counts())
    return df


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CycloneDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        return img, row["label"]
