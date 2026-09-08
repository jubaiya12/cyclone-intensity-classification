"""Model definition: ResNet18 transfer-learning classifier for cyclone intensity."""

import torch
import torch.nn as nn
from torchvision import models

from config import CLASSES


def build_model(device: torch.device) -> nn.Module:
    """ResNet18 pretrained on ImageNet, fine-tuned for 5-class cyclone intensity classification."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASSES))
    return model.to(device)


def compute_class_weights(train_labels, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights to counter class imbalance."""
    counts = train_labels.value_counts().reindex(range(len(CLASSES)), fill_value=1).values
    weights = torch.tensor(1.0 / counts, dtype=torch.float32)
    weights = weights / weights.sum() * len(CLASSES)
    return weights.to(device)
