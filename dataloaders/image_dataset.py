from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================
# IMAGE DATASET
# ==========================================

class ImageDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        # Image path
        image_path = (
            PROJECT_ROOT
            / self.data.iloc[index]["image_path"]
        )

        # Label
        label = int(
            self.data.iloc[index]["label"]
        )

        # Open image
        image = Image.open(image_path).convert("RGB")

        # Apply transform
        if self.transform is not None:

            image = self.transform(image)

        return image, label