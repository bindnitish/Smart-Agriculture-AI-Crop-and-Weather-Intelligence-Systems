from pathlib import Path

from torch.utils.data import DataLoader

from dataloaders.image_dataset import ImageDataset

from preprocessing.transforms import (
    train_transform,
    val_transform,
    test_transform
)


# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_ROOT / "splits"


# ==========================================
# SETTINGS
# ==========================================

BATCH_SIZE = 32
NUM_WORKERS = 0


# ==========================================
# CROP DATASETS
# ==========================================

crop_train_dataset = ImageDataset(
    SPLIT_DIR / "crop_train.csv",
    transform=train_transform
)

crop_val_dataset = ImageDataset(
    SPLIT_DIR / "crop_val.csv",
    transform=val_transform
)

crop_test_dataset = ImageDataset(
    SPLIT_DIR / "crop_test.csv",
    transform=test_transform
)


# ==========================================
# CROP DATALOADERS
# ==========================================

crop_train_loader = DataLoader(
    crop_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

crop_val_loader = DataLoader(
    crop_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

crop_test_loader = DataLoader(
    crop_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ==========================================
# WEATHER DATASETS
# ==========================================

weather_train_dataset = ImageDataset(
    SPLIT_DIR / "weather_train.csv",
    transform=train_transform
)

weather_val_dataset = ImageDataset(
    SPLIT_DIR / "weather_val.csv",
    transform=val_transform
)

weather_test_dataset = ImageDataset(
    SPLIT_DIR / "weather_test.csv",
    transform=test_transform
)


# ==========================================
# WEATHER DATALOADERS
# ==========================================

weather_train_loader = DataLoader(
    weather_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

weather_val_loader = DataLoader(
    weather_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

weather_test_loader = DataLoader(
    weather_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)