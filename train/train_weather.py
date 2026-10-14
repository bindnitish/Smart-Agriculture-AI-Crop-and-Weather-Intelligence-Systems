from pathlib import Path
import time

import torch

from models.efficientnet_model import create_model

from dataloaders.loaders import (
    weather_train_loader,
    weather_val_loader
)

from train.loss import criterion
from train.optimizer import get_optimizer

from train.engine import (
    train_one_epoch,
    validate
)


# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# SETTINGS
# ==========================================

NUM_CLASSES = 11

NUM_EPOCHS = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "weather_efficientnetv2s_best.pth"
)


# ==========================================
# WEATHER CLASS MAPPING
# ==========================================

CLASS_TO_IDX = {
    "dew": 0,
    "fogsmog": 1,
    "frost": 2,
    "glaze": 3,
    "hail": 4,
    "lightning": 5,
    "rain": 6,
    "rainbow": 7,
    "rime": 8,
    "sandstorm": 9,
    "snow": 10
}


# ==========================================
# START
# ==========================================

print("=" * 60)
print("WEATHER MODEL TRAINING")
print("=" * 60)

print(f"Device       : {DEVICE}")
print(f"Epochs       : {NUM_EPOCHS}")
print(f"Classes      : {NUM_CLASSES}")
print(
    f"Train images : "
    f"{len(weather_train_loader.dataset)}"
)
print(
    f"Val images   : "
    f"{len(weather_val_loader.dataset)}"
)


# ==========================================
# CREATE MODEL
# ==========================================

model = create_model(
    num_classes=NUM_CLASSES,
    pretrained=True
)

model = model.to(DEVICE)


# ==========================================
# LOSS
# ==========================================

loss_function = criterion


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = get_optimizer(model)


# ==========================================
# BEST VALIDATION ACCURACY
# ==========================================

best_val_accuracy = 0.0


# ==========================================
# TRAINING LOOP
# ==========================================

for epoch in range(NUM_EPOCHS):

    epoch_start = time.time()

    print("\n" + "=" * 60)
    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}]"
    )
    print("=" * 60)


    # --------------------------------------
    # TRAIN
    # --------------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model=model,
        dataloader=weather_train_loader,
        criterion=loss_function,
        optimizer=optimizer,
        device=DEVICE
    )


    # --------------------------------------
    # VALIDATION
    # --------------------------------------

    val_loss, val_accuracy = validate(
        model=model,
        dataloader=weather_val_loader,
        criterion=loss_function,
        device=DEVICE
    )


    # --------------------------------------
    # TIME
    # --------------------------------------

    epoch_time = time.time() - epoch_start


    # --------------------------------------
    # RESULTS
    # --------------------------------------

    print(
        f"Train Loss      : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss        : {val_loss:.4f}"
    )

    print(
        f"Val Accuracy    : {val_accuracy:.2f}%"
    )

    print(
        f"Epoch Time      : {epoch_time:.2f} seconds"
    )


    # --------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_loss": val_loss,

                "val_accuracy":
                    val_accuracy,

                "class_to_idx":
                    CLASS_TO_IDX
            },
            BEST_MODEL_PATH
        )

        print(
            f"Best model saved! "
            f"Val Accuracy = "
            f"{val_accuracy:.2f}%"
        )


# ==========================================
# TRAINING COMPLETE
# ==========================================

print("\n" + "=" * 60)
print("WEATHER TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best Model Path           : "
    f"{BEST_MODEL_PATH}"
)