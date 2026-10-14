from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from models.efficientnet_model import create_model
from dataloaders.loaders import crop_test_loader


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "crop_efficientnetv2s_best.pth"
)


# ==========================================
# CLASS NAMES
# ==========================================

CLASS_NAMES = [
    "jute",
    "maize",
    "rice",
    "sugarcane",
    "wheat"
]


# ==========================================
# DEVICE
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("CROP MODEL TEST EVALUATION")
print("=" * 60)

print(f"Device      : {DEVICE}")
print(f"Test images : {len(crop_test_loader.dataset)}")


# ==========================================
# CREATE MODEL
# ==========================================

model = create_model(
    num_classes=5,
    pretrained=False
)


# ==========================================
# LOAD BEST MODEL
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


print(
    f"\nBest validation accuracy: "
    f"{checkpoint['val_accuracy']:.2f}%"
)

print(
    f"Best model epoch: "
    f"{checkpoint['epoch']}"
)


# ==========================================
# TEST
# ==========================================

criterion = torch.nn.CrossEntropyLoss()

running_loss = 0.0

correct = 0
total = 0

all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in crop_test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        # Predictions
        _, predictions = torch.max(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        # Store predictions
        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ==========================================
# TEST METRICS
# ==========================================

test_loss = running_loss / total

test_accuracy = (
    correct / total
) * 100


print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.2f}%"
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    digits=4,
    output_dict=True
)
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4
    )
)


# ==========================================
# SAVE METRICS
# ==========================================

import pandas as pd

report_df = pd.DataFrame(
    report
).transpose()

metrics_path = (
    PROJECT_ROOT
    / "evaluation"
    / "crop_metrics.csv"
)

report_df.to_csv(
    metrics_path,
    index=True
)

print(
    f"\nMetrics saved to: {metrics_path}"
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ==========================================
# DISPLAY CONFUSION MATRIX
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(
    figsize=(9, 7)
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title(
    "Crop Classification - Confusion Matrix"
)

plt.tight_layout()

plt.show()


# ==========================================
# FINAL
# ==========================================

print("\n" + "=" * 60)
print("CROP TEST EVALUATION COMPLETED")
print("=" * 60)

