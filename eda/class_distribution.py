from pathlib import Path
import matplotlib.pyplot as plt


# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ==========================================
# COUNT IMAGES
# ==========================================

def count_images(folder):

    return sum(
        1
        for file in folder.rglob("*")
        if (
            file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )
    )


# ==========================================
# CROP CLASS DISTRIBUTION
# ==========================================

crop_dir = DATASET_DIR / "crop"

crop_classes = []
crop_counts = []

for class_dir in sorted(crop_dir.iterdir()):

    if class_dir.is_dir():

        crop_classes.append(class_dir.name)
        crop_counts.append(count_images(class_dir))


print("=" * 50)
print("CROP CLASS DISTRIBUTION")
print("=" * 50)

for name, count in zip(crop_classes, crop_counts):

    print(f"{name:12} : {count}")


# ==========================================
# CROP BAR CHART
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(crop_classes, crop_counts)

plt.title("Crop Dataset Class Distribution")

plt.xlabel("Crop Class")
plt.ylabel("Number of Images")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()


# ==========================================
# WEATHER CLASS DISTRIBUTION
# ==========================================

weather_dir = DATASET_DIR / "weather"

weather_classes = []
weather_counts = []

for class_dir in sorted(weather_dir.iterdir()):

    if class_dir.is_dir():

        weather_classes.append(class_dir.name)
        weather_counts.append(
            count_images(class_dir)
        )


print("\n" + "=" * 50)
print("WEATHER CLASS DISTRIBUTION")
print("=" * 50)

for name, count in zip(
    weather_classes,
    weather_counts
):

    print(f"{name:12} : {count}")


# ==========================================
# WEATHER BAR CHART
# ==========================================

plt.figure(figsize=(12, 6))

plt.bar(
    weather_classes,
    weather_counts
)

plt.title("Weather Dataset Class Distribution")

plt.xlabel("Weather Class")
plt.ylabel("Number of Images")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.show()