from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATASET_DIR = PROJECT_ROOT / "datasets"

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def count_images(folder):
    """Count all images inside a folder."""

    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


# ==================================================
# CROP DATASET
# ==================================================

crop_dir = DATASET_DIR / "crop"

print("=" * 50)
print("CROP DATASET")
print("=" * 50)

total_crop = 0

for class_dir in sorted(crop_dir.iterdir()):

    if class_dir.is_dir():

        count = count_images(class_dir)

        print(f"{class_dir.name:12} : {count}")

        total_crop += count


print("-" * 50)
print(f"Total Crop Images : {total_crop}")


# ==================================================
# WEATHER DATASET
# ==================================================

weather_dir = DATASET_DIR / "weather"

print("\n" + "=" * 50)
print("WEATHER DATASET")
print("=" * 50)

total_weather = 0

for class_dir in sorted(weather_dir.iterdir()):

    if class_dir.is_dir():

        count = count_images(class_dir)

        print(f"{class_dir.name:12} : {count}")

        total_weather += count


print("-" * 50)
print(f"Total Weather Images : {total_weather}")


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 50)
print("FINAL DATASET SUMMARY")
print("=" * 50)

print(f"Crop images    : {total_crop}")
print(f"Weather images : {total_weather}")
print(f"Total images   : {total_crop + total_weather}")