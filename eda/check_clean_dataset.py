from pathlib import Path


# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets"


# ==========================================
# IMAGE EXTENSIONS
# ==========================================

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
# CROP DATASET
# ==========================================

crop_dir = DATASET_DIR / "crop"

print("=" * 60)
print("CLEAN CROP DATASET")
print("=" * 60)

crop_total = 0

for class_dir in sorted(crop_dir.iterdir()):

    if class_dir.is_dir():

        count = count_images(class_dir)

        print(f"{class_dir.name:12} : {count}")

        crop_total += count


print("-" * 60)
print(f"Total Clean Crop Images : {crop_total}")


# ==========================================
# WEATHER DATASET
# ==========================================

weather_dir = DATASET_DIR / "weather"

print("\n" + "=" * 60)
print("CLEAN WEATHER DATASET")
print("=" * 60)

weather_total = 0

for class_dir in sorted(weather_dir.iterdir()):

    if class_dir.is_dir():

        count = count_images(class_dir)

        print(f"{class_dir.name:12} : {count}")

        weather_total += count


print("-" * 60)
print(f"Total Clean Weather Images : {weather_total}")


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("FINAL CLEAN DATASET")
print("=" * 60)

print(f"Crop images    : {crop_total}")
print(f"Weather images : {weather_total}")
print(f"Total images   : {crop_total + weather_total}")