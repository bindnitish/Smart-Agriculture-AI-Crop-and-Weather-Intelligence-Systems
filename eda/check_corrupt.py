from pathlib import Path
from PIL import Image


# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets"


# ==========================================
# SUPPORTED IMAGE FORMATS
# ==========================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ==========================================
# FIND CORRUPT IMAGES
# ==========================================

def find_corrupt_images(folder):

    corrupt_images = []

    for file in folder.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ):

            try:

                # Open image
                with Image.open(file) as img:

                    # Verify image integrity
                    img.verify()

            except Exception:

                corrupt_images.append(file)

    return corrupt_images


# ==========================================
# CROP DATASET
# ==========================================

crop_dir = DATASET_DIR / "crop"

print("=" * 60)
print("CHECKING CROP DATASET")
print("=" * 60)

crop_corrupt = find_corrupt_images(crop_dir)

print(f"Corrupt crop images : {len(crop_corrupt)}")


# ==========================================
# WEATHER DATASET
# ==========================================

weather_dir = DATASET_DIR / "weather"

print("\n" + "=" * 60)
print("CHECKING WEATHER DATASET")
print("=" * 60)

weather_corrupt = find_corrupt_images(weather_dir)

print(f"Corrupt weather images : {len(weather_corrupt)}")


# ==========================================
# SHOW CORRUPT FILES
# ==========================================

if crop_corrupt:

    print("\n" + "=" * 60)
    print("CORRUPT CROP FILES")
    print("=" * 60)

    for file in crop_corrupt:

        print(file)


if weather_corrupt:

    print("\n" + "=" * 60)
    print("CORRUPT WEATHER FILES")
    print("=" * 60)

    for file in weather_corrupt:

        print(file)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("FINAL QUALITY CHECK")
print("=" * 60)

print(f"Crop corrupt images    : {len(crop_corrupt)}")
print(f"Weather corrupt images : {len(weather_corrupt)}")