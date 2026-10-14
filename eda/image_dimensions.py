from pathlib import Path
from collections import Counter
from PIL import Image


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
# GET IMAGE DIMENSIONS
# ==========================================

def get_image_dimensions(folder):

    dimensions = Counter()

    for file in folder.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ):

            try:

                with Image.open(file) as img:

                    dimensions[img.size] += 1

            except Exception:
                pass

    return dimensions


# ==========================================
# CROP
# ==========================================

crop_dir = DATASET_DIR / "crop"

print("=" * 60)
print("CROP IMAGE DIMENSIONS")
print("=" * 60)

crop_dimensions = get_image_dimensions(crop_dir)

for size, count in crop_dimensions.most_common(15):

    print(f"{str(size):15} : {count}")


print(f"\nUnique crop image sizes : {len(crop_dimensions)}")


# ==========================================
# WEATHER
# ==========================================

weather_dir = DATASET_DIR / "weather"

print("\n" + "=" * 60)
print("WEATHER IMAGE DIMENSIONS")
print("=" * 60)

weather_dimensions = get_image_dimensions(weather_dir)

for size, count in weather_dimensions.most_common(15):

    print(f"{str(size):15} : {count}")


print(
    f"\nUnique weather image sizes : "
    f"{len(weather_dimensions)}"
)