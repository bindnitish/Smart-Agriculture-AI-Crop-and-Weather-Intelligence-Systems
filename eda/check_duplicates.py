from pathlib import Path
from collections import defaultdict
import hashlib


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
# IMAGE HASH
# ==========================================

def get_image_hash(file_path):

    md5 = hashlib.md5()

    with open(file_path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            md5.update(chunk)

    return md5.hexdigest()


# ==========================================
# FIND DUPLICATES
# ==========================================

def find_duplicates(dataset_dir):

    hash_map = defaultdict(list)

    total_images = 0

    for file in dataset_dir.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ):

            total_images += 1

            try:

                image_hash = get_image_hash(file)

                hash_map[image_hash].append(file)

            except Exception as e:

                print(f"Error processing: {file}")
                print(f"Error: {e}")

    # Only hashes appearing more than once
    duplicate_groups = {
        image_hash: files
        for image_hash, files in hash_map.items()
        if len(files) > 1
    }

    return total_images, duplicate_groups


# ==========================================
# ANALYZE DUPLICATES
# ==========================================

def analyze_duplicates(duplicate_groups):

    same_class_groups = 0
    cross_class_groups = 0
    duplicate_images = 0

    for files in duplicate_groups.values():

        duplicate_images += len(files)

        classes = {
            file.parent.name
            for file in files
        }

        if len(classes) == 1:

            same_class_groups += 1

        else:

            cross_class_groups += 1

    return (
        same_class_groups,
        cross_class_groups,
        duplicate_images
    )


# ==========================================
# CROP
# ==========================================

print("=" * 60)
print("CHECKING CROP DUPLICATES")
print("=" * 60)

crop_dir = DATASET_DIR / "crop"

crop_total, crop_duplicates = find_duplicates(crop_dir)

crop_same, crop_cross, crop_duplicate_images = analyze_duplicates(
    crop_duplicates
)

print(f"Total images              : {crop_total}")
print(f"Duplicate groups          : {len(crop_duplicates)}")
print(f"Same-class groups         : {crop_same}")
print(f"Cross-class groups        : {crop_cross}")
print(f"Images in duplicate groups: {crop_duplicate_images}")


# ==========================================
# WEATHER
# ==========================================

print("\n" + "=" * 60)
print("CHECKING WEATHER DUPLICATES")
print("=" * 60)

weather_dir = DATASET_DIR / "weather"

weather_total, weather_duplicates = find_duplicates(
    weather_dir
)

weather_same, weather_cross, weather_duplicate_images = analyze_duplicates(
    weather_duplicates
)

print(f"Total images              : {weather_total}")
print(f"Duplicate groups          : {len(weather_duplicates)}")
print(f"Same-class groups         : {weather_same}")
print(f"Cross-class groups        : {weather_cross}")
print(f"Images in duplicate groups: {weather_duplicate_images}")


# ==========================================
# CROSS-CLASS DUPLICATE DETAILS
# ==========================================

def show_cross_class_duplicates(duplicate_groups, dataset_name):

    found = 0

    for files in duplicate_groups.values():

        classes = {
            file.parent.name
            for file in files
        }

        if len(classes) > 1:

            found += 1

            print(
                f"\n{dataset_name} "
                f"Cross-class duplicate group {found}"
            )

            print("-" * 50)

            for file in files:

                print(f"Class : {file.parent.name}")
                print(f"Path  : {file}")

            # Show maximum 10 groups
            if found >= 10:
                break

    if found == 0:

        print(
            f"No cross-class duplicates found in "
            f"{dataset_name}."
        )


# ==========================================
# SHOW CROSS-CLASS RESULTS
# ==========================================

print("\n" + "=" * 60)
print("CROSS-CLASS DUPLICATES")
print("=" * 60)

show_cross_class_duplicates(
    crop_duplicates,
    "CROP"
)

show_cross_class_duplicates(
    weather_duplicates,
    "WEATHER"
)