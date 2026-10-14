from pathlib import Path
from collections import defaultdict
import hashlib
import shutil


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets"

QUARANTINE_DIR = PROJECT_ROOT / "duplicates_quarantine"


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
# HASH FUNCTION
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
# FIND DUPLICATE GROUPS
# ==========================================

def find_duplicates(dataset_dir):

    hash_map = defaultdict(list)

    for file in dataset_dir.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ):

            try:

                image_hash = get_image_hash(file)

                hash_map[image_hash].append(file)

            except Exception as e:

                print(f"Could not process: {file}")
                print(f"Error: {e}")

    duplicate_groups = {
        image_hash: files
        for image_hash, files in hash_map.items()
        if len(files) > 1
    }

    return duplicate_groups


# ==========================================
# SAFE MOVE
# ==========================================

def move_to_quarantine(file_path, dataset_dir, quarantine_dir):

    relative_path = file_path.relative_to(dataset_dir)

    destination = quarantine_dir / relative_path

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Avoid overwriting if a file with same name exists
    if destination.exists():

        counter = 1

        while True:

            new_name = (
                destination.stem
                + f"_duplicate_{counter}"
                + destination.suffix
            )

            new_destination = (
                destination.parent / new_name
            )

            if not new_destination.exists():

                destination = new_destination
                break

            counter += 1

    shutil.move(
        str(file_path),
        str(destination)
    )

    return destination


# ==========================================
# CLEAN ONE DATASET
# ==========================================

def clean_dataset(dataset_name):

    dataset_dir = DATASET_DIR / dataset_name

    quarantine_dir = QUARANTINE_DIR / dataset_name

    print("\n" + "=" * 60)
    print(f"CLEANING {dataset_name.upper()} DUPLICATES")
    print("=" * 60)

    duplicate_groups = find_duplicates(dataset_dir)

    same_class_groups = 0
    cross_class_groups = 0

    moved_same_class = 0
    moved_cross_class = 0

    # --------------------------------------
    # Process duplicate groups
    # --------------------------------------

    for image_hash, files in duplicate_groups.items():

        classes = {
            file.parent.name
            for file in files
        }

        # ==================================
        # CROSS-CLASS DUPLICATE
        # ==================================

        if len(classes) > 1:

            cross_class_groups += 1

            print("\nCross-class duplicate found:")

            for file in files:

                print(f"  {file}")

                move_to_quarantine(
                    file,
                    dataset_dir,
                    quarantine_dir
                )

                moved_cross_class += 1

        # ==================================
        # SAME-CLASS DUPLICATE
        # ==================================

        else:

            same_class_groups += 1

            # Keep first image
            keep_file = files[0]

            print(
                f"\nSame-class duplicate group:"
            )

            print(f"  KEEP: {keep_file}")

            # Move remaining copies
            for duplicate_file in files[1:]:

                print(
                    f"  MOVE: {duplicate_file}"
                )

                move_to_quarantine(
                    duplicate_file,
                    dataset_dir,
                    quarantine_dir
                )

                moved_same_class += 1

    # ======================================
    # SUMMARY
    # ======================================

    print("\n" + "-" * 60)
    print(f"{dataset_name.upper()} CLEANUP SUMMARY")
    print("-" * 60)

    print(
        f"Duplicate groups      : "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Same-class groups     : "
        f"{same_class_groups}"
    )

    print(
        f"Cross-class groups    : "
        f"{cross_class_groups}"
    )

    print(
        f"Same-class files moved: "
        f"{moved_same_class}"
    )

    print(
        f"Cross-class files moved: "
        f"{moved_cross_class}"
    )

    print(
        f"Total files moved     : "
        f"{moved_same_class + moved_cross_class}"
    )


# ==========================================
# CREATE QUARANTINE FOLDER
# ==========================================

QUARANTINE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# CLEAN CROP
# ==========================================

clean_dataset("crop")


# ==========================================
# CLEAN WEATHER
# ==========================================

clean_dataset("weather")


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n" + "=" * 60)
print("DUPLICATE CLEANUP COMPLETED")
print("=" * 60)

print(
    f"\nQuarantine location:\n"
    f"{QUARANTINE_DIR}"
)

print(
    "\nOriginal duplicate files were MOVED, "
    "not permanently deleted."
)