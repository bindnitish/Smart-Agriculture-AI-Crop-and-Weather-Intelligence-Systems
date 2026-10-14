from pathlib import Path
from PIL import Image, UnidentifiedImageError
from collections import Counter
import hashlib
import pandas as pd


# ==========================================================
# CHANGE THESE PATHS
# ==========================================================

CROP_ROOT = Path("datasets/crop")
WEATHER_ROOT = Path("datasets/weather")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}


# ==========================================================
# BASIC FUNCTIONS
# ==========================================================

def is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def get_image_files(root):
    return sorted([
        p for p in root.rglob("*")
        if is_image_file(p)
    ])


def check_image(path):
    """
    Check whether image can be opened and collect
    basic image information.
    """
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            return {
                "readable": True,
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format
            }

    except (UnidentifiedImageError, OSError, ValueError):
        return {
            "readable": False,
            "width": None,
            "height": None,
            "mode": None,
            "format": None
        }


def md5_hash(path):
    """
    Generate MD5 hash for exact duplicate detection.
    """
    h = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ==========================================================
# DATASET AUDIT
# ==========================================================

def audit_dataset(root, dataset_name):

    if not root.exists():
        print(f"\nERROR: {root} does not exist.")
        return pd.DataFrame()

    files = get_image_files(root)

    print("\n" + "=" * 70)
    print(dataset_name.upper(), "DATASET")
    print("=" * 70)

    print("Dataset path:", root)
    print("Total image files:", len(files))

    records = []

    for path in files:

        info = check_image(path)

        relative_path = path.relative_to(root)

        # --------------------------------------------------
        # Detect split
        # --------------------------------------------------

        parts = relative_path.parts

        if len(parts) >= 3 and parts[0].lower() in {
            "train", "training",
            "val", "valid", "validation",
            "test", "testing"
        }:
            split = parts[0].lower()

            if split in {"training"}:
                split = "train"

            elif split in {"valid", "validation"}:
                split = "val"

            elif split in {"testing"}:
                split = "test"

            class_name = parts[1]

        elif len(parts) >= 2:

            split = "unsplit"
            class_name = parts[0]

        else:

            split = "unknown"
            class_name = "unknown"


        # --------------------------------------------------
        # Hash
        # --------------------------------------------------

        file_hash = None

        if info["readable"]:
            file_hash = md5_hash(path)


        # --------------------------------------------------
        # Record
        # --------------------------------------------------

        records.append({

            "dataset": dataset_name,

            "filename": path.name,

            "path": str(path),

            "relative_path": str(relative_path),

            "class": class_name,

            "split": split,

            "extension": path.suffix.lower(),

            "readable": info["readable"],

            "width": info["width"],

            "height": info["height"],

            "mode": info["mode"],

            "format": info["format"],

            "aspect_ratio": (
                info["width"] / info["height"]
                if info["readable"] and info["height"]
                else None
            ),

            "file_size_bytes": path.stat().st_size,

            "md5": file_hash

        })


    df = pd.DataFrame(records)

    return df


# ==========================================================
# SUMMARY
# ==========================================================

def print_summary(df, dataset_name):

    if df.empty:
        return

    print("\n" + "=" * 70)
    print(dataset_name.upper(), "SUMMARY")
    print("=" * 70)

    readable = df[df["readable"]]

    # ------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------

    print("\nBasic Information")
    print("-------------------------")

    print("Total files:", len(df))
    print("Readable images:", len(readable))
    print("Corrupted images:", len(df) - len(readable))

    print("Number of classes:",
          readable["class"].nunique())

    print("Classes:",
          sorted(readable["class"].unique()))


    # ------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------

    print("\nClass Distribution")
    print("-------------------------")

    class_counts = (
        readable["class"]
        .value_counts()
        .sort_index()
    )

    total = class_counts.sum()

    for cls, count in class_counts.items():

        percentage = (count / total) * 100

        print(
            f"{cls:15s} : "
            f"{count:6d} images "
            f"({percentage:6.2f}%)"
        )


    # ------------------------------------------------------
    # Split distribution
    # ------------------------------------------------------

    print("\nSplit Distribution")
    print("-------------------------")

    split_counts = readable["split"].value_counts()

    for split, count in split_counts.items():

        percentage = (count / len(readable)) * 100

        print(
            f"{split:10s} : "
            f"{count:6d} images "
            f"({percentage:6.2f}%)"
        )


    # ------------------------------------------------------
    # Image dimensions
    # ------------------------------------------------------

    print("\nImage Dimensions")
    print("-------------------------")

    print(
        "Minimum width :",
        readable["width"].min()
    )

    print(
        "Maximum width :",
        readable["width"].max()
    )

    print(
        "Mean width    :",
        round(readable["width"].mean(), 2)
    )

    print(
        "Minimum height:",
        readable["height"].min()
    )

    print(
        "Maximum height:",
        readable["height"].max()
    )

    print(
        "Mean height   :",
        round(readable["height"].mean(), 2)
    )


    # ------------------------------------------------------
    # 128 x 128 check
    # ------------------------------------------------------

    count_128 = (
        (readable["width"] == 128) &
        (readable["height"] == 128)
    ).sum()

    percentage_128 = (
        count_128 / len(readable) * 100
        if len(readable) > 0
        else 0
    )

    print("\n128 × 128 Images")
    print("-------------------------")

    print(
        f"{count_128}/{len(readable)} "
        f"({percentage_128:.2f}%)"
    )


    # ------------------------------------------------------
    # Image modes
    # ------------------------------------------------------

    print("\nImage Modes")
    print("-------------------------")

    print(
        readable["mode"]
        .value_counts()
        .to_string()
    )


    # ------------------------------------------------------
    # File extensions
    # ------------------------------------------------------

    print("\nFile Extensions")
    print("-------------------------")

    print(
        readable["extension"]
        .value_counts()
        .to_string()
    )


    # ------------------------------------------------------
    # Class imbalance
    # ------------------------------------------------------

    min_count = class_counts.min()
    max_count = class_counts.max()

    if min_count > 0:

        imbalance_ratio = max_count / min_count

        print("\nClass Imbalance")
        print("-------------------------")

        print(
            "Minimum class count:",
            min_count
        )

        print(
            "Maximum class count:",
            max_count
        )

        print(
            "Max/Min ratio:",
            round(imbalance_ratio, 3)
        )


# ==========================================================
# DUPLICATE CHECK
# ==========================================================

def check_duplicates(df, dataset_name):

    if df.empty:
        return

    readable = df[
        df["readable"] &
        df["md5"].notna()
    ]

    duplicates = (
        readable[
            readable["md5"].duplicated(
                keep=False
            )
        ]
        .sort_values("md5")
    )

    print("\n" + "=" * 70)
    print(dataset_name.upper(), "DUPLICATE CHECK")
    print("=" * 70)

    print(
        "Duplicate image rows:",
        len(duplicates)
    )

    if duplicates.empty:

        print("No exact duplicate images found.")

    else:

        print("\nDuplicate files:")

        print(
            duplicates[
                [
                    "split",
                    "class",
                    "relative_path",
                    "md5"
                ]
            ]
            .to_string(index=False)
        )


        # Cross-split duplicates
        print("\nCross-split duplicate check")

        for md5, group in duplicates.groupby("md5"):

            splits = group["split"].unique()

            if len(splits) > 1:

                print("\nWARNING:")
                print("Same image exists in splits:")

                print(
                    group[
                        [
                            "split",
                            "relative_path"
                        ]
                    ]
                    .to_string(index=False)
                )


# ==========================================================
# SAVE CSV
# ==========================================================

def save_results(df, dataset_name):

    if df.empty:
        return

    filename = (
        f"{dataset_name.lower()}_dataset_audit.csv"
    )

    df.to_csv(
        filename,
        index=False
    )

    print(
        f"\nDetailed audit saved to: {filename}"
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    crop_df = audit_dataset(
        CROP_ROOT,
        "Crop"
    )

    weather_df = audit_dataset(
        WEATHER_ROOT,
        "Weather"
    )

    print_summary(
        crop_df,
        "Crop"
    )

    print_summary(
        weather_df,
        "Weather"
    )

    check_duplicates(
        crop_df,
        "Crop"
    )

    check_duplicates(
        weather_df,
        "Weather"
    )

    save_results(
        crop_df,
        "Crop"
    )

    save_results(
        weather_df,
        "Weather"
    )

    print("\nAudit completed successfully.")