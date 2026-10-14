from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets"
SPLIT_DIR = PROJECT_ROOT / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# SETTINGS
# ==========================================

RANDOM_SEED = 42

TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ==========================================
# CREATE DATAFRAME
# ==========================================

def create_dataframe(dataset_dir):

    records = []

    class_dirs = sorted(
        [
            folder
            for folder in dataset_dir.iterdir()
            if folder.is_dir()
        ]
    )

    class_to_idx = {
        class_dir.name: idx
        for idx, class_dir in enumerate(class_dirs)
    }

    for class_dir in class_dirs:

        class_name = class_dir.name
        class_index = class_to_idx[class_name]

        for file in class_dir.rglob("*"):

            if (
                file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
            ):

                records.append(
                    {
                        "image_path": str(
                            file.relative_to(PROJECT_ROOT)
                        ),
                        "class": class_name,
                        "label": class_index
                    }
                )

    df = pd.DataFrame(records)

    return df, class_to_idx


# ==========================================
# CREATE SPLITS
# ==========================================

def create_dataset_splits(dataset_name):

    dataset_dir = DATASET_DIR / dataset_name

    print("\n" + "=" * 60)
    print(f"CREATING {dataset_name.upper()} SPLITS")
    print("=" * 60)

    df, class_to_idx = create_dataframe(dataset_dir)

    print(f"Total images: {len(df)}")

    # --------------------------------------
    # 80% TRAIN
    # 20% TEMP
    # --------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_SEED,
        stratify=df["label"]
    )

    # --------------------------------------
    # TEMP → 50% VAL + 50% TEST
    # --------------------------------------

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"]
    )

    # --------------------------------------
    # SHUFFLE
    # --------------------------------------

    train_df = train_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    val_df = val_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    test_df = test_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    # --------------------------------------
    # SAVE
    # --------------------------------------

    train_df.to_csv(
        SPLIT_DIR / f"{dataset_name}_train.csv",
        index=False
    )

    val_df.to_csv(
        SPLIT_DIR / f"{dataset_name}_val.csv",
        index=False
    )

    test_df.to_csv(
        SPLIT_DIR / f"{dataset_name}_test.csv",
        index=False
    )

    # --------------------------------------
    # PRINT SUMMARY
    # --------------------------------------

    print("\nClass mapping:")

    for class_name, index in class_to_idx.items():
        print(f"{index} -> {class_name}")

    print("\nSplit sizes:")

    print(f"Train      : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Test       : {len(test_df)}")

    print("\nClass distribution:")

    print("\nTRAIN")
    print(train_df["class"].value_counts().sort_index())

    print("\nVALIDATION")
    print(val_df["class"].value_counts().sort_index())

    print("\nTEST")
    print(test_df["class"].value_counts().sort_index())


# ==========================================
# CREATE CROP SPLITS
# ==========================================

create_dataset_splits("crop")


# ==========================================
# CREATE WEATHER SPLITS
# ==========================================

create_dataset_splits("weather")


print("\n" + "=" * 60)
print("ALL SPLITS CREATED")
print("=" * 60)

print(f"Saved inside: {SPLIT_DIR}")