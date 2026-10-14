from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_images(folder):

    return [
        file
        for file in folder.rglob("*")
        if (
            file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]


def show_samples(dataset_type, samples_per_class=3):

    dataset_dir = DATASET_DIR / dataset_type

    class_dirs = sorted(
        [
            folder
            for folder in dataset_dir.iterdir()
            if folder.is_dir()
        ]
    )

    for class_dir in class_dirs:

        images = get_images(class_dir)

        selected_images = images[:samples_per_class]

        plt.figure(figsize=(12, 4))

        for i, image_path in enumerate(selected_images):

            try:

                image = Image.open(image_path)

                plt.subplot(
                    1,
                    len(selected_images),
                    i + 1
                )

                plt.imshow(image)

                plt.title(class_dir.name)

                plt.axis("off")

            except Exception as e:

                print(
                    f"Could not open {image_path}: {e}"
                )

        plt.suptitle(
            f"{dataset_type.upper()} - {class_dir.name}"
        )

        plt.tight_layout()

        plt.show()


print("=" * 50)
print("CROP SAMPLE IMAGES")
print("=" * 50)

show_samples(
    "crop",
    samples_per_class=3
)


print("\n" + "=" * 50)
print("WEATHER SAMPLE IMAGES")
print("=" * 50)

show_samples(
    "weather",
    samples_per_class=3
)