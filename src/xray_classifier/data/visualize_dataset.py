from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


DATASET_ROOT = Path("/Users/afaquemogni/Downloads/deep_learning/Data")
TRAIN_DIR = DATASET_ROOT / "train"
OUTPUT_DIR = Path("reports/figures")

CLASSES = ("COVID19", "NORMAL", "PNEUMONIA")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SAMPLES_PER_CLASS = 3


def get_image_files(folder: Path) -> list[Path]:
    return [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def plot_class_distribution() -> None:
    class_counts = {
        class_name: len(get_image_files(TRAIN_DIR / class_name))
        for class_name in CLASSES
    }

    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(class_counts.keys(), class_counts.values())

    axis.set_title("Training-set Class Distribution")
    axis.set_xlabel("Class")
    axis.set_ylabel("Number of images")

    for bar, count in zip(bars, class_counts.values()):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            count,
            str(count),
            ha="center",
            va="bottom",
        )

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close(figure)


def plot_sample_images() -> None:
    random_generator = random.Random(42)

    figure, axes = plt.subplots(
        nrows=len(CLASSES),
        ncols=SAMPLES_PER_CLASS,
        figsize=(12, 10),
    )

    for row, class_name in enumerate(CLASSES):
        image_files = get_image_files(TRAIN_DIR / class_name)
        selected_files = random_generator.sample(image_files, SAMPLES_PER_CLASS)

        for column, image_path in enumerate(selected_files):
            with Image.open(image_path) as image:
                axes[row, column].imshow(image.convert("RGB"))

            axes[row, column].set_title(class_name)
            axes[row, column].axis("off")

    figure.suptitle("Random Chest X-ray Samples from Each Class", fontsize=16)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "sample_xrays.png", dpi=150)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_class_distribution()
    plot_sample_images()

    print(f"Saved figures to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
