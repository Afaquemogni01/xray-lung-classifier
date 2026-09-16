from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


DATASET_ROOT = Path("/Users/afaquemogni/Downloads/deep_learning/Data")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_image_files(folder: Path) -> list[Path]:
    return [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def inspect_split(split_name: str) -> None:
    split_path = DATASET_ROOT / split_name

    if not split_path.exists():
        raise FileNotFoundError(f"Missing split folder: {split_path}")

    class_folders = sorted(path for path in split_path.iterdir() if path.is_dir())

    print(f"\n{'=' * 60}")
    print(f"{split_name.upper()} SPLIT")
    print(f"{'=' * 60}")

    class_counts = Counter()
    image_sizes = Counter()
    image_modes = Counter()
    invalid_files = []

    for class_folder in class_folders:
        image_files = get_image_files(class_folder)
        class_counts[class_folder.name] = len(image_files)

        for image_path in image_files:
            try:
                with Image.open(image_path) as image:
                    image.verify()

                with Image.open(image_path) as image:
                    image_sizes[image.size] += 1
                    image_modes[image.mode] += 1

            except (UnidentifiedImageError, OSError) as error:
                invalid_files.append((image_path, str(error)))

    total_images = sum(class_counts.values())

    print("\nClass counts:")
    for class_name, count in class_counts.items():
        percentage = (count / total_images) * 100 if total_images else 0
        print(f"  {class_name:<12} {count:>5} images ({percentage:>5.1f}%)")

    print(f"\nTotal images: {total_images}")
    print(f"Invalid images: {len(invalid_files)}")

    print("\nImage modes:")
    for mode, count in image_modes.most_common():
        print(f"  {mode}: {count}")

    print("\nMost common image sizes:")
    for size, count in image_sizes.most_common(10):
        print(f"  {size}: {count}")


def main() -> None:
    print(f"Dataset location: {DATASET_ROOT}")

    for split_name in ("train", "test"):
        inspect_split(split_name)


if __name__ == "__main__":
    main()