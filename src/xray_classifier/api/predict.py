import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from xray_classifier.models.cnn import XRayCNN


MODEL_PATH = Path("models/xray_cnn.pt")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model(device: torch.device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    class_names = checkpoint["class_names"]

    model = XRayCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(
                (checkpoint["image_size"], checkpoint["image_size"])
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=checkpoint["normalization_mean"],
                std=checkpoint["normalization_std"],
            ),
        ]
    )

    return model, class_names, transform


def predict_image(image_path: Path) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = get_device()
    model, class_names, transform = load_model(device)

    with Image.open(image_path) as image:
        image_tensor = transform(image.convert("RGB"))

    image_batch = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_batch)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = probabilities.argmax().item()

    print(f"\nImage: {image_path.name}")
    print(f"Prediction: {class_names[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index].item():.2%}")

    print("\nAll class probabilities:")
    for class_name, probability in zip(class_names, probabilities):
        print(f"  {class_name:<10} {probability.item():.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the class of one chest X-ray image."
    )
    parser.add_argument("image_path", type=Path)
    arguments = parser.parse_args()

    predict_image(arguments.image_path)


if __name__ == "__main__":
    main()