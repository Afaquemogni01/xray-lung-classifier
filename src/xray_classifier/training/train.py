from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from xray_classifier.models.cnn import XRayCNN


DATASET_ROOT = Path("/Users/afaquemogni/Downloads/deep_learning/Data")
MODEL_PATH = Path("models/xray_cnn.pt")

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.manual_seed(42)

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)

train_dataset = datasets.ImageFolder(
    DATASET_ROOT / "train",
    transform=transform,
)

test_dataset = datasets.ImageFolder(
    DATASET_ROOT / "test",
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_dataset.classes

model = XRayCNN(num_classes=len(class_names)).to(device)

class_counts = torch.bincount(torch.tensor(train_dataset.targets)).float()
class_weights = class_counts.sum() / (len(class_counts) * class_counts)

loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Device: {device}")
print(f"Class mapping: {dict(enumerate(class_names))}")
print(f"Training images: {len(train_dataset)}")
print(f"Test images: {len(test_dataset)}")

for epoch in range(1, EPOCHS + 1):
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_images += labels.size(0)

    average_loss = total_loss / total_images
    training_accuracy = correct_predictions / total_images

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"loss: {average_loss:.4f} | "
        f"training accuracy: {training_accuracy:.2%}"
    )

model.eval()

total_test_loss = 0.0
correct_test_predictions = 0
total_test_images = 0

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_function(logits, labels)

        total_test_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)
        correct_test_predictions += (predictions == labels).sum().item()
        total_test_images += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

test_loss = total_test_loss / total_test_images
test_accuracy = correct_test_predictions / total_test_images

print(f"\nFinal test loss: {test_loss:.4f}")
print(f"Final test accuracy: {test_accuracy:.2%}\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "image_size": IMAGE_SIZE,
    "normalization_mean": [0.5, 0.5, 0.5],
    "normalization_std": [0.5, 0.5, 0.5],
}

torch.save(checkpoint, MODEL_PATH)

print(f"Saved model checkpoint: {MODEL_PATH.resolve()}")