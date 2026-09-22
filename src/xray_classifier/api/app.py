import io
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
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


device = get_device()

checkpoint = torch.load(MODEL_PATH, map_location=device)

class_names = checkpoint["class_names"]

model = XRayCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint["normalization_mean"],
            std=checkpoint["normalization_std"],
        ),
    ]
)

app = FastAPI(title="Chest X-ray Classifier")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chest X-ray Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 700px;
                margin: 50px auto;
                padding: 0 20px;
            }

            h1 {
                color: #1f2937;
            }

            form {
                display: flex;
                gap: 12px;
                margin: 24px 0;
            }

            button {
                background: #2563eb;
                color: white;
                border: 0;
                border-radius: 6px;
                padding: 10px 16px;
                cursor: pointer;
            }

            #result {
                background: #f3f4f6;
                border-radius: 8px;
                padding: 16px;
                white-space: pre-line;
            }

            .warning {
                color: #991b1b;
            }
        </style>
    </head>
    <body>
        <h1>Chest X-ray Classifier</h1>
        <p>Upload a chest X-ray image to get a model prediction.</p>

        <form id="prediction-form">
            <input id="image-file" type="file" accept="image/*" required>
            <button type="submit">Predict</button>
        </form>

        <div id="result">Choose an image, then click Predict.</div>

        <p class="warning">
            Educational prototype only. Do not use this result for clinical diagnosis.
        </p>

        <script>
            const form = document.getElementById("prediction-form");
            const fileInput = document.getElementById("image-file");
            const result = document.getElementById("result");

            form.addEventListener("submit", async (event) => {
                event.preventDefault();

                if (!fileInput.files.length) {
                    result.textContent = "Please select an image.";
                    return;
                }

                result.textContent = "Predicting...";

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    result.textContent = `Error: ${data.detail}`;
                    return;
                }

                const probabilities = Object.entries(data.probabilities)
                    .map(([name, value]) => `${name}: ${value}%`)
                    .join("\\n");

                result.textContent =
                    `Prediction: ${data.prediction}\\n` +
                    `Confidence: ${data.confidence}%\\n\\n` +
                    `All probabilities:\\n${probabilities}`;
            });
        </script>
    </body>
    </html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as an image.",
        ) from error

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = probabilities.argmax().item()

    probability_map = {
        class_name: round(probability.item() * 100, 2)
        for class_name, probability in zip(class_names, probabilities)
    }

    return {
        "prediction": class_names[predicted_index],
        "confidence": round(probabilities[predicted_index].item() * 100, 2),
        "probabilities": probability_map,
    }