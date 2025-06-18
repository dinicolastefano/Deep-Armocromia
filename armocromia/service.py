"""Simple FastAPI service for classifying new images using a trained model."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import models, transforms

from .dataset import ArmocromiaDataset


app = FastAPI(title="Deep Armocromia Service")

# global variables set in `load_model`
model = None
label_map = None
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model(weights_path: str | Path, csv_path: str | Path) -> None:
    """Load trained model and label map from disk."""
    global model, label_map
    csv_path = Path(csv_path)
    ds = ArmocromiaDataset(csv_path, csv_path.parent, partition="train")
    label_map = ds.label_map

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(label_map))
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()


@app.on_event("startup")
def startup_event():
    weights = Path("models/armocromia_resnet18.pt")
    csv = Path("annotations.csv")
    if weights.exists() and csv.exists():
        load_model(weights, csv)
    else:
        print("Model weights or annotations not found. Service will not predict.")


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, str]:
    if model is None or label_map is None:
        return {"error": "model not loaded"}
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(img)
        pred_idx = logits.argmax(dim=1).item()
    label = [k for k, v in label_map.items() if v == pred_idx][0]
    return {"label": label}

