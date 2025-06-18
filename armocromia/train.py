"""Example training script for the Deep Armocromia dataset.

Usage:
    python -m armocromia.train --csv annotations.csv --root data/ \
                               --epochs 10 --batch-size 32

This script assumes you have already downloaded and extracted the dataset as
explained in the README. It trains a simple ResNet model on the dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from .dataset import ArmocromiaDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model on Deep Armocromia")
    parser.add_argument("--csv", type=Path, required=True, help="Path to annotations.csv")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = ArmocromiaDataset(args.csv, args.root, partition="train", transform=transform)
    val_ds = ArmocromiaDataset(args.csv, args.root, partition="val", transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.label_map))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # simple validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / total if total else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} - val accuracy: {acc:.2f}%")

    Path("models").mkdir(exist_ok=True)
    model_path = Path("models/armocromia_resnet18.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
