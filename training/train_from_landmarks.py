#!/usr/bin/env python3
"""
train_from_landmarks.py

Simple training harness that loads landmark JSON files and trains a small PyTorch LSTM
classifier. This is a starter script — tune architecture, augmentations, and hyperparameters
for production accuracy.

Dependencies: torch, numpy

Usage:
  python training/train_from_landmarks.py --landmarks training/landmarks --labels training/labels.json

Notes:
 - Expects directory of JSON files produced by `extract_landmarks.py`.
 - `labels.json` should map image basenames to labels, e.g. {"page_0001": "Hello"}

"""
import argparse
import json
import os
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except Exception:
    print("Missing PyTorch. Install with: pip install torch torchvision")
    raise


class LandmarkDataset(Dataset):
    def __init__(self, landmarks_dir: Path, labels_map: dict):
        self.files = sorted(list(landmarks_dir.glob('*.json')))
        self.labels_map = labels_map
        self.classes = sorted(list(set(labels_map.values())))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = json.loads(self.files[idx].read_text())
        base = self.files[idx].stem
        label = self.labels_map.get(base, 'Unknown')
        y = self.class_to_idx.get(label, -1)

        # Flatten first hand landmarks (if present) to a fixed-length vector
        hands = data.get('hands') or []
        if len(hands) == 0:
            x = np.zeros(63, dtype=np.float32)
        else:
            lm = hands[0]  # take primary hand
            arr = []
            for p in lm:
                arr.extend([p['x'], p['y'], p['z']])
            x = np.array(arr, dtype=np.float32)
            if x.size < 63:
                x = np.pad(x, (0, 63 - x.size), 'constant')
            else:
                x = x[:63]

        return torch.from_numpy(x), torch.tensor(max(0, y), dtype=torch.long)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train(landmarks_dir: str, labels_json: str, epochs=10):
    landmarks_dir = Path(landmarks_dir)
    labels_map = json.loads(Path(labels_json).read_text())

    ds = LandmarkDataset(landmarks_dir, labels_map)
    if len(ds) == 0:
        print("No data available in landmarks directory.")
        return

    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = SimpleMLP(63, len(ds.classes))
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total += x.size(0)
            correct += (preds == y).sum().item()

        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch}/{epochs} — Loss: {running_loss/total:.4f} — Acc: {acc:.4f}")

    model_path = Path(__file__).parent / 'trained_model.pt'
    torch.save({'model_state': model.state_dict(), 'classes': ds.classes}, model_path)
    print(f"[*] Saved trained model to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--landmarks', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--epochs', type=int, default=8)
    args = parser.parse_args()
    train(args.landmarks, args.labels, epochs=args.epochs)
