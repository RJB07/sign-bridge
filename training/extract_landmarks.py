#!/usr/bin/env python3
"""
extract_landmarks.py

Scan an images directory (e.g. training/pdf_images/) and extract MediaPipe hand landmarks
for each image. Saves results to `training/landmarks/` as JSON files with the same basename.

Dependencies: opencv-python, mediapipe

Usage:
  python training/extract_landmarks.py --images training/pdf_images --out training/landmarks

"""
import argparse
import os
import json
from pathlib import Path
import cv2

try:
    import mediapipe as mp
except Exception:
    print("Missing dependency mediapipe. Install with: pip install mediapipe opencv-python")
    raise


def extract_from_image(image_path: Path, hands, mp_hands):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    out = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            pts = []
            for lm in hand.landmark:
                pts.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
            out.append(pts)
    return out


def main(images_dir: str, out_dir: str):
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    imgs = sorted(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')))
    if not imgs:
        print(f"No images found in {images_dir}")
        return

    for p in imgs:
        print(f"Processing {p.name}")
        landmarks = extract_from_image(p, hands, mp_hands)
        out_path = out_dir / (p.stem + '.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'image': str(p), 'hands': landmarks}, f, indent=2)

    hands.close()
    print(f"[*] Wrote landmarks for {len(imgs)} images to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--out', required=False, default=str(Path(__file__).parent / 'landmarks'))
    args = parser.parse_args()
    main(args.images, args.out)
