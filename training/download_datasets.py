#!/usr/bin/env python3
"""
SignBridge — Multimodal Open-Domain ASL & MediaPipe Holistic Dataset Ingestion Tool
Downloads and indexes open-domain ASL datasets:
 1. WLASL & WLASL100 (Word-Level ASL Classification: 100-2000 words)
 2. Google AI FLEURS-ASL (googleai/fleurs-asl benchmark)
 3. MediaPipe Holistic 540+ Landmark Sequences
 4. Open-Domain ASL Book Ground Truth Library
"""

import os
import sys
import json
import urllib.request

DATASET_DIR = os.path.join(os.path.dirname(__file__), "raw_datasets")

def init_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"[*] Initialized dataset directory at: {DATASET_DIR}")

def download_wlasl_index():
    url = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
    target_path = os.path.join(DATASET_DIR, "WLASL_v0.3.json")
    if not os.path.exists(target_path):
        print(f"[*] Downloading WLASL index from {url}...")
        try:
            urllib.request.urlretrieve(url, target_path)
            print("[+] WLASL index downloaded successfully.")
        except Exception as e:
            print(f"[!] Warning: Could not download WLASL index: {e}")
    else:
        print("[+] WLASL index already present.")

def init_wlasl100_manifest():
    wlasl100_path = os.path.join(DATASET_DIR, "wlasl100_manifest.json")
    manifest = {
        "dataset_name": "WLASL100 Video Classification Dataset",
        "kaggle_url": "https://www.kaggle.com/datasets/nadimsrabon6/wlasl100-video-classification-dataset",
        "num_classes": 100,
        "features": ["Handshape", "Trajectory", "Spatial Location"],
        "status": "INDEX_ONLY_NOT_USED"
    }
    with open(wlasl100_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Initialized WLASL100 dataset manifest at {wlasl100_path}")

def init_fleurs_asl_manifest():
    fleurs_path = os.path.join(DATASET_DIR, "fleurs_asl_manifest.json")
    manifest = {
        "dataset_name": "FLEURS-ASL",
        "description": "Google DeepMind Sign Language to Text (SL2T) Landmark Sequence Benchmark",
        "kaggle_url": "https://www.kaggle.com/datasets/googleai/fleurs-asl",
        "mediapipe_url": "https://research.google/blog/mediapipe-holistic-simultaneous-face-hand-and-pose-prediction-on-device/",
        "status": "NOT_DOWNLOADED_NOT_USED",
        "recommended_download_cmd": "python -c 'import kagglehub; path = kagglehub.dataset_download(\"googleai/fleurs-asl\"); print(path)'"
    }
    with open(fleurs_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Initialized FLEURS-ASL dataset manifest at {fleurs_path}")

def init_mediapipe_holistic_manifest():
    holistic_path = os.path.join(DATASET_DIR, "mediapipe_holistic_540_manifest.json")
    manifest = {
        "tracker": "MediaPipe Holistic On-Device Predictor",
        "total_landmarks": 540,
        "breakdown": {
          "hands_left": 21,
          "hands_right": 21,
          "face_mesh": 468,
          "pose_body_skeleton": 33
        },
        "privacy_mode": "COORDINATE_STREAMING_ONLY",
        "handedness_support": ["AUTO", "LEFT", "RIGHT"],
        "dictation_modes": ["DUAL_HAND", "ONE_HANDED_MOBILE"]
    }
    with open(holistic_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Initialized MediaPipe Holistic 540+ Landmark manifest at {holistic_path}")

if __name__ == "__main__":
    print("=" * 65)
    print("SignBridge — Open-Domain ASL, WLASL100 & FLEURS-ASL Dataset Ingestion")
    print("=" * 65)
    init_dirs()
    download_wlasl_index()
    init_wlasl100_manifest()
    init_fleurs_asl_manifest()
    init_mediapipe_holistic_manifest()
