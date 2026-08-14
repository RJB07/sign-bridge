# Third-Party Licenses & Credits

SignBridge is 100% free and open source under the MIT License. It relies on the following open-source libraries, pretrained model checkpoints, and open-domain datasets:

---

## 1. Core Runtime Libraries & Frameworks

### React
- **License**: MIT License
- **URL**: https://github.com/facebook/react

### MediaPipe Tasks Vision & Libraries (`@mediapipe/hands`, `@mediapipe/face_mesh`, `@mediapipe/camera_utils`)
- **License**: Apache License 2.0
- **Copyright**: Google LLC
- **URL**: https://github.com/google/mediapipe

### TensorFlow.js (`@tensorflow/tfjs`)
- **License**: Apache License 2.0
- **Copyright**: Google LLC
- **URL**: https://github.com/tensorflow/tfjs

### Vite
- **License**: MIT License
- **URL**: https://github.com/vitejs/vite

---

## 2. Machine Learning Datasets & Models

### ASL Alphabet Dataset (Fingerspelling Phase A)
- **License**: Public Domain / CC0 (Kaggle Open Dataset)
- **Description**: 87,000 images of ASL alphabet signs (A-Z, 0-9) used for landmark extraction training.

### WLASL — Word-Level ASL Dataset (Common Signs Phase B)
- **License**: Open Academic & Non-Commercial / BSD License
- **Description**: ~2,000 word-level video dataset for isolated sign language recognition.
- **URL**: https://github.com/dxli94/WLASL

### AI4Bharat OpenHands
- **License**: Apache License 2.0
- **Copyright**: AI4Bharat
- **Description**: Pretrained pose/hand landmark Transformer & LSTM models for sign language recognition.
- **URL**: https://github.com/AI4Bharat/OpenHands

### How2Sign (Continuous Signing Phase C - Stretch Goal)
- **License**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
- **Description**: Multimodal dataset of continuous American Sign Language.
- **URL**: https://how2sign.github.io/

---

## 3. Audio & Speech Engines

### Web Speech API (`SpeechSynthesis` & `SpeechRecognition`)
- **License**: Browser Native Standards (W3C Specification)
- **Description**: Client-side text-to-speech and speech-to-text execution inside user agent.

### Whisper.cpp / Transformers.js Whisper (Client-Side Reverse Channel)
- **License**: MIT License / Apache 2.0
- **Description**: On-device speech-to-text inference compiled to WebAssembly / WebGPU.
- **URL**: https://github.com/ggerganov/whisper.cpp
