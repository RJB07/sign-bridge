# Third-Party Licenses & Credits

SignBridge is 100% free and open source under the MIT License. It relies on the following open-source libraries, pretrained model checkpoints, and open-domain datasets:

---

## 1. Core Runtime Libraries & Frameworks

### React
- **License**: MIT License
- **URL**: https://github.com/facebook/react

### MediaPipe Tasks Vision (`@mediapipe/tasks-vision`) and the Holistic landmarker
- **License**: Apache License 2.0
- **Copyright**: Google LLC
- **URL**: https://github.com/google/mediapipe

### Vite
- **License**: MIT License
- **URL**: https://github.com/vitejs/vite

---

## 2. Datasets and models used by the recognizer

### PopSign ASL v1.0
- **License**: CC BY 4.0
- **Used for**: the shipped isolated-gloss model
- **What we store**: model weights and a few normalized landmark sequences for replay, not the videos
- **URL**: https://signdata.cc.gatech.edu/ and https://huggingface.co/datasets/sign/popsign-images
- **Cite**: Thad Starner et al., PopSign ASL v1.0 (NeurIPS 2023)

### MediaPipe Holistic landmarker and vision WASM
- **License**: Apache License 2.0
- **Copyright**: Google LLC
- **URL**: https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
- **Vendored for offline use**: `public/models/holistic_landmarker.task` and `public/vendor/mediapipe/` (`vision_wasm_internal` and `vision_wasm_nosimd_internal` from `@mediapipe/tasks-vision` 1.2.0). Phones without WASM SIMD load the nosimd build.

## 3. Datasets considered and not used for training

### WLASL
- Research use. This repo keeps the public gloss index only. Videos are not redistributable and were not used to train the model.

### ASL Citizen
- Microsoft research license: non-commercial, and the data or modifications may not be distributed. Not used.

### FLEURS-ASL
- CC BY-SA 4.0. Continuous translation, not isolated glosses. Not used, so the model is not share-alike.

### Google Isolated Sign Language Recognition (Kaggle) and ASL Fingerspelling Recognition
- Landmark releases of PopSign and of fingerspelling phrases. CC BY 4.0. Full downloads are tens to hundreds of gigabytes and need a Kaggle account. Not used yet. The letter model in the app is trained on the MIT set `sid220/asl-now-fingerspelling`, and that holdout is not signer-independent. See `docs/KAGGLE_DATASETS.md`.

### How2Sign
- CC BY-NC 4.0. Not used.

## 4. Audio

### Web Speech API
- Browser speech synthesis and recognition. Audio is not sent to a SignBridge server.
