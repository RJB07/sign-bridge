SignBridge training helpers
===========================

This folder contains helper scripts to ingest a PDF sign-language book, extract hand landmarks
from page images, and run a baseline training pass.

1) Ingest PDF pages to images + OCR

  pip install pdf2image pillow pytesseract
  # On macOS you may need to `brew install poppler tesseract` for pdf2image/pytesseract

  python training/ingest_pdf.py /path/to/signbook.pdf

  This writes `training/pdf_images/*.png` and `training/pdf_index.json`.

2) Extract MediaPipe hand landmarks for each saved image

  pip install mediapipe opencv-python
  python training/extract_landmarks.py --images training/pdf_images --out training/landmarks

  This writes per-image JSON files to `training/landmarks/` describing detected hand landmarks.

3) Create a labels map

  Create `training/labels.json` mapping each image basename to a canonical label. Example:

  {
    "page_0001": "Hello",
    "page_0002": "Thank you",
    "page_0003": "K"  # fingerspelled letters may be single chars
  }

4) Train a baseline classifier (PyTorch)

  pip install torch torchvision
  python training/train_from_landmarks.py --landmarks training/landmarks --labels training/labels.json --epochs 12

Notes & next steps
- The provided scripts are a starting point. For robust live recognition you should:
  - Build a temporal model (LSTM/Transformer) on sequences of frames, not single-image MLP.
  - Augment data and include video samples, not only single page images.
  - Respect copyright and site usage rules for any scraped content.

If you'd like, I can:
- Add a small UI in `src/` to upload a PDF and run ingestion in-browser (limited).
- Implement a full temporal trainer and export to TF.js for in-browser inference.
