# SignBridge

SignBridge is an open-source browser-based sign language prototype with camera tracking, model training, and local persistence.

## What it does

- Opens the webcam and tracks hand landmarks using MediaPipe Hands
- Displays landmarks and hand connections in real time
- Captures labeled training samples locally in the browser
- Trains a small TensorFlow.js model in the browser
- Saves the trained model and labels in browser storage
- Converts recognized sign labels to speech using built-in TTS

## Features

- Live hand tracking with camera overlay
- Capture training samples for custom gesture labels
- Browser-based training with TensorFlow.js
- Local model storage in `localStorage`
- Clear dataset and model with one button

## Getting started

1. Install dependencies:

```bash
npm install
```

2. Run the development server:

```bash
npm run dev
```

3. Open the local URL shown by Vite and allow camera access.

## How to use

1. Allow camera access.
2. Select a label from the dropdown.
3. Capture a few samples of the gesture.
4. Train the model.
5. Show the gesture again to test live predictions.

## Project structure

- `src/App.tsx` — main UI, capture, training, and prediction flow
- `src/signModel.ts` — TensorFlow.js model and dataset utilities
- `src/storage.ts` — browser storage helpers for dataset and label persistence
- `src/main.tsx` — React entry point
- `src/styles.css` — app styling

## What to build next

- Add dataset export/import for sharing labeled samples and import existing sign datasets
- Expand from 3 labels to a larger sign-language vocabulary
- Replace the sample classifier with a more advanced open model
- Add phrase building, punctuation, and full ASL grammar support
- Add translation or speech-to-text integration

## License

This project is open source and licensed under the MIT License.
