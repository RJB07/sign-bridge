# SignBridge

SignBridge turns one American Sign Language gloss at a time into English text. The camera and the recognizer stay on the device. Nothing is uploaded. Only ASL is implemented. A later sign language would plug in at `src/languages/aslEnglish.ts`.

This is isolated-gloss recognition. A correct result is an English word such as “hello” or “thank you”, not a translated sentence and not full ASL grammar.

![Replay of held-out signs, undo, and a typed reply](docs/demo.gif)

## Quick start

```bash
npm install
npm test
npm run dev
```

Open the URL Vite prints.

1. Allow the camera, or press **Replay a sample**. That runs a real held-out PopSign clip through the same classifier. The first samples are hello, thank you, and please.
2. The English word appears in the large caption. **Undo** removes the last word (Backspace does the same when you are not typing). Tap a word to remove that one.
3. A hearing person types in **Their words**. Speech-to-text is optional. The reply is shown as text. **Speaking aloud** stays off until you turn it on for them.
4. Caption size, and light, dark, and high-contrast themes, are in the top bar. On a phone the caption stays above the camera.

Install: in Chrome, Edge, or Safari, use **Install SignBridge** or **Add to Home Screen**. The first visit has to be online so the app can store the recognizer and the MediaPipe tracker. After that, captions, replay, and the camera work offline. The page uses system fonts, so text does not wait on Google.

```bash
npm run build          # static site plus the service worker
npm run test:e2e       # replay, undo, typed reply, contrast theme, offline reload
```

`npm run build` checks `SIGNATURE.json` first. That file has to keep the lead attribution. End-to-end tests need Chrome (`CHROME_PATH` if it is not on the default path). GitHub Actions runs the unit tests, the Python split checks, the build, and the browser test. It does not download PopSign.

## Using it

The live caption is the largest text on the screen. Sound is off until you turn on “Speaking aloud” for a hearing person. The guess shown while the hand is still moving uses one ensemble member and no mirror, so a low-end phone can keep up. The word that is actually added uses the full model. A meter and a short “Added …” line mark the commit. If the hand is missing, too close, or out of frame, the caption says so.

## What you should expect

The shipped recognizer is three landmark transformers (width 128, 2 layers, float16 weights) averaged together. It sees both hands, arm pose, and frame-to-frame motion. On PopSign’s official signer split (31 train / 8 validation / 8 test, no shared signer):

| | |
| --- | --- |
| Gloss vocabulary | **250** English glosses, the full PopSign / Google ASL Signs list |
| Train / val / test clips | 9198 / 2799 / 2744 |
| Test top-1 | **50.6%** |
| Test top-5 | **75.1%** |
| Macro recall | 50.7% |
| Chance top-1 | 0.4% |
| Live gate on test | 74.4% precision when it commits, on 43.2% of clips |

The same test set, scored the same way, was 50.4% top-1 and 75.7% top-5 for the previous two-member model. This run’s gain is 5 clips out of 2,744. Mirror test-time augmentation added 0.04 points (50.58% to 50.62%). Float16 weights matched that top-1, so the file stores float16 (about 3.0 MB). A distilled one-member student reached 44.5% and was not shipped.

A 32-gloss model earlier on this branch scored 59.1% top-1. That number is not comparable: this model chooses among 250 signs. Chance falls from 3.1% to 0.4%. Conversational glosses are in the 250. Per-class recall on the small test counts: hello 64% (11 clips), thank you 63% (8), please 70% (10), yes 50% (10), no 60% (10), water 91% (11), mom 50% (12), dad 64% (11). **Help is not in this 250-sign list.**

Per-class precision and recall: `training/metrics/asl_en_metrics.json`.

Fingerspelling is a separate static network trained on the MIT dataset `sid220/asl-now-fingerspelling` (A–Z, including J and Z). Filename-hash holdout top-1 is **92.3%** (222 test frames). J recall 91% (11 frames), Z recall 100% (10 frames). **Signer ids are not in that dataset, so this letter number is not signer-independent.** Details: `training/metrics/fingerspelling_metrics.json`.

Live output is suppressed unless the score clears a threshold picked on the validation signers, including a per-class floor when the model file has one. A gloss is added once per motion segment. **Send line** commits the caption into the conversation. Speaking aloud, if you turned it on, reads that line to the hearing person. It does not run by default.

Fingerspelling letters A–Z, including J and Z, come from a static landmark network. See the metrics file for the holdout caveat.

## Dataset and license

Training uses PopSign ASL v1.0 (game subset), CC BY 4.0, via the frame release [`sign/popsign-images`](https://huggingface.co/datasets/sign/popsign-images). Cite Starner et al., PopSign ASL v1.0 (2023). Landmark files are derived and are not a way to identify the signers.

The download list is `training/download_popsign.py`. Shards are chosen so each trained gloss also appears in the official validation and test partitions. `training/splits.py` stops the run if any signer id appears in more than one partition.

Not used for the model, and why:

- **WLASL / MS-ASL** — video rights are research-only. The WLASL index JSON in this repo is not training data.
- **ASL Citizen** — non-commercial research license; the data and modifications may not be redistributed.
- **FLEURS-ASL** — sentence translation, CC BY-SA, wrong task for a small isolated-gloss model.
- **Google Isolated Sign Language Recognition (Kaggle)** — same 250 glosses, already landmarks, CC BY 4.0. Not downloaded for this model. See below.

## More data, when a Kaggle token exists

`docs/KAGGLE_DATASETS.md` lists the two downloads that are worth plugging in later: the full Google ASL Signs landmark release (same 250 glosses, native frame rate, many more examples per sign) and the Google ASL Fingerspelling release (signer ids and the motion of J and Z). ASL Citizen and WLASL are not on that list.

## Retrain

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt
# Linux also needs libEGL and libGL for MediaPipe
PYTHONPATH=training python training/download_popsign.py
PYTHONPATH=training python training/extract_landmarks.py
PYTHONPATH=training python training/train_sequence_model.py
```

Extraction writes `training/cache/` (gitignored). The trainer compares a three-seed ensemble with a smaller distilled student, with and without mirror test-time augmentation, and it tries float16 weights when the test top-1 drop stays within half a point. It replaces `public/models/asl-en-sequence.json` only when the signer-disjoint test top-1 is higher than the file already there. Either way it writes `training/metrics/asl_en_run.json`.

When a run is shipped it also writes:

- `public/models/asl-en-sequence.json` — weights the app loads
- `public/models/demo-clips.json` — held-out replay clips
- `public/models/parity.json` — logits the browser port must match
- `training/metrics/asl_en_metrics.json`

The shipped vocabulary is already the full 250 PopSign glosses. A larger download of the same signs is the Kaggle landmark release in `docs/KAGGLE_DATASETS.md`, not more PopSign image shards.

## Layout

- `src/tracking/MediaPipeTracker.ts` — camera and Holistic landmarks
- `src/recognition/` — features, sequence model, live segmenter, fingerspelling
- `src/textAssembly/SignTextAssembler.ts` — phrase buffer and English punctuation
- `src/languages/aslEnglish.ts` — the only language pack
- `training/` — download, landmark extraction, training
- `docs/AUDIT.md` — architecture notes, including why there is no database

## Limits

- The vocabulary is the 250 PopSign glosses. Help is not one of them. Two-handed signs are represented when both hands are in frame, but PopSign is mostly one-handed phone signing.
- Training clips are short and about 5 frames per second.
- Fingerspelling holdout is not signer-independent.
- Offline use still needs the first online load, or a build that already contains `public/vendor/mediapipe/` and `public/models/holistic_landmarker.task`.
- The app JavaScript does not load TensorFlow.js. Camera tracking is MediaPipe Holistic, about 14 MB of WASM plus the task file, cached by the service worker.
- Do not treat the confidence percentage as a clinical measurement.

## Database

None. Samples the user imports stay in `localStorage`. The transcript is not saved. Details are in `docs/AUDIT.md`.
