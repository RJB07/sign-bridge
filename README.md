# SignBridge

SignBridge turns isolated American Sign Language into English text in the browser. The camera stays on the device. Only ASL is implemented. The language pack in `src/languages/aslEnglish.ts` is the place a later sign language would plug in.

This is isolated-gloss recognition. A correct result is an English word such as “book” or “brother”, not a translated sentence and not full ASL grammar.

## Run the app

```bash
npm install
npm test
npm run dev
```

Open the URL Vite prints and allow the camera. The overlay is mirrored. **Replay held-out sign** runs a real test-set clip through the same classifier without a camera.

`npm run build` typechecks and writes a static site. `SIGNATURE.json` must keep the lead attribution or the prebuild check fails.

## What you should expect

The shipped model is a small conv + GRU over MediaPipe Holistic landmarks. On PopSign signers held out of training (31 train / 8 validation / 8 test, no signer in two splits):

| | |
| --- | --- |
| Gloss vocabulary | 32 English words (TV through brown, including book, boy, bird, brother) |
| Test clips | 447 |
| Top-1 | **59.1%** |
| Top-5 | **84.3%** |
| Macro recall | 59.5% |
| Chance top-1 | 3.1% |
| Live gate on test | 69.7% precision when it commits, on 77.6% of clips (threshold 0.30, margin 0.04, chosen on validation) |

Per-class precision and recall are in `training/metrics/asl_en_metrics.json`. Recall is high for apple, blow, and bed, and low for bad, better, and brother, which are easy to confuse with nearby signs. The same summary is embedded in `public/models/asl-en-sequence.json`.

Live output is suppressed unless the score clears a threshold picked on the validation signers. A gloss is added to the phrase once per motion segment. After a pause, the line is spoken with the Web Speech API.

Fingerspelling is a separate geometric helper for the handshapes A, B, C, F, I, L, V, W, and Y. It is not a measured letter-recognition model. J and Z are not supported.

## Dataset and license

Training uses PopSign ASL v1.0 (game subset), CC BY 4.0, via the frame release [`sign/popsign-images`](https://huggingface.co/datasets/sign/popsign-images). Cite Starner et al., PopSign ASL v1.0 (2023). Landmark files are derived and are not a way to identify the signers.

The download list is `training/download_popsign.py`. Shards are chosen so each trained gloss also appears in the official validation and test partitions. `training/splits.py` stops the run if any signer id appears in more than one partition.

Not used for the model, and why:

- **WLASL / MS-ASL** — video rights are research-only. The WLASL index JSON in this repo is not training data.
- **ASL Citizen** — non-commercial research license; the data and modifications may not be redistributed.
- **FLEURS-ASL** — sentence translation, CC BY-SA, wrong task for a small isolated-gloss model.
- **Google Isolated Sign Language Recognition (Kaggle)** — same PopSign signs, already landmarks, CC BY 4.0, about 56 GB and a Kaggle account. Use it if you want the full 250 glosses.

## Retrain

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt
# Linux also needs libEGL and libGL for MediaPipe
PYTHONPATH=training python training/download_popsign.py
PYTHONPATH=training python training/extract_landmarks.py
PYTHONPATH=training python training/train_sequence_model.py
```

Extraction writes `training/cache/` (gitignored). Training writes:

- `public/models/asl-en-sequence.json` — weights the app loads
- `public/models/demo-clips.json` — held-out replay clips
- `public/models/parity.json` — logits the browser port must match
- `training/metrics/asl_en_metrics.json`

To cover more of the 250 PopSign glosses, add later alphabetical shards to `training/download_popsign.py` for train, validation, and test together, then extract and train again. A larger hidden size or a full pass over every shard is the same script with a higher `--max-train`.

## Layout

- `src/tracking/MediaPipeTracker.ts` — camera and Holistic landmarks
- `src/recognition/` — features, sequence model, live segmenter, fingerspelling
- `src/textAssembly/SignTextAssembler.ts` — phrase buffer and English punctuation
- `src/languages/aslEnglish.ts` — the only language pack
- `training/` — download, landmark extraction, training
- `docs/AUDIT.md` — architecture notes, including why there is no database

## Limits

Isolated signs only. The baseline vocabulary is the PopSign glosses present in every split of the downloaded shards (A through “brown”, including book, boy, bird, brother). Two-handed signs, continuous sentences, and conversational phrases such as “thank you” are outside this model. Training clips are short and about 5 frames per second. Do not treat the confidence percentage as a clinical measurement.

## Database

None. Samples the user imports stay in `localStorage`. The transcript is not saved. Details are in `docs/AUDIT.md`.
