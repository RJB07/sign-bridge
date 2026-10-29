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

The shipped recognizer is an ensemble of two small landmark transformers. It sees both hands, arm pose, and frame-to-frame motion. On PopSign’s official signer split (31 train / 8 validation / 8 test, no shared signer):

| | |
| --- | --- |
| Gloss vocabulary | **250** English glosses, the full PopSign / Google ASL Signs list |
| Train / val / test clips | 9198 / 2799 / 2744 |
| Test top-1 | **50.4%** |
| Test top-5 | **75.7%** |
| Macro recall | 50.4% |
| Chance top-1 | 0.4% |
| Live gate on test | 72.6% precision when it commits, on 56.7% of clips |

A 32-gloss model earlier on this branch scored 59.1% top-1. That number is not comparable: this model chooses among 250 signs. Chance falls from 3.1% to 0.4%. Conversational glosses are in the 250, with noisy per-class recall on the small test counts: thank you 75% (8 clips), no 60% (10), please 50% (10), dad 64% (11), hello 27% (11), yes 30% (10). **Help is not in this 250-sign list.**

Per-class precision and recall: `training/metrics/asl_en_metrics.json`.

Fingerspelling is a separate static network trained on the MIT dataset `sid220/asl-now-fingerspelling` (A–Z, including J and Z). Filename-hash holdout top-1 is **92.3%** (222 test frames). J recall 91% (11 frames), Z recall 100% (10 frames). **Signer ids are not in that dataset, so this letter number is not signer-independent.** Details: `training/metrics/fingerspelling_metrics.json`.

Live output is suppressed unless the score clears a threshold picked on the validation signers. A gloss is added to the phrase once per motion segment. After a pause, the line is spoken with the Web Speech API.

Fingerspelling letters A–Z, including J and Z, come from a static landmark network. See the metrics file for the holdout caveat. A pause joins glosses into a line of English and speaks it.

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

- The vocabulary is the 250 PopSign glosses. Help is not one of them. Two-handed signs are represented when both hands are in frame, but PopSign is mostly one-handed phone signing.
- Training clips are short and about 5 frames per second.
- Fingerspelling holdout is not signer-independent.
- Do not treat the confidence percentage as a clinical measurement.

## Database

None. Samples the user imports stay in `localStorage`. The transcript is not saved. Details are in `docs/AUDIT.md`.
