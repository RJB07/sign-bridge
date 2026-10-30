# SignBridge

SignBridge turns one American Sign Language gloss at a time into English text. The camera and the recognizer stay on the device. Nothing is uploaded. Only ASL is implemented. A later sign language would plug in at `src/languages/aslEnglish.ts`.

The recognizer still reads one gloss at a time. On-device rules then turn the glosses you commit into a short English sentence, and the glosses stay visible under it. That is word order and a few fixed phrases, not full ASL grammar.

The public demo is [https://rjb07.github.io/sign-bridge/](https://rjb07.github.io/sign-bridge/). It appears after a push to `main` once GitHub Pages is set to deploy from GitHub Actions. `CONTRIBUTING.md` has that one-time setting.

![Replay of held-out signs, undo, and a typed reply](docs/demo.gif)

## Quick start

```bash
npm install
npm test
npm run dev
```

Open the URL Vite prints.

1. Allow the camera, or press **Replay a sample**. That runs a real held-out PopSign clip through the same classifier. The first samples are hello, thank you, and please.
2. The large line is an English sentence such as “Hello.” The glosses stay underneath. **Undo** removes the last gloss (Backspace does the same when you are not typing). Tap a gloss to remove it. **Send line** puts the sentence in the conversation and leaves the glosses on that message.
3. **Your signs** saves the last sign under a label you choose. It stays in this browser, video is not stored, and a close match can replace the shared guess.
4. A hearing person types in **Their words**. Speech-to-text is optional. The reply is shown as text. **Speaking aloud** stays off until you turn it on for them.
5. Caption size, and light, dark, and high-contrast themes, are in the top bar. On a phone the caption stays above the camera.

Install: in Chrome, Edge, or Safari, use **Install SignBridge** or **Add to Home Screen**. The first visit has to be online so the app can store the recognizer and the MediaPipe tracker. After that, captions, replay, and the camera work offline. The page uses system fonts, so text does not wait on Google.

```bash
npm run build          # static site plus the service worker
npm run test:e2e       # replay, undo, typed reply, contrast theme, offline reload
```

`npm run build` checks `SIGNATURE.json` first. That file has to keep the lead attribution. End-to-end tests need Chrome (`CHROME_PATH` if it is not on the default path). GitHub Actions runs the unit tests, the Python split checks, the build, and the browser test. It does not download PopSign.

## Using it

The English sentence is the largest text on the screen. The gloss chips under it are what the recognizer committed. Sound is off until you turn on “Speaking aloud” for a hearing person. The guess shown while the hand is still moving uses one ensemble member and no mirror, and it is not recomputed on every camera frame, so a low-end phone can keep up. The word that is actually added uses the full model, then your saved signs if one is very close. This model file does not mirror the clip a second time. Camera frames are not uploaded. Speech to text, off until you press it, sends microphone audio to the browser’s speech service and not to SignBridge. A meter and a short “Added …” line mark the commit. If the hand is missing, too close, or out of frame, the caption says so.

## What you should expect

The shipped recognizer is three landmark transformers (width 192, 3 layers, float16 weights) averaged together. Each frame is the hands relative to the nose, wrist-centered hand shape, arm pose, and frame-to-frame motion. Validation and test are PopSign’s official game signer split (8 and 8 signers, at most 20 clips per gloss). Training is the 64-per-gloss game core from the other 31 signers, plus every non-game training clip whose signer is outside those 16 validation and test signers. One of those non-game signers is not in the 31, so the training file has 32 signers. No validation or test signer is in training.

| | |
| --- | --- |
| Gloss vocabulary | **250** English glosses, the full PopSign / Google ASL Signs list |
| Train / val / test clips | 40,139 / 4,623 / 4,540 |
| Test top-1 | **55.2%** |
| Test top-5 | **78.1%** |
| Macro recall | 55.2% |
| Chance top-1 | 0.4% |
| Live gate on test | 78.1% precision when it commits, on 49.9% of clips |

The stored test scores are 0.5515 top-1 and 0.7806 top-5. One member is the best checkpoint from a 48-epoch run of this architecture (51.4% validation on its own). The other two were trained with strong time-warp augmentation. The previous shipped model, the width-128 ensemble on the same 40,139 clips, scored 0.5388 and 0.7874 on these same 4,540 test clips. Validation top-1 on the unchanged 4,623 clips rose from 52.2% to 53.1%. Of 27,547 indexed non-game clips, 25,384 were long enough to train on. Mirror test-time augmentation did not raise validation top-1 (0.5310 without it, 0.5304 with it), so the file does not mirror at commit time. Float16 weights matched the full-precision top-1. One member previews in 42 ms and the three-member commit takes 124 ms in the browser check, so the 8.6 MB file ships without a distilled student.

The file divides logits by a temperature of **1.12** before the softmax. On the validation clips that lowers negative log likelihood from 2.019 to 1.999 and does not change the argmax, so the stored test top-1 stays 55.2%. The live gate was fit again on those tempered scores: about 73% precision, committing on 56% of validation clips. When the gate will not add a word, three large buttons offer the top guesses. Tap the one you signed. The weakest validation glosses are pen, go, mouth, sleep, tree, and wet, usually mixed with a near neighbor such as pencil, eye, or lips. Details: `training/metrics/val-errors.json`.

An earlier model was 50.6% top-1 and 75.1% top-5 on 2,744 test clips drawn from a partial shard download. That list is smaller than these 4,540 clips, so it is not a paired comparison. A 32-gloss model earlier on this branch scored 59.1% top-1. That number is not comparable either: this model chooses among 250 signs. Chance falls from 3.1% to 0.4%. Conversational glosses are in the 250. Per-class recall on this test list: hello 39% (18 clips), thank you 88% (16), please 69% (16), yes 53% (17), no 53% (19), water 67% (18), mom 63% (16), dad 47% (17). **Help is not in this 250-sign list.**

Per-class precision and recall: `training/metrics/asl_en_metrics.json`.

Fingerspelling is a separate static network trained on the MIT dataset `sid220/asl-now-fingerspelling` (A–Z, including J and Z). Filename-hash holdout top-1 is **92.3%** (222 test frames). J recall 91% (11 frames), Z recall 100% (10 frames). **Signer ids are not in that dataset, so this letter number is not signer-independent.** Details: `training/metrics/fingerspelling_metrics.json`.

Live output is suppressed unless the score clears a threshold picked on the validation signers, including a per-class floor when the model file has one. A gloss is added once per motion segment. If that gate does not commit, the top three words stay on screen as large buttons and one tap adds the word you meant. **Send line** commits the caption into the conversation. Speaking aloud, if you turned it on, reads that line to the hearing person. It does not run by default.

Fingerspelling letters A–Z, including J and Z, come from a static landmark network. See the metrics file for the holdout caveat.

## Dataset and license

Training uses PopSign ASL v1.0 game frames plus non-game frames whose signers are outside the game validation and test partitions, CC BY 4.0, via the frame release [`sign/popsign-images`](https://huggingface.co/datasets/sign/popsign-images). Cite Starner et al., PopSign ASL v1.0 (2023). Landmark files are derived and are not a way to identify the signers.

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
PYTHONPATH=training python training/scan_popsign.py
PYTHONPATH=training python training/build_cache.py --train-cap 200 --nongame-cap 100000 --core-only --with-nongame --workers 3
PYTHONPATH=training python training/featurize_rich.py --cache training/cache --index index.jsonl
PYTHONPATH=training python training/train_sequence_model.py --cache training/cache --index index.jsonl --feature-layout rich --augment strong --hidden 192 --layers 3 --ff 480 --epochs 48 --patience 8 --warmup 3 --batch-size 192 --seeds 0,1,2 --label-smoothing 0.05 --experiment core64-rich192 --hold-test --distill-if-slow --resume
PYTHONPATH=training python training/train_sequence_model.py --cache training/cache --report-test --artifact-dir training/artifacts/core64-rich192
```

`--hold-test` picks mirror augmentation, float16, and a student on validation only. `--report-test` scores that selection on the test list once. The 53.9% model used the 273-d features, width 128, 2 layers, and `--augment base` (`experiment core64-nongame`). The 53.7% model used the same caps with `--nongame-cap 0` and `--index index-core.jsonl` (`experiment core64-base`). `--with-nongame` keeps that validation and test list and adds non-game training clips only.

Extraction writes `training/cache/` (gitignored). The trainer compares a three-seed ensemble with a smaller distilled student, with and without mirror test-time augmentation, and it keeps float16 weights when the validation top-1 drop stays within half a point. It replaces `public/models/asl-en-sequence.json` only when the signer-disjoint test top-1 is higher than the file already there and the browser latency check passes. Either way it writes `training/metrics/asl_en_run.json`.

When a run is shipped it also writes:

- `public/models/asl-en-sequence.json` — weights the app loads
- `public/models/demo-clips.json` — held-out replay clips
- `public/models/parity.json` — logits the browser port must match
- `training/metrics/asl_en_metrics.json`

The shipped vocabulary is already the full 250 PopSign glosses. `training/scan_popsign.py` indexes every game and non-game shard on `sign/popsign-images` with range requests, and `training/build_cache.py` downloads frames only for the selected clips. The number above is that full-catalog game test list. The Kaggle landmark release in `docs/KAGGLE_DATASETS.md` is still the native-frame-rate set and still needs a token.

## Layout

- `src/tracking/MediaPipeTracker.ts` — camera and Holistic landmarks
- `src/recognition/` — features, sequence model, live segmenter, fingerspelling
- `src/textAssembly/SignTextAssembler.ts` — phrase buffer
- `src/textAssembly/glossEnglish.ts` — gloss sequence to a short English sentence
- `src/personalize/personalSigns.ts` — signs saved in the browser
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
