# Audit

## Does this project need a database?

**No.** Not for ASL-to-English recognition now, and a database is not what would make recognition more accurate later.

The running app has no accounts, sessions, or server. These are the only stores:

| What | Where | Survives refresh? |
| --- | --- | --- |
| Optional custom landmark samples and their label list | Browser `localStorage` (`src/storage.ts`) | Yes, on this browser only |
| Optional in-browser TensorFlow.js weights | `localStorage` via TensorFlow.js (`src/signModel.ts`) | Yes, but the live path does not use them |
| Conversation transcript | React state in `src/App.tsx` | No |
| Shipped gloss model | `public/models/asl-en-sequence.json` | Yes, it is a static file |
| Camera frames and landmarks | Memory, then discarded | No |

Nothing in the repo talks to SQL, IndexedDB, Firebase, or Supabase. Import and export of samples is a JSON file the user downloads.

Recognition is webcam → MediaPipe Holistic → a small sequence model → English text, all in the browser. A database cannot see the camera, and the project rule is that camera video stays on the device. Keeping a transcript across reloads can be done later in `localStorage` or IndexedDB. Accounts or cross-device history would be an optional backend, separate from recognition, and would still not store video.

## What the app was actually doing

The live path did not run a learned model. `SignLanguageMatcher` classified a single frame by whether each fingertip was below its knuckle in image coordinates, then returned a hard-coded confidence between 89 and 98. Signs that share an open hand (hello, thank you, please, B) collapsed into one label. The fold test fails when the palm is not upright.

`training/train_sl2t_dataset.py` built 630 landmark files from a circle plus noise, identical across classes, and wrote `training/sl2t_model.json` with `"accuracy": 98.4` and no weights. Those files are not used anymore.

Three MediaPipe graphs (hands, face, pose) ran one after another on every frame even though Holistic was already loaded on the page. That tripled the camera work and called the UI three times per frame. The TensorFlow.js MLP in `src/signModel.ts` was not connected to the camera.

The UI described this as Google DeepMind sign-language-to-text. There is no DeepMind model in the repository.

## What runs now

American Sign Language only, English gloss text out. `src/languages/aslEnglish.ts` is the only language pack. Another sign language would be a new pack plus its own weights, not a change to the camera loop.

1. `@mediapipe/tasks-vision` Holistic landmarker (hands, pose, face), one pass, about 15 frames per second.
2. A 32-step conv + GRU classifier over normalized landmarks (dominant hand, its frame-to-frame delta, and seven pose points).
3. A confidence and margin gate chosen on the validation signers, plus a short motion segment so a gloss is emitted once per sign instead of every frame.
4. A geometric fingerspelling helper for A, B, C, F, I, L, V, W, and Y when the word model abstains and the hand is still. J and Z are not recognized. This helper is not a measured signer-independent letter model.
5. A phrase buffer that joins glosses into a line of English and speaks it after a pause. It does not claim to translate ASL grammar.

## Dataset

Trained on **PopSign ASL v1.0** game clips published as frame sequences in `sign/popsign-images` (CC BY 4.0). Signers are the official train / validation / test partitions from the PopSign paper: 31 / 8 / 8, with **no signer in more than one partition**. The checked overlap on the downloaded shards is empty.

The vocabulary is the glosses that appear in all three partitions of the shards this repo downloads (early alphabetical PopSign signs, 32 glosses in the baseline). It is infant-CDI vocabulary (book, bird, brother, …), not a conversational list such as “hello” or “thank you”. Those glosses are not in this slice. WLASL, MS-ASL, and ASL Citizen were not used: WLASL and MS-ASL do not grant redistribution of the videos, and ASL Citizen forbids distributing the data or modifications. FLEURS-ASL is continuous translation under CC BY-SA and is the wrong task for isolated glosses.

Google’s Isolated Sign Language Recognition release is the same PopSign signs already stored as MediaPipe landmarks (CC BY 4.0) but is about 56 GB and requires a Kaggle login. This pipeline uses the public Hugging Face frame release and runs Holistic locally so the browser and the trainer share one landmarker.

100% accuracy is not a realistic target. On the untouched test signers this baseline reaches **59.1% top-1** and **84.3% top-5** across 32 glosses (chance top-1 is 3.1%). At the validation-chosen gate, test precision is 69.7% and the model commits on 77.6% of clips. Per-class numbers are in `training/metrics/asl_en_metrics.json`.

## Limits

- Isolated gloss in, English word out. Not continuous signing and not ASL grammar.
- One-handed phone signs from PopSign. Two-handed signs and large signing space are weakly represented.
- Training frames are about 5 per second and often only a handful of frames long. Live video is resampled to the same 32-step window, but the motion is denser than the training clips.
- Fingerspelling letters other than the nine geometric shapes are unsupported.
- The model must be loaded from `/models/asl-en-sequence.json`. The first visit also downloads the MediaPipe task file from Google’s storage.
