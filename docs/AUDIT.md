# Audit

## Does this project need a database?

**No.** Not for ASL-to-English recognition now, and a database is not what would make recognition more accurate later.

The running app has no accounts, sessions, or server. These are the only stores:

| What | Where | Survives refresh? |
| --- | --- | --- |
| Signs the user labels (“Your signs”) | Browser `localStorage` (`src/personalize/personalSigns.ts`), a normalized landmark vector, not video | Yes, on this browser only |
| Theme, caption size, speech on/off | Browser `localStorage` | Yes, on this browser only |
| Conversation transcript | React state in `src/App.tsx` | No |
| Shipped gloss model | `public/models/asl-en-sequence.json` | Yes, it is a static file |
| Camera frames | Memory for the current frame, then discarded | No |

Nothing in the repo talks to SQL, IndexedDB, Firebase, or Supabase. The app does not upload camera frames. Speech to text, if the hearing person turns it on, uses the browser’s own speech service for the microphone only. That control is off until they press it, and the screen says the audio does not go to SignBridge.

Recognition is webcam → MediaPipe Holistic → a small sequence model → English text, all in the browser. A database cannot see the camera, and the project rule is that camera video stays on the device. Keeping a transcript across reloads can be done later in `localStorage` or IndexedDB. Accounts or cross-device history would be an optional backend, separate from recognition, and would still not store video.

## What the app was actually doing

The live path did not run a learned model. `SignLanguageMatcher` classified a single frame by whether each fingertip was below its knuckle in image coordinates, then returned a hard-coded confidence between 89 and 98. Signs that share an open hand (hello, thank you, please, B) collapsed into one label. The fold test fails when the palm is not upright.

`training/train_sl2t_dataset.py` built 630 landmark files from a circle plus noise, identical across classes, and wrote `training/sl2t_model.json` with `"accuracy": 98.4` and no weights. Those files are not used anymore.

Three MediaPipe graphs (hands, face, pose) ran one after another on every frame even though Holistic was already loaded on the page. That tripled the camera work and called the UI three times per frame. The TensorFlow.js MLP in `src/signModel.ts` was not connected to the camera.

The UI described this as Google DeepMind sign-language-to-text. There is no DeepMind model in the repository.

## What runs now

American Sign Language only, English text out. `src/languages/aslEnglish.ts` is the only language pack.

1. `@mediapipe/tasks-vision` Holistic landmarker in video mode, one pass, about 15 frames per second. The WASM and the task file are served from this origin (`public/vendor/mediapipe`, `public/models/holistic_landmarker.task`) so a repeat visit works offline.
2. A 32-step landmark transformer over nose-relative hands, wrist-centered shape, seven pose points, and frame-to-frame motion. The shipped file is three members, width 192, 3 layers, float16. The on-screen guess uses one member. The committed gloss uses the full ensemble and, when the file says so, a mirrored second view. This file does not.
3. A confidence and margin gate, including a per-class floor, chosen on the validation signers. A motion segment emits a gloss once per sign. A segment that never becomes still is closed after 75 frames. A non-numeric wrist is treated as no hand. Letters wait a short cooldown after a gloss.
4. A static letter network for A–Z, including J and Z, when the hand is still. The geometric letter helper remains only if that file fails to load.
5. `src/textAssembly/glossEnglish.ts` turns the committed glosses into a short English sentence and leaves the glosses visible. It is word order and a few fixed phrases, not ASL grammar. Speaking that sentence aloud is off until the hearing-person control is turned on.
6. Optional “Your signs” samples in `localStorage` can replace the shared guess when the motion is close to a saved one.

## Dataset

Trained on **PopSign ASL v1.0** in `sign/popsign-images` (CC BY 4.0). Official game signer split: 31 / 8 / 8, no shared signer. Validation and test stay that game list, at most 20 clips per gloss: 4,623 / 4,540, the same clips as the previous shipped model. Training is the 64-per-gloss game core (14,755 clips) plus 25,384 non-game clips (27,547 indexed, 2,163 shorter than two frames). Training has 32 signers. None of the 16 game validation or test signers are among them. Vocabulary is the full 250 PopSign / Google ASL Signs glosses.

100% accuracy is not a realistic target. On the untouched test signers the shipped ensemble is **55.2% top-1** and **78.1% top-5** (stored 0.5515 / 0.7806, chance 0.4%). The previous weights on these same 4,540 clips were 53.9% top-1 and 78.7% top-5. Validation top-1 rose from 52.2% to 53.1%. The ship rule is test top-1, so these weights replaced that file. An earlier partial-shard draw of the same split was 50.6% top-1 and 75.1% top-5 (2,744 test clips). At the validation-chosen gate, test precision is 78.1% and the model commits on 49.9% of clips. An earlier 32-gloss model on this branch was 59.1% top-1; that was an easier decision and is not comparable.

Hello, thank you, please, yes, and no are in the 250. Help is not. ASL Citizen and WLASL were not used.

Fingerspelling A–Z, including J and Z, is trained on `sid220/asl-now-fingerspelling` (MIT). Filename-hash test top-1 is 92.3%. That split is not signer-independent because the files have no participant id.

Google ASL Signs (`asl-signs`, CC BY 4.0) is the same 250 glosses as native-rate landmarks. `training/train_asl_signs.py` downloads it when `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` and `KAGGLE_KEY`) is set, builds a signer-disjoint cache, and retrains. It does not download without a token.

## Limits

- Isolated glosses in. The English sentence is a small rule list, not continuous signing and not full ASL grammar.
- PopSign is mostly one-handed phone signing.
- Training frames are about 5 per second. Live video is resampled to the same 32-step window.
- The letter-model holdout is not signer-independent.
- The first online visit fills the service-worker cache. After that the camera tracker and the recognizer are local.
