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
2. A 32-step landmark transformer over nose-relative hands, wrist-centered shape, seven pose points, frame-to-frame motion, hand-to-face offsets (lips, eyes, cheek proxies), and fingertip curl, plus a small hand-shape embedding. The shipped file is four members, width 192, 3 layers, float16. Two members scale the hand by shoulder width and two scale it by palm size. Logits are averaged with weights 1, 1, 1.5, and 1. The matrix multiply and the attention dots compute several outputs from each input load, which is what brought the four-member commit from 174 ms to about 100 ms. Mirror averaging of those four members takes 194 ms. Five members take 122 ms and six take 147 ms. The best five-member mix on validation is within a tenth of a point of the shipped four, so that mix was not scored on the test list. A width-256, 3-layer, feed-forward-640 member takes 41 ms. The shoulder-scaled one reached 51.3% validation top-1 and the palm-scaled one reached 49.8%. Blending them with the current members, inside 130 ms, stayed at 57.4% validation top-1. The test list was not opened for that run. The on-screen guess uses the first member. The committed gloss uses the full ensemble and, when the file says so, a mirrored second view. This file does not. There is no WebGPU or extra runtime to cache.
3. A confidence and margin gate, including a per-class floor, chosen on the validation signers after temperature 1.057. A motion segment emits a gloss once per sign. A brief slowdown followed by motion again splits the next sign, so a short sentence does not need a full pause. If the gate does not emit, the top three words stay on screen as large tap targets. A segment that never becomes still is closed after 75 frames. Low light (no face and no hands) and fingertips outside the frame are coaching warnings, and a non-numeric wrist is treated as no hand. Letters wait a short cooldown after a gloss.
4. A static letter network for A–Z, including J and Z, when the hand is still. The geometric letter helper remains only if that file fails to load.
5. `src/textAssembly/glossEnglish.ts` turns the committed glosses into a short English sentence and leaves the glosses visible. It is word order and a few fixed phrases, not ASL grammar. Speaking that sentence aloud is off until the hearing-person control is turned on.
6. Optional “Your signs” samples in `localStorage` can replace the shared guess when the motion is close to a saved one.

## Dataset

Trained on **PopSign ASL v1.0** in `sign/popsign-images` (CC BY 4.0). Official game signer split: 31 / 8 / 8, no shared signer. Validation and test stay that game list, at most 20 clips per gloss: 4,623 / 4,540, the same clips as the previous shipped model. Training is the 64-per-gloss game core (14,755 clips) plus 25,384 non-game clips (27,547 indexed, 2,163 shorter than two frames). Training has 32 signers. None of the 16 game validation or test signers are among them. Vocabulary is the full 250 PopSign / Google ASL Signs glosses.

100% accuracy is not a realistic target. On the untouched test signers the shipped ensemble is **56.9% top-1** and **79.7% top-5** (stored 0.5694 / 0.7974, chance 0.4%). The previous weights on these same 4,540 clips were 56.8% top-1 and 79.7% top-5. Validation top-1 is 57.4% and validation top-5 is 81.8%. The four-member blend was already ahead on validation, and it shipped after the blocked matmul brought the commit from 174 ms to 100 ms. The test list for this blend was scored once. At the validation-chosen gate, test precision is 78.4% and the model commits on 53.3% of clips. Int8 quantization did not make the matmul shorter. Pruning the feed-forward width to 240 dropped one member from 52.8% to 40.5% validation top-1. An earlier partial-shard draw of the same split was 50.6% top-1 and 75.1% top-5 (2,744 test clips). An earlier 32-gloss model on this branch was 59.1% top-1; that was an easier decision and is not comparable.

The shipped file divides logits by 1.057 before the softmax. That temperature was fit on this blend’s validation logits and does not change the argmax. If the gate does not commit, the screen keeps the top three words as large tap targets. The extra features are the distances from the wrist and fingertips to the lips, eyes, and cheek proxies, plus fingertip curl, because validation errors clustered on near pairs such as pen/pencil, cheek/eye, and mouth/lips.

A distilled width-160, 2-layer copy of this ensemble was measured so mirror averaging could fit under 130 ms. Its validation top-1 was 53.7% and top-5 was 80.9%. Mirror averaging was lower, at 53.5%. The test list was not opened. The shipped weights stay the 56.7% file. Weak glosses in that run (finger, pen, mouth, go, read, sleep, and others) used a wider speed warp and a shorter crop.

A later 192×3 run scaled the wrist-centered hand by palm size, kept the left-to-right mirror, and added 3D camera, body-proportion, and frame-rate augmentation. On its own, validation top-1 was 53.8% without mirror averaging and 54.1% with it (top-5 80.7%). That is below 55.5%, so that ensemble was not opened on the test list by itself. Two of its members are in the shipped four-member blend. Mirror averaging of the signer ensemble commits in 259 ms. The copy without it commits in 131 ms.

Hello, thank you, please, yes, and no are in the 250. Help is not. The quick-phrase row also covers water, potty, and sick, which are in the 250, and sorry, name, doctor, and bathroom, which are not. Potty is the gloss; bathroom is not. A missing word can be tapped onto the caption as English, or fingerspelled while the hand is still. Google ASL Signs (CC BY 4.0) is the same 250 and does not add help, sorry, name, doctor, or bathroom. Google ASL Fingerspelling, treated as CC BY 4.0 after the data page is confirmed, can spell those words with signer ids. The MIT letter model already spells them, and that split is not signer-independent. ASL Citizen and WLASL were not used.

Fingerspelling A–Z, including J and Z, is trained on `sid220/asl-now-fingerspelling` (MIT). Filename-hash test top-1 is 92.3%. That split is not signer-independent because the files have no participant id.

Google ASL Signs (`asl-signs`, CC BY 4.0) is the same 250 glosses as native-rate landmarks. `training/train_asl_signs.py` downloads it when `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` and `KAGGLE_KEY`) is set, builds a signer-disjoint cache, and retrains. It does not download without a token.

## Limits

- Isolated glosses in. A brief slowdown can separate two signs, but the English sentence is still a small rule list, not full ASL grammar.
- PopSign is mostly one-handed phone signing.
- Training frames are about 5 per second. Live video is resampled to the same 32-step window.
- The letter-model holdout is not signer-independent.
- The first online visit fills the service-worker cache. After that the camera tracker and the recognizer are local.
