# SignBridge — Complete Architecture Prompt

Copy everything below into your coding IDE. This is the full, self-contained build spec — it supersedes the two earlier smaller prompts (tracking-only, and the first MVP prompt). Feed it in as-is, or split it phase-by-phase across multiple sessions if your IDE works better with smaller chunks.

---

## Prompt

I'm building **SignBridge**: a free, open-source, production-ready web application that lets a Deaf or non-speaking person communicate in real time with someone who doesn't know sign language. A webcam captures the signer, the app recognizes American Sign Language (ASL) in real time, converts it to on-screen text and spoken audio for the hearing person — and, to make this a genuine two-way conversation tool rather than one-directional, the hearing person's spoken or typed reply is shown as on-screen text for the Deaf/non-speaking person to read.

Build this completely: architecture, data pipeline, application code, and production-readiness (error handling, performance, accessibility, deployment). I want to be able to actually ship and use this, not just get a prototype.

### Non-negotiable constraints
- **100% free and open source.** No paid APIs, no paid model inference, no required paid backend. Every dataset, model, and library used must be free/open-license — cite the license of anything pulled in.
- **Privacy-first, client-side by default.** Camera video and microphone audio must be processed **on-device in the browser** and must never be uploaded or sent to any server. This is both a technical and a trust requirement — users are a vulnerable population and must be able to trust that their video never leaves their device. If any component genuinely cannot run client-side, flag it to me explicitly before implementing rather than silently adding a server call.
- **No required backend for core functionality.** The app must be deployable as a static site (e.g. GitHub Pages) with zero server costs. A backend is only acceptable for optional, clearly-labeled features (e.g. account sync), never for core sign recognition or voice/text conversion.
- **Tech stack:** React + TypeScript for the web app. Python for offline training/data-prep scripts (kept in a separate folder, not part of the runtime app).

---

### System architecture (high level)

```
[Signer's camera] 
   → Tracking layer (MediaPipe Holistic: hands + pose + face landmarks)
   → Sign Recognition layer (client-side ML model: landmarks → gloss/word)
   → Text Assembly layer (buffers signs into words/phrases, basic sentence structure)
   → Output: on-screen live captions  +  Text-to-Speech (Piper) spoken aloud

[Hearing person's mic or keyboard]
   → Speech-to-Text (Whisper, running client-side) OR direct text input
   → Displayed as on-screen text for the Deaf/non-speaking person to read

Both directions share one conversation view so the exchange reads like a chat/subtitle log.
```

---

### Layer 1 — Capture & Tracking
- Webcam access via `getUserMedia`, streamed into a `<video>` element.
- **MediaPipe Tasks Vision** (`@mediapipe/tasks-vision`) for real-time landmark extraction: hands (handshape, orientation, location, movement), pose (arms/shoulders/torso/head), and face (eyebrows, eyes, mouth — these are grammatically load-bearing in ASL, not cosmetic, so don't simplify or drop them).
- Runs per-frame (throttled to a sensible FPS for performance/battery) with a canvas overlay showing detected landmarks, and an on-screen FPS counter.
- Expose a clean callback (`onFrameLandmarks({ hands, pose, face })`) so downstream layers plug in without touching tracking code.
- Graceful handling of: no camera permission, no person in frame, poor lighting/low detection confidence — clear on-screen status messages, never a silent failure.

### Layer 2 — Sign Recognition (build in phases, ship each as a working milestone)

**Phase A — Fingerspelling (static, single-frame):** recognize the ASL alphabet (A–Z) and digits (0–9) from a single frame of hand landmarks.
- Data: ASL Alphabet dataset (Kaggle, free).
- Model: MediaPipe hand landmarks → small MLP/CNN classifier.
- Export to **ONNX** and/or **TensorFlow.js**, run inference client-side.

**Phase B — Common isolated words/phrases:** recognize ~50–100 everyday signs (hello, thank you, help, yes, no, please, eat, more, stop, water, name, sorry, understand, etc.) as landmark *sequences* (not single frames).
- Data: **WLASL** (Word-Level ASL, free, ~2,000 signs available if you want to expand later).
- Check first whether **AI4Bharat's OpenHands** (github.com/AI4Bharat/OpenHands, Apache 2.0) has a usable pretrained WLASL checkpoint for the classes needed — fine-tune that instead of training from scratch if so.
- Model: LSTM or small Transformer over landmark sequences.

**Phase C — Continuous sentence-level signing (stretch goal — do not attempt until A and B are solid and shipped):**
- Data: **How2Sign** (free, video + transcripts).
- This needs sequence-to-sequence modeling and is a genuinely open research problem — scope it as a clearly separated future milestone, not part of the production-ready v1.

### Layer 3 — Text Assembly
- Buffers recognized signs/fingerspelled letters into words and phrases (e.g. debounce repeated frame predictions, detect sign boundaries via pauses/movement resets).
- Basic cleanup: capitalization, simple punctuation inference (e.g. treat eyebrow-raise non-manual marker, if available from Layer 1, as a question-mark signal — optional enhancement, not required for v1).
- Maintains a running transcript for the conversation view.

### Layer 4 — Output (Signer → Hearing person)
- Live on-screen captions of recognized text.
- **Text-to-speech** via **Piper** (prefer a browser/WASM-compatible build so this stays client-side; document the tradeoff if a lightweight self-hosted fallback is used instead).
- Auto-speak-on-phrase-complete, plus a manual "Speak" button and a mute toggle.

### Layer 5 — Reverse channel (Hearing person → Signer)
- Hearing person can **speak** (transcribed via client-side **Whisper**, e.g. `whisper.cpp` compiled to WASM or a Transformers.js Whisper pipeline) or **type directly**.
- Result is displayed as large, clear on-screen text for the Deaf/non-speaking person to read.
- Both directions appear in one shared, scrollable conversation/subtitle log so the exchange is easy to follow for both parties.

---

### Production-readiness requirements
- **Accessibility (this app IS an accessibility tool, so this is not optional):** WCAG-conformant contrast and text sizing for the caption display, full keyboard navigability, screen-reader-friendly markup for all non-camera UI, adjustable caption text size.
- **Performance targets:** state target FPS for tracking + inference on a mid-range laptop/phone, and degrade gracefully (e.g. lower tracking FPS) rather than freezing on lower-end hardware.
- **Offline capability:** structure as a PWA where feasible, so it works without a network connection once models are cached locally.
- **Error handling:** every layer (camera, tracking, model inference, TTS, STT) needs a visible, human-readable failure state — never a silent crash, given the user may depend on this for real communication.
- **Testing:** unit tests for text assembly/buffering logic at minimum; a documented manual test checklist for the camera/ML pipeline (since that's harder to unit test).
- **Deployment:** static build deployable to GitHub Pages (or equivalent free static host), documented in the README.
- **Documentation:** README covering the mission, full architecture, free/open-source stack rationale, local setup, how to retrain/extend models, and how to add a new sign language in the future (keep the model/data layer decoupled enough that this is realistic).
- **Licensing:** pick MIT or Apache 2.0 for the project, and list the license of every dataset/model dependency (ASL Alphabet, WLASL, How2Sign, OpenHands, MediaPipe, Whisper, Piper) in a `THIRD_PARTY_LICENSES.md`.

### Repo structure
```
/app          → React/TypeScript web app (capture, tracking, recognition, text assembly, TTS, STT, UI)
/models       → exported ONNX/TFJS model files + README on how each was produced
/training     → Python: dataset download/prep, landmark extraction, training, export scripts
/docs         → architecture notes, dataset/model licenses, setup + contribution guide
README.md
LICENSE
THIRD_PARTY_LICENSES.md
```

### Build order (ship each as a working, demoable milestone — don't jump ahead)
1. Tracking layer working with live landmark overlay (no recognition yet).
2. Phase A fingerspelling model trained, exported, wired into the app, recognizing letters live.
3. Layer 3 text assembly + Layer 4 TTS output working end to end for fingerspelled input.
4. Phase B common-words model added alongside fingerspelling.
5. Layer 5 reverse channel (Whisper STT + typed input) added, conversation view unifies both directions.
6. Production-readiness pass: accessibility, offline/PWA, error states, tests, deployment, docs.
7. Phase C (continuous sentences) as an explicitly separate future milestone — do not start until 1–6 are solid.

### Ground rules for you (the coding agent)
- Ask me before adding any paid dependency, any required backend for core features, or any component that would send camera/mic data off-device.
- Build and demo incrementally in the order above — don't skip ahead to later phases before earlier ones are working.
- If a "free" tool/dataset turns out to have licensing restrictions that conflict with open-source redistribution, stop and flag it rather than including it.
