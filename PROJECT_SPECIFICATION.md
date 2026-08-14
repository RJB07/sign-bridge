# 🌉 SignBridge — Complete System Architecture & Specification

SignBridge is an AI-powered assistive sign language translation engine. It captures real-time visual tracking data—hand movement trajectories, labeled finger landmarks (`T1-P5`), and facial mesh parameters—and uses an **Open-Domain Sign Language Book Library Ground Truth** paired with **Large Language Models (LLMs)** to interpret sign language visual data into natural text in real time, which is then immediately synthesized into **spoken voice audio**.

---

## 🎯 1. Mission & System Philosophy

### Primary Purpose
To empower deaf and non-vocal individuals by transforming natural sign language gestures into real-time spoken communication. 

### Why Visual Landmark & Finger Tracking Matter
Sign language is a complex, high-dimensional visual language composed of:
1. **Hand Shapes & Finger Positions**: Pinpointed via 21 3D landmarks and finger tip labels (`T1` Thumb, `I2` Index, `M3` Middle, `R4` Ring, `P5` Pinky).
2. **Hand Movement & Spatial Trajectory**: Tracked across consecutive video frames to capture motion direction and velocity.
3. **Facial Expressions & Posture**: 468 facial mesh landmarks capturing eye movements, lip posture, and eyebrow expressions that modify grammatical meaning.

This multimodal visual data forms the structured feature input fed into the translation engine.

---

## 🏗️ 2. End-to-End Architectural Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. VISUAL INPUT CAPTURE                         │
│   Webcam Video Stream ➔ MediaPipe 21 Hand Landmarks & 468 Face Mesh    │
│   Finger Tip Labeling (T1, I2, M3, R4, P5) & Spatial Coordinate Vectors│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                2. OPEN-DOMAIN SIGN LANGUAGE BOOK REFERENCE             │
│   Ground-Truth Sign Vocabulary & Grammar Rules from Open-Domain Books  │
│   Reference Mapping: Handshapes + Trajectories + Facial Markers ➔ Words │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   3. REAL-TIME MULTIMODAL LLM REASONING                │
│   Processes Temporal Landmark Trajectories & Expression Vectors        │
│   Generates Grammatically Accurate Text Sentences in Real Time         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   4. INSTANT TEXT-TO-SPEECH SYNTHESIS                  │
│   Translates Output Text into Spoken Voice Audio (Web Speech Engine)   │
│   Audible Output for Seamless Real-Time Hearing & Non-Vocal Dialog     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 3. Technical Core Requirements

### 3.1. Visual Tracking Layer (`App.tsx`)
- **Hand Skeleton**: 21 landmarks per hand rendered with Blue (`#3b82f6`) connectors and Yellow (`#facc15`) joints.
- **Finger Anatomy Pins**:
  - `T1`: Thumb Tip (Landmark 4)
  - `I2`: Index Tip (Landmark 8)
  - `M3`: Middle Tip (Landmark 12)
  - `R4`: Ring Tip (Landmark 16)
  - `P5`: Pinky Tip (Landmark 20)
- **Facial Mesh**: 468 facial points capturing expression context.

### 3.2. Sign Book Knowledge Base & Ground Truth
- Open-domain sign language dictionaries and reference book datasets serve as the authoritative standard for mapping handshapes, movements, and expressions to word definitions and ASL grammar structures.

### 3.3. LLM Translation & Reasoning Engine
- Landmark spatial coordinates ($X, Y, Z$ trajectory per frame) are compiled into structured JSON feature streams.
- Large Language Models (LLMs) interpret landmark sequences against the sign language book library, outputting fluent, grammatically corrected sentences in real time.

### 3.4. Real-Time Text-to-Speech Engine
- Converted text sentences are instantly synthesized into spoken speech using browser-native `window.speechSynthesis` so anyone nearby can hear the translation in real time.

---

## 🎨 4. Oberon UI/UX Design Token Standards

- **Background**: Light Warm Stone (`#efece6`) with 60px technical grid lines.
- **Eyebrow Badges**: Vermillion Orange `01` block + `INTRO` light gray pill (`#e2ded6`).
- **Typography**: `Geist Mono` monospaced font for labels, finger pins (`T1-P5`), and telemetry + `Inter` for display text.
- **Layout**: Clean two-column workspace (Camera Viewport Grid + Translation & Speech Dashboard Grid).

---

## 🚀 5. Verification Command

```bash
npm run build
```
