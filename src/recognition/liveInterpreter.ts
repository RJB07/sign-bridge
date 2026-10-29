import { classifyFingerspelling } from './fingerspelling';
import { sequenceFeatures, type FeatureFrame, type HandSide, type Landmark } from './features';
import { glossToEnglish, type Prediction } from './sequenceModel';

export type GlossClassifier = {
  labels: string[];
  emitThreshold: number;
  marginThreshold: number;
  classThresholds?: number[];
  predict: (sequence: number[][]) => Prediction;
  /** Cheaper score used for the on-screen guess. Commit still uses predict. */
  predictPreview?: (sequence: number[][]) => Prediction;
};

export type RawFrame = FeatureFrame & { timeMs: number };

export type Commit = {
  text: string;
  kind: 'gloss' | 'letter';
  confidence: number;
};

export type LiveSnapshot = {
  candidate: string | null;
  confidence: number;
  phase: 'idle' | 'signing' | 'fingerspelling';
  committed: Commit | null;
  top: Array<{ label: string; probability: number }>;
};

export type InterpreterOptions = {
  forcedHand?: HandSide | null;
  startSpeed?: number;
  stopSpeed?: number;
  stopFrames?: number;
  minFrames?: number;
  earlyIntervalMs?: number;
  earlyBoost?: number;
  letterHoldFrames?: number;
  letterCooldownMs?: number;
  maxSegmentFrames?: number;
  previewIntervalMs?: number;
  letterPredict?: (landmarks: Landmark[]) => { label: string; confidence: number } | null;
};

const DEFAULTS = {
  startSpeed: 0.012,
  stopSpeed: 0.007,
  stopFrames: 10,
  minFrames: 6,
  earlyIntervalMs: 220,
  earlyBoost: 0.12,
  letterHoldFrames: 8,
  letterCooldownMs: 450,
  maxSegmentFrames: 75,
  previewIntervalMs: 120,
};

function wrist(frame: RawFrame, side: 'left' | 'right'): Landmark | null {
  const hand = frame[side];
  const point = hand && hand.length >= 21 ? hand[0] : null;
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return null;
  return point;
}

function speedBetween(prev: RawFrame, cur: RawFrame): number {
  let best = 0;
  for (const side of ['left', 'right'] as const) {
    const a = wrist(prev, side);
    const b = wrist(cur, side);
    if (!a || !b) continue;
    best = Math.max(best, Math.hypot(a.x - b.x, a.y - b.y));
  }
  return best;
}

function handPresent(frame: RawFrame): boolean {
  return Boolean(wrist(frame, 'left') || wrist(frame, 'right'));
}

/**
 * Turns a landmark stream into English glosses (trained model) or letters
 * (geometric fingerspelling) without committing every frame.
 */
export class LiveInterpreter {
  private readonly options: Required<Omit<InterpreterOptions, 'forcedHand' | 'letterPredict'>> & {
    forcedHand: HandSide | null;
    letterPredict?: InterpreterOptions['letterPredict'];
  };
  private frames: RawFrame[] = [];
  private segmentStart: number | null = null;
  private lowMotion = 0;
  private lastClassifyAt = -Infinity;
  private committedThisSegment = false;
  private pendingLetter: string | null = null;
  private letterHold = 0;
  private lastLetter = '';
  private suppressLettersUntil = 0;
  private waitForStill = false;
  private lastPreviewAt = -Infinity;
  private lastPreview: { label: string; confidence: number; top: LiveSnapshot['top'] } | null = null;

  constructor(
    private readonly classifier: GlossClassifier,
    options: InterpreterOptions = {},
  ) {
    this.options = { ...DEFAULTS, forcedHand: null, ...options };
  }

  setForcedHand(hand: HandSide | null): void {
    this.options.forcedHand = hand;
  }

  reset(): void {
    this.frames = [];
    this.segmentStart = null;
    this.lowMotion = 0;
    this.committedThisSegment = false;
    this.pendingLetter = null;
    this.letterHold = 0;
    this.lastLetter = '';
    this.suppressLettersUntil = 0;
    this.waitForStill = false;
    this.lastPreviewAt = -Infinity;
    this.lastPreview = null;
  }

  push(frame: RawFrame): LiveSnapshot {
    const previous = this.frames[this.frames.length - 1];
    this.frames.push(frame);
    const elapsed = previous ? frame.timeMs - previous.timeMs : 1;
    const speed = previous && elapsed > 0 ? speedBetween(previous, frame) : 0;
    const present = handPresent(frame);
    let committed: Commit | null = null;

    if (this.waitForStill && (!present || speed < this.options.startSpeed)) this.waitForStill = false;
    if (present && speed >= this.options.startSpeed && !this.waitForStill) {
      if (this.segmentStart === null) {
        this.segmentStart = Math.max(0, this.frames.length - 2);
        this.committedThisSegment = false;
      }
      this.lowMotion = 0;
      this.pendingLetter = null;
      this.letterHold = 0;
    }

    if (this.segmentStart !== null) {
      const moving = present && speed >= this.options.stopSpeed;
      this.lowMotion = moving ? 0 : this.lowMotion + 1;
      const segmentLength = this.frames.length - this.segmentStart;
      const handLeft = !present && this.lowMotion >= 4;
      const capped = segmentLength >= this.options.maxSegmentFrames;
      const ended = this.lowMotion >= this.options.stopFrames || handLeft || capped;
      if (ended) {
        if (capped && present && speed >= this.options.stopSpeed) this.waitForStill = true;
        if (!this.committedThisSegment && segmentLength >= this.options.minFrames) {
          committed = this.commitGloss(this.segmentStart, this.frames.length - this.lowMotion, this.classifier.emitThreshold);
        }
        this.segmentStart = null;
        this.lowMotion = 0;
        this.committedThisSegment = false;
        this.lastPreview = null;
      } else if (
        !this.committedThisSegment &&
        segmentLength >= this.options.minFrames &&
        frame.timeMs - this.lastClassifyAt >= this.options.earlyIntervalMs
      ) {
        committed = this.commitGloss(
          this.segmentStart,
          this.frames.length,
          this.classifier.emitThreshold + this.options.earlyBoost,
        );
        this.lastClassifyAt = frame.timeMs;
      }
    }

    let phase: LiveSnapshot['phase'] = this.segmentStart !== null ? 'signing' : 'idle';
    let candidate: string | null = null;
    let confidence = 0;
    let top: LiveSnapshot['top'] = [];

    if (this.segmentStart !== null && this.frames.length - this.segmentStart >= this.options.minFrames) {
      const due = frame.timeMs - this.lastPreviewAt >= this.options.previewIntervalMs;
      if (due || !this.lastPreview) {
        this.lastPreview = this.rank(this.segmentStart, this.frames.length);
        this.lastPreviewAt = frame.timeMs;
      }
      if (this.lastPreview) {
        candidate = glossToEnglish(this.lastPreview.label);
        confidence = this.lastPreview.confidence;
        top = this.lastPreview.top;
      }
    }

    if (!committed && this.segmentStart === null && present && speed < this.options.stopSpeed && frame.timeMs >= this.suppressLettersUntil) {
      const letter = this.currentLetter(frame);
      if (letter && letter.label === this.pendingLetter) this.letterHold += 1;
      else {
        this.pendingLetter = letter?.label ?? null;
        this.letterHold = letter ? 1 : 0;
      }
      if (letter) {
        phase = 'fingerspelling';
        candidate = letter.label;
        confidence = letter.confidence;
        const ready = this.letterHold >= this.options.letterHoldFrames;
        const repeat = letter.label === this.lastLetter && this.letterHold >= this.options.letterHoldFrames * 3;
        const first = letter.label !== this.lastLetter && ready;
        if (first || repeat) {
          committed = { text: letter.label, kind: 'letter', confidence: letter.confidence };
          this.lastLetter = letter.label;
          if (repeat) this.letterHold = 0;
        }
      }
    }

    if (this.frames.length > 180) {
      const drop = this.frames.length - 180;
      this.frames.splice(0, drop);
      if (this.segmentStart !== null) this.segmentStart = Math.max(0, this.segmentStart - drop);
    }

    return { candidate, confidence, phase, committed, top };
  }

  /** Classify a prepared feature sequence. Used by the held-out replay control. */
  classifyPrepared(features: number[][]): { label: string; confidence: number; margin: number } {
    const prediction = this.classifier.predict(features);
    return {
      label: glossToEnglish(prediction.label),
      confidence: prediction.confidence,
      margin: prediction.margin,
    };
  }

  private currentLetter(frame: RawFrame): { label: string; confidence: number } | null {
    const side = this.options.forcedHand === 'Left' ? 'left' : this.options.forcedHand === 'Right' ? 'right' : null;
    const hands = side ? [frame[side]] : [frame.right, frame.left];
    for (const hand of hands) {
      if (!hand) continue;
      if (this.options.letterPredict) {
        const predicted = this.options.letterPredict(hand);
        if (!predicted) continue;
        const minimum = predicted.label === 'J' || predicted.label === 'Z' ? 0.75 : 0.55;
        if (predicted.confidence >= minimum) return predicted;
        continue;
      }
      const letter = classifyFingerspelling(hand);
      if (letter) return { label: letter, confidence: 0.7 };
    }
    return null;
  }

  private rank(start: number, end: number): { label: string; confidence: number; top: LiveSnapshot['top'] } | null {
    const slice = this.frames.slice(Math.max(0, start), Math.max(start, end));
    const features = sequenceFeatures(slice, this.options.forcedHand);
    if (features.length < 2) return null;
    const prediction = (this.classifier.predictPreview ?? this.classifier.predict)(features);
    const order = prediction.probabilities
      .map((probability, index) => ({ label: glossToEnglish(this.classifier.labels[index] ?? ''), probability }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 5);
    return { label: prediction.label, confidence: prediction.confidence, top: order };
  }

  private commitGloss(start: number, end: number, threshold: number): Commit | null {
    const slice = this.frames.slice(Math.max(0, start), Math.max(start, end));
    const features = sequenceFeatures(slice, this.options.forcedHand);
    if (features.length < 2) return null;
    const prediction = this.classifier.predict(features);
    const classIndex = this.classifier.labels.indexOf(prediction.label);
    const classMin = this.classifier.classThresholds?.[classIndex] ?? this.classifier.emitThreshold;
    if (prediction.confidence < Math.max(threshold, classMin) || prediction.margin < this.classifier.marginThreshold) return null;
    this.committedThisSegment = true;
    const endFrame = this.frames[Math.min(this.frames.length - 1, Math.max(0, end - 1))];
    this.suppressLettersUntil = (endFrame?.timeMs ?? 0) + this.options.letterCooldownMs;
    return { text: glossToEnglish(prediction.label), kind: 'gloss', confidence: prediction.confidence };
  }
}
