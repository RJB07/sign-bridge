import { describe, expect, it } from 'vitest';
import type { Landmark } from './features';
import { LiveInterpreter, type GlossClassifier, type LiveSnapshot, type RawFrame } from './liveInterpreter';

function handAt(x: number): Landmark[] {
  return Array.from({ length: 21 }, (_, index) => ({ x: x + index * 0.001, y: 0.2, z: 0 }));
}

function frame(timeMs: number, x: number): RawFrame {
  return { timeMs, left: null, right: handAt(x), pose: null };
}

function classifier(confidence: number): GlossClassifier {
  return {
    labels: ['apple'],
    emitThreshold: 0.5,
    marginThreshold: 0.1,
    predict: () => ({
      label: 'apple',
      confidence,
      margin: 0.4,
      probabilities: [confidence],
    }),
  };
}

describe('live interpreter', () => {
  it('commits a gloss once while the hand is moving', () => {
    const interpreter = new LiveInterpreter(classifier(0.92), {
      startSpeed: 0.01,
      stopSpeed: 0.004,
      minFrames: 3,
      earlyIntervalMs: 0,
      earlyBoost: 0,
      stopFrames: 8,
    });
    let committed = null as string | null;
    for (let i = 0; i < 6; i++) {
      const snap = interpreter.push(frame(i * 50, i * 0.05));
      if (snap.committed) committed = snap.committed.text;
    }
    expect(committed).toBe('apple');
  });

  it('does not commit a low-confidence gloss', () => {
    const interpreter = new LiveInterpreter(classifier(0.2), {
      startSpeed: 0.01,
      minFrames: 3,
      earlyIntervalMs: 0,
      earlyBoost: 0,
      stopFrames: 3,
    });
    const texts: string[] = [];
    for (let i = 0; i < 8; i++) {
      const snap = interpreter.push(frame(i * 40, i < 4 ? i * 0.05 : 0.2));
      if (snap.committed) texts.push(snap.committed.text);
    }
    expect(texts).not.toContain('apple');
  });

  it('offers the top three words when the gate will not commit', () => {
    const interpreter = new LiveInterpreter(
      {
        labels: ['pen', 'pencil', 'book'],
        emitThreshold: 0.8,
        marginThreshold: 0.2,
        predict: () => ({ label: 'pen', confidence: 0.4, margin: 0.05, probabilities: [0.4, 0.35, 0.25] }),
      },
      { startSpeed: 0.01, stopSpeed: 0.004, minFrames: 3, earlyIntervalMs: 10_000, earlyBoost: 0, stopFrames: 2 },
    );
    let choices: string[] = [];
    for (let i = 0; i < 8; i++) {
      const snap = interpreter.push(frame(i * 40, i < 4 ? i * 0.05 : 0.2));
      if (snap.choices.length > 0) choices = snap.choices.map((choice) => choice.label);
    }
    expect(choices).toEqual(['pen', 'pencil', 'book']);
    interpreter.dismissChoices();
    expect(interpreter.push(frame(400, 0.2)).choices).toEqual([]);
  });

  it('previews with the fast classifier and commits with the full one', () => {
    let preview = 0;
    const interpreter = new LiveInterpreter(
      {
        labels: ['apple', 'book'],
        emitThreshold: 0.5,
        marginThreshold: 0.1,
        predictPreview: () => {
          preview += 1;
          return { label: 'apple', confidence: 0.3, margin: 0.1, probabilities: [0.3, 0.2] };
        },
        predict: () => ({ label: 'book', confidence: 0.95, margin: 0.8, probabilities: [0.05, 0.95] }),
      },
      { startSpeed: 0.01, stopSpeed: 0.004, minFrames: 3, earlyIntervalMs: 10_000, earlyBoost: 0, stopFrames: 2 },
    );
    let committed = null as string | null;
    for (let i = 0; i < 5; i++) {
      const snap = interpreter.push(frame(i * 50, i * 0.05));
      if (snap.candidate) expect(snap.candidate).toBe('apple');
      if (snap.committed) committed = snap.committed.text;
    }
    for (let i = 5; i < 9; i++) {
      const snap = interpreter.push(frame(i * 50, 0.2));
      if (snap.committed) committed = snap.committed.text;
    }
    expect(preview).toBeGreaterThan(0);
    expect(committed).toBe('book');
  });

  it('commits when the hand leaves and ignores a non-numeric wrist', () => {
    const interpreter = new LiveInterpreter(classifier(0.95), {
      startSpeed: 0.01,
      stopSpeed: 0.004,
      minFrames: 4,
      earlyIntervalMs: 0,
      earlyBoost: 0.5,
      stopFrames: 30,
    });
    let text: string | null = null;
    for (let i = 0; i < 6; i++) {
      const snap = interpreter.push(frame(i * 40, i * 0.05));
      if (snap.committed) text = snap.committed.text;
    }
    expect(text).toBeNull();
    for (let i = 6; i < 12; i++) {
      const snap = interpreter.push({ timeMs: i * 40, left: null, right: null, pose: null });
      if (snap.committed) text = snap.committed.text;
    }
    expect(text).toBe('apple');

    const bad = frame(0, 0.2);
    bad.right![0] = { x: Number.NaN, y: 0.2, z: 0 };
    expect(interpreter.push(bad).phase).toBe('idle');
  });

  it('closes a segment that never becomes still', () => {
    const interpreter = new LiveInterpreter(classifier(0.2), {
      startSpeed: 0.01,
      stopSpeed: 0.004,
      minFrames: 3,
      earlyIntervalMs: 100000,
      stopFrames: 100,
      maxSegmentFrames: 8,
    });
    let phase: LiveSnapshot['phase'] = 'idle';
    for (let i = 0; i < 12; i++) phase = interpreter.push(frame(i * 40, i * 0.05)).phase;
    expect(phase).toBe('idle');
  });

  it('does not fingerspell in the moment after a gloss', () => {
    const texts: string[] = [];
    const interpreter = new LiveInterpreter(classifier(0.95), {
      startSpeed: 0.01,
      stopSpeed: 0.004,
      minFrames: 3,
      earlyIntervalMs: 0,
      earlyBoost: 0,
      stopFrames: 2,
      letterHoldFrames: 2,
      letterCooldownMs: 500,
      letterPredict: () => ({ label: 'A', confidence: 0.99 }),
    });
    for (let i = 0; i < 6; i++) {
      const snap = interpreter.push(frame(i * 40, i * 0.05));
      if (snap.committed) texts.push(snap.committed.text);
    }
    for (let i = 6; i < 10; i++) {
      const snap = interpreter.push(frame(i * 40, 0.25));
      if (snap.committed) texts.push(snap.committed.text);
    }
    expect(texts).toContain('apple');
    expect(texts).not.toContain('A');
  });
});
