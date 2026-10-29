import { describe, expect, it } from 'vitest';
import type { Landmark } from './features';
import { LiveInterpreter, type GlossClassifier, type RawFrame } from './liveInterpreter';

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
});
