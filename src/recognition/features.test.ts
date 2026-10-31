import { describe, expect, it } from 'vitest';
import fixture from './feature-fixture.json';
import fineFixture from './feature-fine-fixture.json';
import richFixture from './feature-rich-fixture.json';
import { resampleSequence, sequenceFeatures, sequenceFeaturesFine, sequenceFeaturesRich, sequenceFeaturesSigner, type FeatureFrame, type HandSide, type Landmark } from './features';

function toLandmarks(points: number[][] | null): Landmark[] | null {
  if (!points) return null;
  return points.map((point) => ({ x: point[0], y: point[1], z: point[2] ?? 0 }));
}

describe('sequence features', () => {
  it('matches the Python feature contract', () => {
    for (const testCase of fixture.cases) {
      const frames: FeatureFrame[] = testCase.frames.map((frame) => ({
        left: toLandmarks(frame.left),
        right: toLandmarks(frame.right),
        pose: toLandmarks(frame.pose),
      }));
      const forced: HandSide | null = testCase.forced === 'Left' || testCase.forced === 'Right' ? testCase.forced : null;
      const actual = sequenceFeatures(frames, forced);
      expect(actual.length).toBe(testCase.features.length);
      for (let t = 0; t < actual.length; t++) {
        for (let d = 0; d < actual[t].length; d++) {
          expect(actual[t][d]).toBeCloseTo(testCase.features[t][d], 5);
        }
      }
    }
  });

  it('matches the Python face-relative feature contract', () => {
    for (const testCase of richFixture.cases) {
      const frames: FeatureFrame[] = testCase.frames.map((frame) => ({
        left: toLandmarks(frame.left),
        right: toLandmarks(frame.right),
        pose: toLandmarks(frame.pose),
      }));
      const forced: HandSide | null = testCase.forced === 'Left' || testCase.forced === 'Right' ? testCase.forced : null;
      const actual = sequenceFeaturesRich(frames, forced);
      expect(actual.length).toBe(testCase.features.length);
      expect(actual[0]?.length).toBe(294);
      for (let t = 0; t < actual.length; t++) {
        for (let d = 0; d < actual[t].length; d++) {
          expect(actual[t][d]).toBeCloseTo(testCase.features[t][d], 5);
        }
      }
    }
  });

  it('matches the Python face-offset feature contract', () => {
    for (const testCase of fineFixture.cases) {
      const frames: FeatureFrame[] = testCase.frames.map((frame) => ({
        left: toLandmarks(frame.left),
        right: toLandmarks(frame.right),
        pose: toLandmarks(frame.pose),
        lips: toLandmarks(frame.lips),
      }));
      const actual = sequenceFeaturesFine(frames, null);
      expect(actual[0]?.length).toBe(389);
      for (let t = 0; t < actual.length; t++) {
        for (let d = 0; d < actual[t].length; d++) {
          expect(actual[t][d]).toBeCloseTo(testCase.features[t][d], 4);
        }
      }
    }
  });

  it('scales signer shape by palm size and mirrors a left hand', () => {
    const pose: Landmark[] = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.4, z: 0 }));
    pose[0] = { x: 0.5, y: 0.2, z: 0 };
    pose[11] = { x: 0.4, y: 0.45, z: 0 };
    pose[12] = { x: 0.6, y: 0.45, z: 0 };
    const hand = (x: number, scale: number): Landmark[] =>
      Array.from({ length: 21 }, (_, index) => ({
        x: x + index * 0.002 * scale,
        y: 0.3 + index * 0.001 * scale,
        z: 0.01,
      }));
    const base: FeatureFrame[] = [0, 1, 2].map((step) => ({
      left: null,
      right: hand(0.7, 1).map((point) => ({ ...point, y: point.y + step * 0.02 })),
      pose,
    }));
    const grown: FeatureFrame[] = base.map((frame) => {
      const wrist = frame.right![0];
      return {
        left: null,
        right: frame.right!.map((point) => ({
          x: wrist.x + (point.x - wrist.x) * 1.8,
          y: wrist.y + (point.y - wrist.y) * 1.8,
          z: wrist.z + (point.z - wrist.z) * 1.8,
        })),
        pose,
      };
    });
    const signer = sequenceFeaturesSigner(base);
    const signerGrown = sequenceFeaturesSigner(grown);
    expect(signer[0]).toHaveLength(389);
    for (let d = 147; d < 210; d += 1) {
      expect(signerGrown[0][d]).toBeCloseTo(signer[0][d], 4);
    }
    const fine = sequenceFeaturesFine(base);
    const fineGrown = sequenceFeaturesFine(grown);
    const shapeMoved = fine.some((row, time) => row.slice(147, 210).some((value, index) => Math.abs(value - fineGrown[time][index + 147]) > 1e-3));
    expect(shapeMoved).toBe(true);
    const left: FeatureFrame[] = base.map((frame) => ({
      left: frame.right!.map((point) => ({ ...point, x: 1 - point.x })),
      right: null,
      pose: frame.pose!.map((point) => ({ ...point, x: 1 - point.x })),
    }));
    const mirrored = sequenceFeaturesSigner(left);
    for (let t = 0; t < signer.length; t += 1) {
      for (let d = 0; d < signer[t].length; d += 1) {
        expect(mirrored[t][d]).toBeCloseTo(signer[t][d], 4);
      }
    }
  });

  it('resamples endpoints without moving them', () => {
    const seq = [
      [0, 1],
      [2, 3],
      [4, 5],
    ];
    const out = resampleSequence(seq, 5);
    expect(out[0][0]).toBeCloseTo(0, 5);
    expect(out[4][0]).toBeCloseTo(4, 5);
    expect(out).toHaveLength(5);
  });
});
