import { describe, expect, it } from 'vitest';
import fixture from './feature-fixture.json';
import fineFixture from './feature-fine-fixture.json';
import richFixture from './feature-rich-fixture.json';
import { resampleSequence, sequenceFeatures, sequenceFeaturesFine, sequenceFeaturesRich, type FeatureFrame, type HandSide, type Landmark } from './features';

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
