import type { Landmark } from './features';
import { decodeF32 } from './sequenceModel';

type LetterFile = {
  labels: string[];
  weights: Record<string, string>;
  weightShapes: Record<string, number[]>;
};

export type LetterModel = {
  labels: string[];
  layers: Array<{ weight: Float32Array; bias: Float32Array; out: number; inn: number }>;
};

export function letterModelFromJson(payload: LetterFile): LetterModel {
  const names = Object.keys(payload.weights).filter((name) => name.endsWith('.weight')).sort();
  const layers = names.map((name) => {
    const biasName = name.replace('.weight', '.bias');
    const shape = payload.weightShapes[name];
    return {
      weight: decodeF32(payload.weights[name]),
      bias: decodeF32(payload.weights[biasName]),
      out: shape[0],
      inn: shape[1],
    };
  });
  return { labels: payload.labels, layers };
}

const LETTER_CURLS: Array<[number, number, number]> = [[1, 2, 4], [5, 6, 8], [9, 10, 12], [13, 14, 16], [17, 18, 20]];

function letterCurl(a: Landmark, b: Landmark, c: Landmark): number {
  const toward = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
  const away = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const left = Math.hypot(toward.x, toward.y, toward.z);
  const right = Math.hypot(away.x, away.y, away.z);
  if (left < 1e-6 || right < 1e-6) return 0;
  return (toward.x * away.x + toward.y * away.y + toward.z * away.z) / (left * right);
}

/** Wrist-centered, palm-scaled hand plus five fingertip curls. 68 floats. */
export function handFeatures(landmarks: Landmark[]): Float32Array | null {
  if (landmarks.length < 21) return null;
  const out = new Float32Array(68);
  const ox = landmarks[0].x;
  const oy = landmarks[0].y;
  const oz = landmarks[0].z;
  const scale = Math.hypot(landmarks[9].x - ox, landmarks[9].y - oy) || 1;
  const placed: Landmark[] = [];
  for (let i = 0; i < 21; i++) {
    const point = {
      x: (landmarks[i].x - ox) / scale,
      y: (landmarks[i].y - oy) / scale,
      z: (landmarks[i].z - oz) / scale,
    };
    placed.push(point);
    out[i * 3] = point.x;
    out[i * 3 + 1] = point.y;
    out[i * 3 + 2] = point.z;
  }
  LETTER_CURLS.forEach(([a, b, c], index) => {
    out[63 + index] = letterCurl(placed[a], placed[b], placed[c]);
  });
  return out;
}

export function normalizeHand(landmarks: Landmark[]): Float32Array | null {
  const features = handFeatures(landmarks);
  return features ? features.slice(0, 63) : null;
}

export function predictNormalized(model: LetterModel, input: Float32Array): { label: string; confidence: number; logits: Float32Array } {
  let values = input;
  model.layers.forEach((layer, index) => {
    const next = new Float32Array(layer.out);
    for (let o = 0; o < layer.out; o++) {
      let sum = layer.bias[o];
      const row = o * layer.inn;
      for (let i = 0; i < layer.inn; i++) sum += layer.weight[row + i] * values[i];
      next[o] = index === model.layers.length - 1 ? sum : Math.max(0, sum);
    }
    values = next;
  });
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) if (values[i] > max) max = values[i];
  let denom = 0;
  const probs = new Array<number>(values.length);
  for (let i = 0; i < values.length; i++) {
    probs[i] = Math.exp(values[i] - max);
    denom += probs[i];
  }
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { label: model.labels[best], confidence: probs[best] / denom, logits: values };
}

export function predictLetter(model: LetterModel, landmarks: Landmark[]): { label: string; confidence: number } | null {
  const features = handFeatures(landmarks);
  if (!features) return null;
  const width = model.layers[0]?.inn ?? features.length;
  const input = width === features.length ? features : features.slice(0, width);
  const result = predictNormalized(model, input);
  return { label: result.label, confidence: result.confidence };
}
