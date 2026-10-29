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

export function normalizeHand(landmarks: Landmark[]): Float32Array | null {
  if (landmarks.length < 21) return null;
  const out = new Float32Array(63);
  const ox = landmarks[0].x;
  const oy = landmarks[0].y;
  const oz = landmarks[0].z;
  const scale = Math.hypot(landmarks[9].x - ox, landmarks[9].y - oy) || 1;
  for (let i = 0; i < 21; i++) {
    out[i * 3] = (landmarks[i].x - ox) / scale;
    out[i * 3 + 1] = (landmarks[i].y - oy) / scale;
    out[i * 3 + 2] = (landmarks[i].z - oz) / scale;
  }
  return out;
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
  const input = normalizeHand(landmarks);
  if (!input) return null;
  const result = predictNormalized(model, input);
  return { label: result.label, confidence: result.confidence };
}
