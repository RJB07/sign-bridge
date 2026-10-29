import { FEATURE_DIM, SEQ_LEN, resampleSequence } from './features';

export type Prediction = {
  label: string;
  confidence: number;
  margin: number;
  probabilities: number[];
};

type ModelFile = {
  featureDim: number;
  seqLen: number;
  hidden: number;
  labels: string[];
  emitThreshold: number;
  marginThreshold: number;
  weights: Record<string, string>;
  weightShapes: Record<string, number[]>;
};

export type SequenceModel = {
  labels: string[];
  emitThreshold: number;
  marginThreshold: number;
  hidden: number;
  featureDim: number;
  seqLen: number;
  weights: {
    conv1Weight: Float32Array;
    conv1Bias: Float32Array;
    conv2Weight: Float32Array;
    conv2Bias: Float32Array;
    gruWeightIh: Float32Array;
    gruWeightHh: Float32Array;
    gruBiasIh: Float32Array;
    gruBiasHh: Float32Array;
    fcWeight: Float32Array;
    fcBias: Float32Array;
  };
};

export function decodeF32(encoded: string): Float32Array {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer.slice(0));
}

export function modelFromJson(payload: ModelFile): SequenceModel {
  const w = payload.weights;
  return {
    labels: payload.labels,
    emitThreshold: payload.emitThreshold,
    marginThreshold: payload.marginThreshold,
    hidden: payload.hidden,
    featureDim: payload.featureDim,
    seqLen: payload.seqLen,
    weights: {
      conv1Weight: decodeF32(w['conv1.weight']),
      conv1Bias: decodeF32(w['conv1.bias']),
      conv2Weight: decodeF32(w['conv2.weight']),
      conv2Bias: decodeF32(w['conv2.bias']),
      gruWeightIh: decodeF32(w['gru.weight_ih']),
      gruWeightHh: decodeF32(w['gru.weight_hh']),
      gruBiasIh: decodeF32(w['gru.bias_ih']),
      gruBiasHh: decodeF32(w['gru.bias_hh']),
      fcWeight: decodeF32(w['fc.weight']),
      fcBias: decodeF32(w['fc.bias']),
    },
  };
}

function reluInto(values: Float32Array): void {
  for (let i = 0; i < values.length; i++) if (values[i] < 0) values[i] = 0;
}

function conv1dSame(seq: Float32Array, time: number, channels: number, weight: Float32Array, bias: Float32Array, outChannels: number): Float32Array {
  const out = new Float32Array(time * outChannels);
  for (let t = 0; t < time; t++) {
    for (let oc = 0; oc < outChannels; oc++) {
      let total = bias[oc];
      for (let ic = 0; ic < channels; ic++) {
        const wBase = (oc * channels + ic) * 3;
        for (let k = 0; k < 3; k++) {
          const src = t + k - 1;
          if (src < 0 || src >= time) continue;
          total += weight[wBase + k] * seq[src * channels + ic];
        }
      }
      out[t * outChannels + oc] = total;
    }
  }
  return out;
}

function sigmoid(value: number): number {
  if (value > 40) return 1;
  if (value < -40) return 0;
  return 1 / (1 + Math.exp(-value));
}

function gruMean(seq: Float32Array, time: number, hidden: number, weightIh: Float32Array, weightHh: Float32Array, biasIh: Float32Array, biasHh: Float32Array): Float32Array {
  const state = new Float32Array(hidden);
  const sum = new Float32Array(hidden);
  const gates = 3 * hidden;
  for (let t = 0; t < time; t++) {
    const gx = new Float32Array(gates);
    const gh = new Float32Array(gates);
    for (let i = 0; i < gates; i++) {
      let sx = biasIh[i];
      let sh = biasHh[i];
      const row = i * hidden;
      const xBase = t * hidden;
      for (let j = 0; j < hidden; j++) {
        sx += weightIh[row + j] * seq[xBase + j];
        sh += weightHh[row + j] * state[j];
      }
      gx[i] = sx;
      gh[i] = sh;
    }
    for (let i = 0; i < hidden; i++) {
      const r = sigmoid(gx[i] + gh[i]);
      const z = sigmoid(gx[hidden + i] + gh[hidden + i]);
      const n = Math.tanh(gx[2 * hidden + i] + r * gh[2 * hidden + i]);
      state[i] = (1 - z) * n + z * state[i];
      sum[i] += state[i];
    }
  }
  for (let i = 0; i < hidden; i++) sum[i] /= time;
  return sum;
}

export function logitsForSequence(model: SequenceModel, sequence: number[][]): Float32Array {
  const prepared = sequence.length === model.seqLen ? sequence : resampleSequence(sequence, model.seqLen);
  if (prepared.length !== model.seqLen || prepared[0]?.length !== model.featureDim) {
    throw new Error(`Expected ${model.seqLen}x${model.featureDim} features`);
  }
  const flat = new Float32Array(model.seqLen * model.featureDim);
  for (let t = 0; t < model.seqLen; t++) {
    for (let d = 0; d < model.featureDim; d++) flat[t * model.featureDim + d] = prepared[t][d];
  }
  const hidden = model.hidden;
  const h1 = conv1dSame(flat, model.seqLen, FEATURE_DIM, model.weights.conv1Weight, model.weights.conv1Bias, hidden);
  reluInto(h1);
  const h2 = conv1dSame(h1, model.seqLen, hidden, model.weights.conv2Weight, model.weights.conv2Bias, hidden);
  reluInto(h2);
  const pooled = gruMean(
    h2,
    model.seqLen,
    hidden,
    model.weights.gruWeightIh,
    model.weights.gruWeightHh,
    model.weights.gruBiasIh,
    model.weights.gruBiasHh,
  );
  const classes = model.labels.length;
  const logits = new Float32Array(classes);
  for (let c = 0; c < classes; c++) {
    let total = model.weights.fcBias[c];
    const row = c * hidden;
    for (let h = 0; h < hidden; h++) total += model.weights.fcWeight[row + h] * pooled[h];
    logits[c] = total;
  }
  return logits;
}

export function predictSequence(model: SequenceModel, sequence: number[][]): Prediction {
  const logits = logitsForSequence(model, sequence);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const probabilities = new Array(logits.length);
  let denom = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    probabilities[i] = e;
    denom += e;
  }
  let best = 0;
  let second = -Infinity;
  for (let i = 0; i < probabilities.length; i++) {
    probabilities[i] /= denom;
    if (probabilities[i] > probabilities[best]) best = i;
  }
  for (let i = 0; i < probabilities.length; i++) {
    if (i !== best && probabilities[i] > second) second = probabilities[i];
  }
  return {
    label: model.labels[best] ?? '',
    confidence: probabilities[best] ?? 0,
    margin: (probabilities[best] ?? 0) - (second === -Infinity ? 0 : second),
    probabilities,
  };
}

export function glossToEnglish(label: string): string {
  const known: Record<string, string> = { callonphone: 'call on phone' };
  return known[label] ?? label.replace(/[_]+/g, ' ').trim();
}
