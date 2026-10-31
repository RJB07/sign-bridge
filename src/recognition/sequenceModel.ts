import { FEATURE_DIM, SEQ_LEN, mirrorSequence, resampleSequence } from './features';

export type Prediction = {
  label: string;
  confidence: number;
  margin: number;
  probabilities: number[];
};

export type WeightBlob = {
  weights: Record<string, string>;
  weightShapes: Record<string, number[]>;
  dtype?: 'f16' | 'f32';
  /** Which landmark layout this member was trained on, when the ensemble mixes them. */
  featureLayout?: 'legacy' | 'rich' | 'fine' | 'signer';
  /** Logit weight. Members are averaged by these weights. */
  blendWeight?: number;
};

export type SequenceModel = {
  format: 'signbridge-tx-v1';
  labels: string[];
  emitThreshold: number;
  marginThreshold: number;
  featureDim: number;
  seqLen: number;
  dModel: number;
  heads: number;
  layers: number;
  classThresholds: number[];
  tta: boolean;
  /** `rich` is nose-relative hands. `fine` adds face offsets and finger curl. `signer` scales shape by palm size. */
  featureLayout: 'legacy' | 'rich' | 'fine' | 'signer';
  /** Softmax temperature. 1 leaves confidence unchanged. Values above 1 reduce overconfidence. */
  temperature: number;
  members: WeightBlob[];
};

type ModelFile = {
  format?: string;
  featureDim: number;
  seqLen: number;
  dModel: number;
  heads: number;
  layers: number;
  labels: string[];
  emitThreshold: number;
  marginThreshold: number;
  classThresholds?: number[];
  tta?: boolean;
  dtype?: 'f16' | 'f32';
  featureLayout?: 'legacy' | 'rich' | 'fine' | 'signer';
  temperature?: number;
  members: WeightBlob[];
};

export type PredictOptions = {
  /** How many ensemble members to run. The live preview uses one. */
  members?: number;
  /** Mirror the sequence and average. Defaults to the model flag. */
  tta?: boolean;
};

const COMPOUNDS: Record<string, string> = {
  thankyou: 'thank you',
  callonphone: 'call on phone',
  frenchfries: 'french fries',
  glasswindow: 'glass window',
  haveto: 'have to',
  icecream: 'ice cream',
  minemy: 'mine',
  hesheit: 'he',
};

export function glossToEnglish(label: string): string {
  return COMPOUNDS[label] ?? label.replace(/[_]+/g, ' ').trim();
}

export function decodeF32(encoded: string): Float32Array {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer.slice(0));
}

export function float16BitsToFloat32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x03ff;
  if (exp === 0) {
    if (frac === 0) return sign * 0;
    return sign * 2 ** -14 * (frac / 1024);
  }
  if (exp === 31) return frac ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

export function decodeF16(encoded: string): Float32Array {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.byteLength / 2);
  for (let i = 0; i < out.length; i++) out[i] = float16BitsToFloat32(view.getUint16(i * 2, true));
  return out;
}

const decodedWeights = new WeakMap<WeightBlob, Map<string, { data: Float32Array; shape: number[] }>>();

function tensor(blob: WeightBlob, name: string): { data: Float32Array; shape: number[] } {
  let table = decodedWeights.get(blob);
  if (!table) {
    table = new Map();
    decodedWeights.set(blob, table);
  }
  const cached = table.get(name);
  if (cached) return cached;
  const encoded = blob.weights[name];
  const value = {
    data: blob.dtype === 'f16' ? decodeF16(encoded) : decodeF32(encoded),
    shape: blob.weightShapes[name],
  };
  table.set(name, value);
  return value;
}

function linearScalar(x: Float32Array, rows: number, cols: number, weight: Float32Array, bias: Float32Array, out: Float32Array, outCols: number): void {
  for (let r = 0; r < rows; r++) {
    const xRow = r * cols;
    for (let c = 0; c < outCols; c++) {
      let sum = bias[c];
      const wRow = c * cols;
      for (let k = 0; k < cols; k++) sum += x[xRow + k] * weight[wRow + k];
      out[r * outCols + c] = sum;
    }
  }
}

/** Four outputs share each input load. Same math as the scalar loop, aside from one ulp. */
function linearBlocked(x: Float32Array, rows: number, cols: number, weight: Float32Array, bias: Float32Array): Float32Array {
  const outCols = weight.length / cols;
  const out = new Float32Array(rows * outCols);
  if (rows < 8 || cols < 64 || outCols < 8) {
    linearScalar(x, rows, cols, weight, bias, out, outCols);
    return out;
  }
  const tail = cols & ~3;
  for (let r = 0; r < rows; r++) {
    const xRow = r * cols;
    let c = 0;
    for (; c + 3 < outCols; c += 4) {
      let s0 = bias[c];
      let s1 = bias[c + 1];
      let s2 = bias[c + 2];
      let s3 = bias[c + 3];
      const w0 = c * cols;
      const w1 = w0 + cols;
      const w2 = w1 + cols;
      const w3 = w2 + cols;
      let k = 0;
      for (; k < tail; k += 4) {
        const x0 = x[xRow + k];
        const x1 = x[xRow + k + 1];
        const x2 = x[xRow + k + 2];
        const x3 = x[xRow + k + 3];
        s0 += x0 * weight[w0 + k] + x1 * weight[w0 + k + 1] + x2 * weight[w0 + k + 2] + x3 * weight[w0 + k + 3];
        s1 += x0 * weight[w1 + k] + x1 * weight[w1 + k + 1] + x2 * weight[w1 + k + 2] + x3 * weight[w1 + k + 3];
        s2 += x0 * weight[w2 + k] + x1 * weight[w2 + k + 1] + x2 * weight[w2 + k + 2] + x3 * weight[w2 + k + 3];
        s3 += x0 * weight[w3 + k] + x1 * weight[w3 + k + 1] + x2 * weight[w3 + k + 2] + x3 * weight[w3 + k + 3];
      }
      for (; k < cols; k++) {
        const xv = x[xRow + k];
        s0 += xv * weight[w0 + k];
        s1 += xv * weight[w1 + k];
        s2 += xv * weight[w2 + k];
        s3 += xv * weight[w3 + k];
      }
      const base = r * outCols + c;
      out[base] = s0;
      out[base + 1] = s1;
      out[base + 2] = s2;
      out[base + 3] = s3;
    }
    for (; c < outCols; c++) {
      let sum = bias[c];
      const wRow = c * cols;
      for (let k = 0; k < cols; k++) sum += x[xRow + k] * weight[wRow + k];
      out[r * outCols + c] = sum;
    }
  }
  return out;
}

function linear(x: Float32Array, rows: number, cols: number, weight: Float32Array, bias: Float32Array): Float32Array {
  return linearBlocked(x, rows, cols, weight, bias);
}

function layerNorm(x: Float32Array, rows: number, cols: number, weight: Float32Array, bias: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let r = 0; r < rows; r++) {
    let mean = 0;
    for (let c = 0; c < cols; c++) mean += x[r * cols + c];
    mean /= cols;
    let variance = 0;
    for (let c = 0; c < cols; c++) {
      const d = x[r * cols + c] - mean;
      variance += d * d;
    }
    variance /= cols;
    const inv = 1 / Math.sqrt(variance + 1e-5);
    for (let c = 0; c < cols; c++) {
      out[r * cols + c] = (x[r * cols + c] - mean) * inv * weight[c] + bias[c];
    }
  }
  return out;
}

function conv1dDepthwise(x: Float32Array, time: number, channels: number, weight: Float32Array, bias: Float32Array, kernel: number): Float32Array {
  const out = new Float32Array(time * channels);
  const pad = Math.floor(kernel / 2);
  for (let t = 0; t < time; t++) {
    for (let c = 0; c < channels; c++) {
      let sum = bias[c];
      const wBase = c * kernel;
      for (let k = 0; k < kernel; k++) {
        const src = t + k - pad;
        if (src < 0 || src >= time) continue;
        sum += x[src * channels + c] * weight[wBase + k];
      }
      out[t * channels + c] = sum;
    }
  }
  return out;
}

function conv1dSame(x: Float32Array, time: number, channels: number, weight: Float32Array, bias: Float32Array, kernel: number): Float32Array {
  const outCh = bias.length;
  const out = new Float32Array(time * outCh);
  const pad = Math.floor(kernel / 2);
  for (let t = 0; t < time; t++) {
    for (let oc = 0; oc < outCh; oc++) {
      let sum = bias[oc];
      const wBase = oc * channels * kernel;
      for (let ic = 0; ic < channels; ic++) {
        const icBase = wBase + ic * kernel;
        for (let k = 0; k < kernel; k++) {
          const src = t + k - pad;
          if (src < 0 || src >= time) continue;
          sum += x[src * channels + ic] * weight[icBase + k];
        }
      }
      out[t * outCh + oc] = sum;
    }
  }
  return out;
}

function mha(x: Float32Array, time: number, dim: number, inW: Float32Array, inB: Float32Array, outW: Float32Array, outB: Float32Array, heads: number): Float32Array {
  const qkv = linear(x, time, dim, inW, inB);
  const dHead = dim / heads;
  const scale = 1 / Math.sqrt(dHead);
  const mixed = new Float32Array(time * dim);
  for (let h = 0; h < heads; h++) {
    const scores = new Float32Array(time * time);
    for (let i = 0; i < time; i++) {
      let max = -Infinity;
      for (let j = 0; j < time; j++) {
        let dot = 0;
        for (let k = 0; k < dHead; k++) {
          const q = qkv[i * dim * 3 + h * dHead + k];
          const key = qkv[j * dim * 3 + dim + h * dHead + k];
          dot += q * key;
        }
        const value = dot * scale;
        scores[i * time + j] = value;
        if (value > max) max = value;
      }
      let denom = 0;
      for (let j = 0; j < time; j++) {
        const e = Math.exp(scores[i * time + j] - max);
        scores[i * time + j] = e;
        denom += e;
      }
      for (let j = 0; j < time; j++) scores[i * time + j] /= denom;
      for (let k = 0; k < dHead; k++) {
        let sum = 0;
        for (let j = 0; j < time; j++) {
          sum += scores[i * time + j] * qkv[j * dim * 3 + 2 * dim + h * dHead + k];
        }
        mixed[i * dim + h * dHead + k] = sum;
      }
    }
  }
  return linear(mixed, time, dim, outW, outB);
}

export function logitsForMember(sequence: number[][], blob: WeightBlob, dModel: number, heads: number, layers: number): Float32Array {
  const time = sequence.length;
  const feat = sequence[0].length;
  const flat = new Float32Array(time * feat);
  for (let t = 0; t < time; t++) {
    for (let d = 0; d < feat; d++) flat[t * feat + d] = sequence[t][d];
  }
  const proj = tensor(blob, 'proj.weight');
  const projB = tensor(blob, 'proj.bias');
  const pos = tensor(blob, 'pos');
  let h = linear(flat, time, feat, proj.data, projB.data);
  if (blob.weights['shape.fc1.weight']) {
    const shape = new Float32Array(time * 63);
    for (let t = 0; t < time; t++) {
      for (let d = 0; d < 63; d++) shape[t * 63 + d] = flat[t * feat + 147 + d];
    }
    const fc1 = tensor(blob, 'shape.fc1.weight');
    let hidden = linear(shape, time, 63, fc1.data, tensor(blob, 'shape.fc1.bias').data);
    for (let i = 0; i < hidden.length; i++) if (hidden[i] < 0) hidden[i] = 0;
    const fc2 = tensor(blob, 'shape.fc2.weight');
    const embedded = linear(hidden, time, 32, fc2.data, tensor(blob, 'shape.fc2.bias').data);
    for (let i = 0; i < h.length; i++) h[i] += embedded[i];
  }
  if (blob.weights['dw1.weight']) {
    const dw1 = tensor(blob, 'dw1.weight');
    let y = conv1dDepthwise(h, time, dModel, dw1.data, tensor(blob, 'dw1.bias').data, dw1.shape[2]);
    const pw1 = tensor(blob, 'pw1.weight');
    y = conv1dSame(y, time, dModel, pw1.data, tensor(blob, 'pw1.bias').data, pw1.shape[2]);
    for (let i = 0; i < y.length; i++) if (y[i] < 0) y[i] = 0;
    const dw2 = tensor(blob, 'dw2.weight');
    y = conv1dDepthwise(y, time, dModel, dw2.data, tensor(blob, 'dw2.bias').data, dw2.shape[2]);
    const pw2 = tensor(blob, 'pw2.weight');
    y = conv1dSame(y, time, pw2.shape[1], pw2.data, tensor(blob, 'pw2.bias').data, pw2.shape[2]);
    for (let i = 0; i < y.length; i++) if (y[i] < 0) y[i] = 0;
    y = layerNorm(y, time, dModel, tensor(blob, 'stem_ln.weight').data, tensor(blob, 'stem_ln.bias').data);
    for (let i = 0; i < h.length; i++) h[i] += y[i];
  } else if (blob.weights['conv1.weight']) {
    const conv1 = tensor(blob, 'conv1.weight');
    const conv1Bias = tensor(blob, 'conv1.bias');
    let y = conv1dSame(h, time, dModel, conv1.data, conv1Bias.data, conv1.shape[2]);
    for (let i = 0; i < y.length; i++) if (y[i] < 0) y[i] = 0;
    const conv2 = tensor(blob, 'conv2.weight');
    const conv2Bias = tensor(blob, 'conv2.bias');
    y = conv1dSame(y, time, conv2.shape[1], conv2.data, conv2Bias.data, conv2.shape[2]);
    for (let i = 0; i < y.length; i++) if (y[i] < 0) y[i] = 0;
    const stemLn = tensor(blob, 'stem_ln.weight');
    const stemBias = tensor(blob, 'stem_ln.bias');
    y = layerNorm(y, time, dModel, stemLn.data, stemBias.data);
    for (let i = 0; i < h.length; i++) h[i] += y[i];
  }
  for (let t = 0; t < time; t++) {
    for (let d = 0; d < dModel; d++) h[t * dModel + d] += pos.data[t * dModel + d];
  }
  for (let layer = 0; layer < layers; layer++) {
    const p = `b${layer}`;
    const ln1 = layerNorm(h, time, dModel, tensor(blob, `${p}.ln1.weight`).data, tensor(blob, `${p}.ln1.bias`).data);
    const attn = mha(
      ln1,
      time,
      dModel,
      tensor(blob, `${p}.attn.in_weight`).data,
      tensor(blob, `${p}.attn.in_bias`).data,
      tensor(blob, `${p}.attn.out_weight`).data,
      tensor(blob, `${p}.attn.out_bias`).data,
      heads,
    );
    for (let i = 0; i < h.length; i++) h[i] += attn[i];
    const ln2 = layerNorm(h, time, dModel, tensor(blob, `${p}.ln2.weight`).data, tensor(blob, `${p}.ln2.bias`).data);
    const ff1w = tensor(blob, `${p}.ff1.weight`);
    const ff1b = tensor(blob, `${p}.ff1.bias`);
    const hidden = linear(ln2, time, dModel, ff1w.data, ff1b.data);
    for (let i = 0; i < hidden.length; i++) if (hidden[i] < 0) hidden[i] = 0;
    const ff2 = linear(hidden, time, ff1w.shape[0], tensor(blob, `${p}.ff2.weight`).data, tensor(blob, `${p}.ff2.bias`).data);
    for (let i = 0; i < h.length; i++) h[i] += ff2[i];
  }
  const normed = layerNorm(h, time, dModel, tensor(blob, 'final_ln.weight').data, tensor(blob, 'final_ln.bias').data);
  const pooled = new Float32Array(dModel);
  for (let t = 0; t < time; t++) {
    for (let d = 0; d < dModel; d++) pooled[d] += normed[t * dModel + d];
  }
  for (let d = 0; d < dModel; d++) pooled[d] /= time;
  const fc = tensor(blob, 'fc.weight');
  return linear(pooled, 1, dModel, fc.data, tensor(blob, 'fc.bias').data);
}

export function modelFromJson(payload: ModelFile): SequenceModel {
  if (payload.format && payload.format !== 'signbridge-tx-v1') {
    throw new Error(`Unsupported model format ${payload.format}`);
  }
  return {
    format: 'signbridge-tx-v1',
    labels: payload.labels,
    emitThreshold: payload.emitThreshold,
    marginThreshold: payload.marginThreshold,
    classThresholds: payload.classThresholds ?? payload.labels.map(() => payload.emitThreshold),
    tta: Boolean(payload.tta),
    featureLayout: payload.featureLayout === 'signer'
      ? 'signer'
      : payload.featureLayout === 'fine'
        ? 'fine'
        : payload.featureLayout === 'rich'
          ? 'rich'
          : 'legacy',
    temperature: typeof payload.temperature === 'number' && payload.temperature > 0 ? payload.temperature : 1,
    featureDim: payload.featureDim,
    seqLen: payload.seqLen,
    dModel: payload.dModel,
    heads: payload.heads,
    layers: payload.layers,
    members: payload.members.map((member) => ({
      ...member,
      dtype: payload.dtype === 'f16' ? 'f16' as const : 'f32' as const,
      featureLayout: member.featureLayout === 'signer' || member.featureLayout === 'fine' || member.featureLayout === 'rich' || member.featureLayout === 'legacy'
        ? member.featureLayout
        : undefined,
      blendWeight: typeof member.blendWeight === 'number' && member.blendWeight > 0 ? member.blendWeight : undefined,
    })),
  };
}

function blendWeight(member: WeightBlob): number {
  return member.blendWeight && member.blendWeight > 0 ? member.blendWeight : 1;
}

function prepareSequence(sequence: number[][], model: SequenceModel): number[][] {
  return sequence.length === model.seqLen ? sequence : resampleSequence(sequence, model.seqLen);
}

export function logitsForSequence(model: SequenceModel, sequence: number[][]): Float32Array {
  const prepared = sequence.length === model.seqLen ? sequence : resampleSequence(sequence, model.seqLen);
  if (prepared.length !== model.seqLen || prepared[0]?.length !== model.featureDim) {
    throw new Error(`Expected ${model.seqLen}x${model.featureDim} features`);
  }
  const acc = new Float32Array(model.labels.length);
  let weight = 0;
  for (const member of model.members) {
    const scale = blendWeight(member);
    const logits = logitsForMember(prepared, member, model.dModel, model.heads, model.layers);
    for (let i = 0; i < acc.length; i++) acc[i] += logits[i] * scale;
    weight += scale;
  }
  for (let i = 0; i < acc.length; i++) acc[i] /= weight || 1;
  return acc;
}

export type LayoutSequences = {
  fine?: number[][];
  signer?: number[][];
  legacy?: number[][];
  rich?: number[][];
};

export function predictPrepared(model: SequenceModel, sequences: LayoutSequences, options: PredictOptions = {}): Prediction {
  const useTta = options.tta ?? model.tta;
  const memberCount = Math.max(1, Math.min(options.members ?? model.members.length, model.members.length));
  const members = model.members.slice(0, memberCount);
  const cached = new Map<string, number[][]>();
  const viewFor = (layout: SequenceModel['featureLayout']): number[][] => {
    const source = layout === 'signer'
      ? sequences.signer ?? sequences.fine
      : layout === 'legacy'
        ? sequences.legacy ?? sequences.fine
        : layout === 'rich'
          ? sequences.rich ?? sequences.fine
          : sequences.fine ?? sequences.signer ?? sequences.legacy ?? sequences.rich;
    if (!source || source.length === 0) throw new Error('Missing features for this ensemble');
    const key = layout + (useTta ? ':tta' : '');
    const hit = cached.get(key);
    if (hit) return hit;
    const prepared = prepareSequence(source, model);
    cached.set(key, prepared);
    return prepared;
  };
  const acc = new Float32Array(model.labels.length);
  let weight = 0;
  for (const member of members) {
    const layout = member.featureLayout ?? model.featureLayout;
    const prepared = viewFor(layout);
    const mirrorDims = layout === 'fine' || layout === 'signer' ? 384 : undefined;
    const views = useTta ? [prepared, mirrorSequence(prepared, mirrorDims)] : [prepared];
    const scale = blendWeight(member);
    for (const view of views) {
      const logits = logitsForMember(view, member, model.dModel, model.heads, model.layers);
      for (let i = 0; i < acc.length; i++) acc[i] += logits[i] * scale;
      weight += scale;
    }
  }
  const temperature = model.temperature > 0 ? model.temperature : 1;
  for (let i = 0; i < acc.length; i++) acc[i] /= (weight || 1) * temperature;
  let max = -Infinity;
  for (let i = 0; i < acc.length; i++) if (acc[i] > max) max = acc[i];
  const probabilities = new Array<number>(acc.length);
  let denom = 0;
  for (let i = 0; i < acc.length; i++) {
    const e = Math.exp(acc[i] - max);
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

export function predictSequence(model: SequenceModel, sequence: number[][], options: PredictOptions = {}): Prediction {
  return predictPrepared(model, { fine: sequence, signer: sequence, legacy: sequence, rich: sequence }, options);
}

export const MODEL_FEATURE_DIM = FEATURE_DIM;
export const MODEL_SEQ_LEN = SEQ_LEN;
