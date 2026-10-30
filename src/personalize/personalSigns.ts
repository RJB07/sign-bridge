export type PersonalSample = {
  id: string;
  label: string;
  vector: number[];
  createdAt: number;
};

const STORAGE_KEY = 'signbridge-personal-signs';
const MATCH_THRESHOLD = 0.8;
const MAX_SAMPLES = 40;

export function sequenceVector(sequence: number[][]): number[] {
  if (sequence.length === 0 || sequence[0].length === 0) return [];
  const width = sequence[0].length;
  const mean = new Array<number>(width).fill(0);
  for (const frame of sequence) {
    for (let index = 0; index < width; index += 1) mean[index] += frame[index] ?? 0;
  }
  let norm = 0;
  for (let index = 0; index < width; index += 1) {
    mean[index] /= sequence.length;
    norm += mean[index] * mean[index];
  }
  norm = Math.sqrt(norm);
  if (norm < 1e-8) return mean.map(() => 0);
  return mean.map((value) => Math.round((value / norm) * 10000) / 10000);
}

function cosine(left: number[], right: number[]): number {
  const width = Math.min(left.length, right.length);
  if (width === 0) return 0;
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < width; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] * left[index];
    rightNorm += right[index] * right[index];
  }
  if (leftNorm < 1e-8 || rightNorm < 1e-8) return 0;
  return dot / Math.sqrt(leftNorm * rightNorm);
}

export function sampleFromSequence(label: string, sequence: number[][]): PersonalSample {
  return {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    label: label.trim(),
    vector: sequenceVector(sequence),
    createdAt: Date.now(),
  };
}

export function matchPersonal(samples: PersonalSample[], sequence: number[][]): { label: string; score: number } | null {
  if (samples.length === 0) return null;
  const query = sequenceVector(sequence);
  const byLabel = new Map<string, number[][]>();
  for (const sample of samples) {
    if (!sample.label || sample.vector.length === 0 || sample.vector.length !== query.length) continue;
    const group = byLabel.get(sample.label) ?? [];
    group.push(sample.vector);
    byLabel.set(sample.label, group);
  }
  let bestLabel = '';
  let best = -1;
  let second = -1;
  for (const [label, vectors] of byLabel) {
    const width = vectors[0].length;
    const mean = new Array<number>(width).fill(0);
    for (const vector of vectors) {
      for (let index = 0; index < width; index += 1) mean[index] += vector[index] ?? 0;
    }
    for (let index = 0; index < width; index += 1) mean[index] /= vectors.length;
    const score = cosine(query, mean);
    if (score > best) {
      second = best;
      best = score;
      bestLabel = label;
    } else if (score > second) {
      second = score;
    }
  }
  if (!bestLabel || best < MATCH_THRESHOLD) return null;
  if (second >= 0 && best - second < 0.05) return null;
  return { label: bestLabel, score: best };
}

export function applyPersonalLabel<T extends { label: string; confidence: number }>(
  prediction: T,
  sequence: number[][],
  samples: PersonalSample[],
): T {
  const hit = matchPersonal(samples, sequence);
  if (!hit) return prediction;
  return { ...prediction, label: hit.label, confidence: Math.max(prediction.confidence, hit.score) };
}

export function loadPersonalSamples(): PersonalSample[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as PersonalSample[];
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((sample) => typeof sample.label === 'string' && Array.isArray(sample.vector)).slice(-MAX_SAMPLES);
  } catch {
    return [];
  }
}

export function persistPersonalSamples(samples: PersonalSample[]): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(samples.slice(-MAX_SAMPLES)));
}
