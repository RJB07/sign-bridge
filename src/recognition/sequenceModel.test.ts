import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { decodeF32, logitsForSequence, modelFromJson } from './sequenceModel';

const modelPath = 'public/models/asl-en-sequence.json';
const parityPath = 'public/models/parity.json';
const ready = existsSync(modelPath) && existsSync(parityPath);

describe('exported sequence model', () => {
  it.skipIf(!ready)('matches the PyTorch logits on the parity clip', () => {
    const model = modelFromJson(JSON.parse(readFileSync(modelPath, 'utf8')));
    const parity = JSON.parse(readFileSync(parityPath, 'utf8')) as { input: string; logits: string };
    const flat = decodeF32(parity.input);
    const sequence: number[][] = [];
    for (let t = 0; t < model.seqLen; t++) {
      sequence.push(Array.from(flat.slice(t * model.featureDim, (t + 1) * model.featureDim)));
    }
    const actual = logitsForSequence(model, sequence);
    const expected = decodeF32(parity.logits);
    let max = 0;
    for (let i = 0; i < expected.length; i++) max = Math.max(max, Math.abs(actual[i] - expected[i]));
    expect(max).toBeLessThan(1e-3);
  });
});
