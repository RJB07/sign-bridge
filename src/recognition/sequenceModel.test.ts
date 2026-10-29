import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import smoke from './transformer-smoke.json';
import { decodeF32, logitsForSequence, modelFromJson } from './sequenceModel';

describe('landmark transformer', () => {
  it('matches PyTorch on a tiny network', () => {
    const model = modelFromJson(smoke);
    const flat = decodeF32(smoke.input);
    const sequence: number[][] = [];
    for (let t = 0; t < model.seqLen; t++) {
      sequence.push(Array.from(flat.slice(t * model.featureDim, (t + 1) * model.featureDim)));
    }
    const actual = logitsForSequence(model, sequence);
    const expected = decodeF32(smoke.logits);
    let max = 0;
    for (let i = 0; i < expected.length; i++) max = Math.max(max, Math.abs(actual[i] - expected[i]));
    expect(max).toBeLessThan(1e-4);
  });

  it('matches the shipped ensemble when that file is the transformer format', () => {
    const raw = readFileSync('public/models/asl-en-sequence.json', 'utf8');
    const payload = JSON.parse(raw);
    if (payload.format !== 'signbridge-tx-v1') return;
    const parity = JSON.parse(readFileSync('public/models/parity.json', 'utf8'));
    const model = modelFromJson(payload);
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
