import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import smoke from './transformer-smoke.json';
import stemSmoke from './transformer-stem-smoke.json';
import { decodeF16, decodeF32, float16BitsToFloat32, logitsForSequence, modelFromJson, predictSequence } from './sequenceModel';

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

  it('matches PyTorch when a temporal conv stem is present', () => {
    const model = modelFromJson(stemSmoke);
    const flat = decodeF32(stemSmoke.input);
    const sequence: number[][] = [];
    for (let t = 0; t < model.seqLen; t++) {
      sequence.push(Array.from(flat.slice(t * model.featureDim, (t + 1) * model.featureDim)));
    }
    const actual = logitsForSequence(model, sequence);
    const expected = decodeF32(stemSmoke.logits);
    let max = 0;
    for (let i = 0; i < expected.length; i++) max = Math.max(max, Math.abs(actual[i] - expected[i]));
    expect(max).toBeLessThan(1e-4);
    expect(model.members[0].weights['dw1.weight']).toBeTruthy();
  });

  it('temperature leaves the label and softens confidence', () => {
    const model = modelFromJson(smoke);
    const flat = decodeF32(smoke.input);
    const sequence: number[][] = [];
    for (let t = 0; t < model.seqLen; t++) {
      sequence.push(Array.from(flat.slice(t * model.featureDim, (t + 1) * model.featureDim)));
    }
    const sharp = predictSequence(model, sequence, { tta: false });
    const soft = predictSequence({ ...model, temperature: 4 }, sequence, { tta: false });
    expect(soft.label).toBe(sharp.label);
    expect(soft.confidence).toBeLessThanOrEqual(sharp.confidence + 1e-6);
    expect(modelFromJson(smoke).temperature).toBe(1);
  });

  it('decodes float16 weights and can score with one ensemble member', () => {
    expect(float16BitsToFloat32(0x3c00)).toBe(1);
    expect(float16BitsToFloat32(0xc000)).toBe(-2);
    expect(float16BitsToFloat32(0x3800)).toBe(0.5);
    const bytes = new Uint8Array([0x00, 0x3c, 0x00, 0xc0, 0x00, 0x38]);
    let binary = '';
    for (const byte of bytes) binary += String.fromCharCode(byte);
    const decoded = decodeF16(btoa(binary));
    expect(Array.from(decoded)).toEqual([1, -2, 0.5]);

    const model = modelFromJson(smoke);
    const flat = decodeF32(smoke.input);
    const sequence: number[][] = [];
    for (let t = 0; t < model.seqLen; t++) {
      sequence.push(Array.from(flat.slice(t * model.featureDim, (t + 1) * model.featureDim)));
    }
    const full = predictSequence(model, sequence, { members: model.members.length, tta: false });
    const fast = predictSequence(model, sequence, { members: 1, tta: false });
    expect(full.label.length).toBeGreaterThan(0);
    expect(fast.probabilities).toHaveLength(model.labels.length);
  });
});
