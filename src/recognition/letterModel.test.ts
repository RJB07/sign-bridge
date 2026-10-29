import { describe, expect, it } from 'vitest';
import smoke from './letter-smoke.json';
import letterFile from '../../public/models/asl-fingerspelling.json';
import { decodeF32 } from './sequenceModel';
import { letterModelFromJson, predictNormalized } from './letterModel';

describe('fingerspelling network', () => {
  it('matches PyTorch logits on one vector', () => {
    const model = letterModelFromJson(letterFile);
    const actual = predictNormalized(model, decodeF32(smoke.input)).logits;
    const expected = decodeF32(smoke.logits);
    let max = 0;
    for (let i = 0; i < expected.length; i++) max = Math.max(max, Math.abs(actual[i] - expected[i]));
    expect(max).toBeLessThan(1e-4);
  });
});
