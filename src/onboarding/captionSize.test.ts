import { describe, expect, it } from 'vitest';
import { defaultCaptionSize } from './captionSize';

describe('caption size', () => {
  it('uses the largest step on a phone when nothing is saved', () => {
    expect(defaultCaptionSize(390, null)).toBe('xl');
    expect(defaultCaptionSize(800, null)).toBe('lg');
  });

  it('keeps a saved choice', () => {
    expect(defaultCaptionSize(390, 'sm')).toBe('sm');
  });
});
