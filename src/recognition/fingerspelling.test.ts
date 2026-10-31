import { describe, expect, it } from 'vitest';
import { classifyFingerspelling, type FingerspelledLetter } from './fingerspelling';
import type { Landmark } from './features';

function point(x: number, y: number, z = 0): Landmark {
  return { x, y, z };
}

function buildHand(extended: [number, number, number, number], thumb: 'out' | 'tucked' | 'near-index-tip'): Landmark[] {
  const landmarks: Landmark[] = Array.from({ length: 21 }, () => point(0, 0, 0));
  const mcps = [5, 9, 13, 17];
  const tips = [8, 12, 16, 20];
  mcps.forEach((mcp, finger) => {
    const base = point(0.12 * finger, 0.5, 0);
    landmarks[mcp] = base;
    const reach = extended[finger] * Math.hypot(base.x, base.y);
    landmarks[tips[finger]] = point(base.x, base.y + reach, 0);
    landmarks[mcp + 1] = point(base.x, base.y + reach * 0.4, 0);
    landmarks[mcp + 2] = point(base.x, base.y + reach * 0.7, 0);
  });
  if (thumb === 'tucked') landmarks[4] = { ...landmarks[5] };
  else if (thumb === 'near-index-tip') landmarks[4] = { ...landmarks[8] };
  else landmarks[4] = point(-0.9, 0.1, 0);
  landmarks[1] = point(-0.1, 0.05, 0);
  landmarks[2] = point(-0.2, 0.1, 0);
  landmarks[3] = point(-0.3, 0.15, 0);
  return landmarks;
}

const shapes: Array<[FingerspelledLetter, [number, number, number, number], 'out' | 'tucked' | 'near-index-tip']> = [
  ['A', [0.2, 0.2, 0.2, 0.2], 'tucked'],
  ['B', [1.1, 1.1, 1.1, 1.1], 'tucked'],
  ['L', [1.1, 0.2, 0.2, 0.2], 'out'],
  ['V', [1.2, 1.2, 0.2, 0.2], 'tucked'],
  ['W', [1.1, 1.1, 1.1, 0.2], 'tucked'],
  ['Y', [0.2, 0.2, 0.2, 1.2], 'out'],
  ['I', [0.2, 0.2, 0.2, 1.2], 'tucked'],
  ['F', [0.3, 1.1, 1.1, 1.1], 'near-index-tip'],
  ['C', [0.7, 0.7, 0.7, 0.7], 'out'],
];

describe('fingerspelling handshapes', () => {
  it.each(shapes)('reads %s from distances rather than image-up', (letter, extended, thumb) => {
    const hand = buildHand(extended, thumb);
    if (letter === 'V') {
      hand[12] = point(0.7, 1.1, 0);
    }
    if (letter === 'C') {
      hand[4] = point(0.35, 0.7, 0);
    }
    expect(classifyFingerspelling(hand)).toBe(letter);
  });

  it('ignores an incomplete hand', () => {
    expect(classifyFingerspelling([{ x: 0, y: 0, z: 0 }])).toBeNull();
  });
});
