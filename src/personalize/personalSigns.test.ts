import { describe, expect, it } from 'vitest';
import { applyPersonalLabel, matchPersonal, sampleFromSequence, type PersonalSample } from './personalSigns';

function frame(first: number, second: number): number[][] {
  return [
    [first, second, 0.2],
    [first, second, 0.2],
  ];
}

describe('personal signs', () => {
  it('matches a saved sign and ignores a different one', () => {
    const hello = sampleFromSequence('hello', frame(1, 0));
    const water = sampleFromSequence('water', frame(0, 1));
    expect(matchPersonal([hello, water], frame(1, 0))?.label).toBe('hello');
    expect(matchPersonal([hello], frame(0, 1))).toBeNull();
  });

  it('replaces the shared guess when the saved sign is close', () => {
    const samples: PersonalSample[] = [sampleFromSequence('home sign', frame(1, 0))];
    const next = applyPersonalLabel({ label: 'book', confidence: 0.4 }, frame(1, 0), samples);
    expect(next.label).toBe('home sign');
    expect(next.confidence).toBeGreaterThan(0.4);
  });
});
