import { describe, expect, it } from 'vitest';
import { correctSpelling } from './spellCorrect';

describe('fingerspelled words', () => {
  it('keeps a correct everyday word and fixes a close miss', () => {
    expect(correctSpelling('HELP')).toBe('help');
    expect(correctSpelling('HLEP')).toBe('help');
    expect(correctSpelling('DOCTER')).toBe('doctor');
    expect(correctSpelling('DOKTOR')).toBe('doctor');
    expect(correctSpelling('sorry')).toBe('sorry');
  });

  it('leaves a string alone when nothing in the list is close', () => {
    expect(correctSpelling('XYZQ')).toBe('XYZQ');
    expect(correctSpelling('NO')).toBe('NO');
  });
});
