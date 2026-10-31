import { describe, expect, it } from 'vitest';
import model from '../../public/models/asl-en-sequence.json';
import { glossesToEnglish } from '../textAssembly/glossEnglish';
import { QUICK_PHRASES, phraseInRecognizer } from './vocabularyGap';

describe('everyday vocabulary', () => {
  it('matches the shipped 250 glosses', () => {
    const labels = new Set(model.labels);
    expect(labels.size).toBe(250);
    for (const phrase of QUICK_PHRASES) {
      expect(phraseInRecognizer(phrase)).toBe(phrase.gloss !== null && labels.has(phrase.gloss));
    }
    expect(QUICK_PHRASES.filter((phrase) => phrase.gloss === null).map((phrase) => phrase.text)).toEqual([
      'help',
      'sorry',
      'name',
      'doctor',
      'bathroom',
    ]);
  });

  it('puts a missing word on the caption as English', () => {
    expect(glossesToEnglish(['help']).english).toBe('Help.');
    expect(glossesToEnglish(['sorry']).english).toBe('Sorry.');
    expect(glossesToEnglish(['doctor']).english).toBe('Doctor.');
    expect(glossesToEnglish(['bathroom']).english).toBe('Bathroom.');
    expect(glossesToEnglish(['thank you', 'please']).english).toBe('Thank you, please.');
  });
});
