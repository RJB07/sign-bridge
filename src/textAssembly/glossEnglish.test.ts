import { describe, expect, it } from 'vitest';
import { glossesToEnglish } from './glossEnglish';

describe('gloss to English', () => {
  it('formats greetings and keeps the gloss line', () => {
    const line = glossesToEnglish(['hello', 'thank you', 'please']);
    expect(line.english).toBe('Hello, thank you, please.');
    expect(line.glosses).toBe('hello · thank you · please');
  });

  it('moves a final wh-word and asks the question', () => {
    expect(glossesToEnglish(['apple', 'where']).english).toBe('Where is the apple?');
    expect(glossesToEnglish(['you', 'go', 'where']).english).toBe('Where do you go?');
    expect(glossesToEnglish(['yesterday', 'store', 'where']).english).toBe('Yesterday, where is the store?');
  });

  it('builds a few high-frequency clause shapes', () => {
    expect(glossesToEnglish(['me', 'want', 'water']).english).toBe('I want water.');
    expect(glossesToEnglish(['me', 'not', 'want', 'milk']).english).toBe('I do not want milk.');
    expect(glossesToEnglish(['my', 'name', 'CAT']).english).toBe('My name is CAT.');
    expect(glossesToEnglish(['yesterday', 'me', 'go', 'store']).english).toBe('Yesterday, I go to the store.');
    expect(glossesToEnglish(['water', 'please']).english).toBe('Water, please.');
  });

  it('covers questions, negation, time, and topic-comment', () => {
    expect(glossesToEnglish(['where', 'you', 'go']).english).toBe('Where do you go?');
    expect(glossesToEnglish(['you', 'not', 'like', 'milk']).english).toBe('You do not like milk.');
    expect(glossesToEnglish(['tomorrow', 'me', 'eat']).english).toBe('Tomorrow, I eat.');
    expect(glossesToEnglish(['store', 'me', 'go']).english).toBe('I go to the store.');
    expect(glossesToEnglish(['apple', 'me', 'want']).english).toBe('I want apple.');
    expect(glossesToEnglish(['milk', 'me', 'not', 'want']).english).toBe('I do not want milk.');
    expect(glossesToEnglish(['morning', 'store', 'me', 'go']).english).toBe('Morning, I go to the store.');
  });

  it('leaves an unknown line close to the glosses', () => {
    const line = glossesToEnglish(['book', 'bird']);
    expect(line.english).toBe('Book bird.');
    expect(line.note).toMatch(/Not full ASL grammar/);
  });
});
