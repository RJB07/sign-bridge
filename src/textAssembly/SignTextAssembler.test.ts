import { describe, expect, it } from 'vitest';
import { SignTextAssembler } from './SignTextAssembler';

describe('sign text assembler', () => {
  it('debounces a repeated gloss and keeps the next one', () => {
    let phrase: string[] = [];
    let time = 0;
    const assembler = new SignTextAssembler((next) => {
      phrase = next;
    }, () => undefined, () => time);
    time = 1000;
    assembler.pushSign('apple', 0.9);
    time = 1200;
    assembler.pushSign('apple', 0.9);
    time = 2000;
    assembler.pushSign('bed', 0.8);
    expect(phrase).toEqual(['apple', 'bed']);
  });

  it('groups fingerspelled letters and formats a sentence after a pause', () => {
    let phrase: string[] = [];
    const sentences: string[] = [];
    let time = 0;
    const assembler = new SignTextAssembler(
      (next) => {
        phrase = next;
      },
      (message) => sentences.push(message.text),
      () => time,
    );
    time = 0;
    assembler.pushSign('C', 0.8);
    time = 700;
    assembler.pushSign('A', 0.8);
    time = 1400;
    assembler.pushSign('T', 0.8);
    expect(phrase).toEqual(['CAT']);
    time = 5000;
    assembler.checkPauseBoundary();
    expect(sentences).toEqual(['CAT.']);
  });

  it('adds a question mark when the phrase ends with a wh-word', () => {
    const sentences: string[] = [];
    let time = 0;
    const assembler = new SignTextAssembler(() => undefined, (message) => sentences.push(message.text), () => time);
    time = 10;
    assembler.pushSign('apple', 0.9);
    time = 800;
    assembler.pushSign('where', 0.9);
    time = 5000;
    assembler.checkPauseBoundary();
    expect(sentences[0]).toBe('Apple where?');
  });

  it('undoes the last word, and a letter before a finished word', () => {
    let phrase: string[] = [];
    let time = 0;
    const assembler = new SignTextAssembler((next) => {
      phrase = next;
    }, () => undefined, () => time);
    time = 10;
    assembler.pushSign('hello', 0.9);
    time = 800;
    assembler.pushSign('yes', 0.9);
    assembler.undo();
    expect(phrase).toEqual(['hello']);
    time = 1600;
    assembler.pushSign('C', 0.8);
    assembler.undo();
    expect(assembler.getSpelling()).toEqual([]);
    expect(phrase).toEqual(['hello']);
  });

  it('sends the current line without waiting for a pause', () => {
    const sentences: string[] = [];
    let time = 0;
    const assembler = new SignTextAssembler(() => undefined, (message) => sentences.push(message.text), () => time);
    time = 10;
    assembler.pushSign('please', 0.9);
    assembler.commitLine();
    expect(sentences).toEqual(['Please.']);
    expect(assembler.getWords()).toEqual([]);
  });
});
