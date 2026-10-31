/**
 * Everyday words checked against the 250 PopSign glosses.
 * A phrase with a gloss is in the recognizer. The others can be tapped onto
 * the caption as English, or fingerspelled while the hand is still.
 */
export type QuickPhrase = {
  id: string;
  label: string;
  /** Words placed on the caption. */
  text: string;
  /** PopSign gloss id when this phrase is one of the 250. */
  gloss: string | null;
};

export const QUICK_PHRASES: QuickPhrase[] = [
  { id: 'hello', label: 'Hello', text: 'hello', gloss: 'hello' },
  { id: 'thank-you', label: 'Thank you', text: 'thank you', gloss: 'thankyou' },
  { id: 'please', label: 'Please', text: 'please', gloss: 'please' },
  { id: 'yes', label: 'Yes', text: 'yes', gloss: 'yes' },
  { id: 'no', label: 'No', text: 'no', gloss: 'no' },
  { id: 'water', label: 'Water', text: 'water', gloss: 'water' },
  { id: 'potty', label: 'Potty', text: 'potty', gloss: 'potty' },
  { id: 'sick', label: 'Sick', text: 'sick', gloss: 'sick' },
  { id: 'help', label: 'Help', text: 'help', gloss: null },
  { id: 'sorry', label: 'Sorry', text: 'sorry', gloss: null },
  { id: 'name', label: 'Name', text: 'name', gloss: null },
  { id: 'doctor', label: 'Doctor', text: 'doctor', gloss: null },
  { id: 'bathroom', label: 'Bathroom', text: 'bathroom', gloss: null },
];

export function phraseInRecognizer(phrase: QuickPhrase): boolean {
  return phrase.gloss !== null;
}
