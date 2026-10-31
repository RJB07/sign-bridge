/** Everyday words a fingerspelled sequence is allowed to snap to. */
const DICTIONARY = [
  'apple',
  'bathroom',
  'book',
  'cat',
  'dad',
  'deaf',
  'doctor',
  'dog',
  'eat',
  'friend',
  'hello',
  'help',
  'home',
  'hospital',
  'love',
  'milk',
  'mom',
  'more',
  'name',
  'nurse',
  'pain',
  'phone',
  'please',
  'sorry',
  'stop',
  'wait',
  'water',
  'yes',
];

function damerau(left: string, right: string): number {
  const rows = left.length + 1;
  const cols = right.length + 1;
  const score: number[][] = Array.from({ length: rows }, () => new Array(cols).fill(0));
  for (let i = 0; i < rows; i += 1) score[i][0] = i;
  for (let j = 0; j < cols; j += 1) score[0][j] = j;
  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < cols; j += 1) {
      const cost = left[i - 1] === right[j - 1] ? 0 : 1;
      score[i][j] = Math.min(score[i - 1][j] + 1, score[i][j - 1] + 1, score[i - 1][j - 1] + cost);
      if (i > 1 && j > 1 && left[i - 1] === right[j - 2] && left[i - 2] === right[j - 1]) {
        score[i][j] = Math.min(score[i][j], score[i - 2][j - 2] + 1);
      }
    }
  }
  return score[left.length][right.length];
}

/**
 * Snap a finished fingerspelled string to one dictionary word.
 * An exact letter sequence wins. A single close neighbor wins.
 * Two equally close neighbors, or no neighbor, leave the spelling unchanged.
 */
export function correctSpelling(raw: string): string {
  const letters = raw.trim();
  if (letters.length < 3) return letters;
  const key = letters.toLowerCase();
  const exact = DICTIONARY.find((word) => word === key);
  if (exact) return exact;
  const limit = key.length >= 6 ? 2 : 1;
  let best: string[] = [];
  let bestDistance = limit + 1;
  for (const word of DICTIONARY) {
    if (Math.abs(word.length - key.length) > limit) continue;
    const distance = damerau(key, word);
    if (distance < bestDistance && distance <= limit) {
      bestDistance = distance;
      best = [word];
    } else if (distance === bestDistance && distance <= limit) {
      best.push(word);
    }
  }
  return best.length === 1 ? best[0] : letters;
}
