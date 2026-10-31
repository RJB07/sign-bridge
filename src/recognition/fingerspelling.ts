import type { Landmark } from './features';

/** Static ASL handshapes. J and Z need a drawn path and are not claimed here. */
const SUPPORTED = ['A', 'B', 'C', 'F', 'I', 'L', 'V', 'W', 'Y'] as const;
export type FingerspelledLetter = (typeof SUPPORTED)[number];

function dist(a: Landmark, b: Landmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function extension(landmarks: Landmark[], tip: number, mcp: number): number {
  return dist(landmarks[tip], landmarks[mcp]) / (dist(landmarks[mcp], landmarks[0]) + 1e-6);
}

export function classifyFingerspelling(landmarks: Landmark[] | null): FingerspelledLetter | null {
  if (!landmarks || landmarks.length < 21) return null;
  const index = extension(landmarks, 8, 5);
  const middle = extension(landmarks, 12, 9);
  const ring = extension(landmarks, 16, 13);
  const pinky = extension(landmarks, 20, 17);
  const palm = dist(landmarks[0], landmarks[9]) + 1e-6;
  const thumbOut = dist(landmarks[4], landmarks[5]) / palm;
  const thumbIndex = dist(landmarks[4], landmarks[8]) / palm;
  const indexMiddle = dist(landmarks[8], landmarks[12]) / palm;

  const up = (value: number) => value >= 0.9;
  const down = (value: number) => value <= 0.55;
  const fourDown = down(index) && down(middle) && down(ring) && down(pinky);
  const fourUp = up(index) && up(middle) && up(ring) && up(pinky);

  if (thumbIndex < 0.45 && up(middle) && up(ring) && up(pinky) && !up(index)) return 'F';
  if (up(index) && up(middle) && up(ring) && down(pinky)) return 'W';
  if (up(index) && up(middle) && down(ring) && down(pinky) && indexMiddle > 0.35) return 'V';
  if (up(index) && down(middle) && down(ring) && down(pinky) && thumbOut > 0.85) return 'L';
  if (down(index) && down(middle) && down(ring) && up(pinky) && thumbOut > 0.85) return 'Y';
  if (down(index) && down(middle) && down(ring) && up(pinky)) return 'I';
  if (fourUp && thumbOut < 0.75) return 'B';
  if (fourDown && thumbOut < 0.8) return 'A';
  const curved = [index, middle, ring, pinky].every((value) => value > 0.55 && value < 0.95);
  if (curved && thumbIndex > 0.35 && thumbIndex < 0.9) return 'C';
  return null;
}

export function supportedLetters(): readonly FingerspelledLetter[] {
  return SUPPORTED;
}
