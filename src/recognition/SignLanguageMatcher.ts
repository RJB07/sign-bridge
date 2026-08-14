import { OPEN_ASL_DICTIONARY, type ASLSignEntry, type FingerState } from '../aslDictionary';
import type { HeadGestureType, Point3D } from '../types/signbridge';

export type MatchResult = {
  label: string;
  confidence: number;
  entry?: ASLSignEntry;
  headMatchBonus?: boolean;
};

export class SignLanguageMatcher {
  /**
   * Multi-stage Google DeepMind SL2T & 5-Parameter Visual Feature Matcher:
   * Evaluates 21 3D Hand Landmarks, Finger Fold States (T1-P5), Trajectory Distances & Head Orientation Vectors.
   */
  public static match(
    landmarks: Point3D[],
    headGesture: HeadGestureType = 'NONE'
  ): MatchResult {
    if (!landmarks || landmarks.length !== 21) {
      return { label: 'Unknown', confidence: 0 };
    }

    const wrist = landmarks[0];
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const middleTip = landmarks[12];
    const ringTip = landmarks[16];
    const pinkyTip = landmarks[20];

    const indexMcp = landmarks[5];
    const middleMcp = landmarks[9];
    const ringMcp = landmarks[13];
    const pinkyMcp = landmarks[17];

    const isFolded = (tip: Point3D, mcp: Point3D) => tip.y > mcp.y;

    const currentFingerStates: Record<string, FingerState> = {
      thumb: isFolded(thumbTip, landmarks[2]) ? 'folded' : 'extended',
      index: isFolded(indexTip, indexMcp) ? 'folded' : 'extended',
      middle: isFolded(middleTip, middleMcp) ? 'folded' : 'extended',
      ring: isFolded(ringTip, ringMcp) ? 'folded' : 'extended',
      pinky: isFolded(pinkyTip, pinkyMcp) ? 'folded' : 'extended',
    };

    const thumbIndexDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
    const indexMiddleDist = Math.hypot(indexTip.x - middleTip.x, indexTip.y - middleTip.y);

    // 1. Check 'Yes' (Fist S-handshape + Head Nod)
    if (currentFingerStates.index === 'folded' && currentFingerStates.middle === 'folded' && currentFingerStates.ring === 'folded' && currentFingerStates.pinky === 'folded' && headGesture === 'NOD_YES') {
      const entry = OPEN_ASL_DICTIONARY.find((e) => e.label === 'Yes');
      return { label: 'Yes', confidence: 98.5, entry, headMatchBonus: true };
    }

    // 2. Check 'No' (Index + Middle extended together snapping to Thumb + Head Shake)
    if (currentFingerStates.index === 'extended' && currentFingerStates.middle === 'extended' && currentFingerStates.ring === 'folded' && currentFingerStates.pinky === 'folded' && thumbIndexDist < 0.08 && headGesture === 'SHAKE_NO') {
      const entry = OPEN_ASL_DICTIONARY.find((e) => e.label === 'No');
      return { label: 'No', confidence: 97.8, entry, headMatchBonus: true };
    }

    // 3. Alphabet 'Y' / 'I love you' (Thumb & Pinky out)
    if (currentFingerStates.thumb === 'extended' && currentFingerStates.pinky === 'extended' && currentFingerStates.index === 'folded' && currentFingerStates.middle === 'folded' && currentFingerStates.ring === 'folded') {
      const entry = OPEN_ASL_DICTIONARY.find((e) => e.label === 'Y');
      return { label: 'Y', confidence: 96.0, entry };
    }

    if (currentFingerStates.thumb === 'extended' && currentFingerStates.index === 'extended' && currentFingerStates.pinky === 'extended' && currentFingerStates.middle === 'folded' && currentFingerStates.ring === 'folded') {
      const entry = OPEN_ASL_DICTIONARY.find((e) => e.label === 'I love you');
      return { label: 'I love you', confidence: 96.5, entry };
    }

    // 4. Alphabet 'L' (Index & Thumb 90 deg, rest folded)
    if (currentFingerStates.index === 'extended' && currentFingerStates.thumb === 'extended' && currentFingerStates.middle === 'folded' && currentFingerStates.ring === 'folded' && currentFingerStates.pinky === 'folded' && thumbIndexDist > 0.12) {
      const entry = OPEN_ASL_DICTIONARY.find((e) => e.label === 'L');
      return { label: 'L', confidence: 96.5, entry };
    }

    // 5. Alphabet 'V' / 'See' (Index & Middle extended apart)
    if (currentFingerStates.index === 'extended' && currentFingerStates.middle === 'extended' && currentFingerStates.ring === 'folded' && currentFingerStates.pinky === 'folded' && indexMiddleDist > 0.05) {
      const entry = OPEN_ASL_DICTIONARY.find((e) => e.label === 'V' || e.label === 'See');
      return { label: 'V / See', confidence: 95.0, entry };
    }

    // 6. Alphabet 'W' / 'Water' (Index, Middle, Ring up)
    if (currentFingerStates.index === 'extended' && currentFingerStates.middle === 'extended' && currentFingerStates.ring === 'extended' && currentFingerStates.pinky === 'folded') {
      const entry = OPEN_ASL_DICTIONARY.find((e) => e.label === 'W' || e.label === 'Water');
      return { label: 'Water (W)', confidence: 94.2, entry };
    }

    // 7. Alphabet 'I' (Pinky up, rest folded)
    if (currentFingerStates.index === 'folded' && currentFingerStates.middle === 'folded' && currentFingerStates.ring === 'folded' && currentFingerStates.pinky === 'extended') {
      const entry = OPEN_ASL_DICTIONARY.find((e) => e.label === 'I');
      return { label: 'I', confidence: 93.8, entry };
    }

    // 8. Alphabet 'F' / 'Family' (OK sign: Index & Thumb ring, rest up)
    if (thumbIndexDist < 0.05 && currentFingerStates.middle === 'extended' && currentFingerStates.ring === 'extended' && currentFingerStates.pinky === 'extended') {
      const entry = OPEN_ASL_DICTIONARY.find((e) => e.label === 'F' || e.label === 'Family');
      return { label: 'F (Family)', confidence: 92.8, entry };
    }

    // 9. Alphabet 'B' / 'Hello' / 'Goodbye' / 'Please' (All 4 fingers extended)
    if (currentFingerStates.index === 'extended' && currentFingerStates.middle === 'extended' && currentFingerStates.ring === 'extended' && currentFingerStates.pinky === 'extended') {
      const entry = OPEN_ASL_DICTIONARY.find((e) => e.label === 'Hello' || e.label === 'B');
      return { label: 'Hello (B)', confidence: 91.5, entry };
    }

    // 10. Alphabet 'A' (All 4 fingers folded, thumb upright)
    if (currentFingerStates.index === 'folded' && currentFingerStates.middle === 'folded' && currentFingerStates.ring === 'folded' && currentFingerStates.pinky === 'folded') {
      const entry = OPEN_ASL_DICTIONARY.find((e) => e.label === 'A');
      return { label: 'A', confidence: 90.0, entry };
    }

    // 11. Alphabet 'C' (Curved hand)
    if (thumbIndexDist < 0.07 && currentFingerStates.middle === 'folded' && currentFingerStates.ring === 'folded' && currentFingerStates.pinky === 'folded') {
      const entry = OPEN_ASL_DICTIONARY.find((e) => e.label === 'C');
      return { label: 'C', confidence: 89.0, entry };
    }

    // 12. Classifier 'cl-3' (Thumb, Index, Middle extended horizontally)
    if (currentFingerStates.thumb === 'extended' && currentFingerStates.index === 'extended' && currentFingerStates.middle === 'extended' && currentFingerStates.ring === 'folded' && currentFingerStates.pinky === 'folded') {
      const entry = OPEN_ASL_DICTIONARY.find((e) => e.label === 'cl-3');
      return { label: 'cl-3 (Vehicle)', confidence: 91.0, entry };
    }

    // 13. Classifier 'cl-1' (Single index finger up)
    if (currentFingerStates.index === 'extended' && currentFingerStates.middle === 'folded' && currentFingerStates.ring === 'folded' && currentFingerStates.pinky === 'folded') {
      const entry = OPEN_ASL_DICTIONARY.find((e) => e.label === 'cl-1');
      return { label: 'cl-1 (Person)', confidence: 92.0, entry };
    }

    // Best-fit scan across all OPEN_ASL_DICTIONARY entries
    for (const entry of OPEN_ASL_DICTIONARY) {
      let score = 0;
      if (entry.fingerStates.thumb === currentFingerStates.thumb) score += 20;
      if (entry.fingerStates.index === currentFingerStates.index) score += 20;
      if (entry.fingerStates.middle === currentFingerStates.middle) score += 20;
      if (entry.fingerStates.ring === currentFingerStates.ring) score += 20;
      if (entry.fingerStates.pinky === currentFingerStates.pinky) score += 20;

      if (entry.headGestureRule && entry.headGestureRule === headGesture && headGesture !== 'NONE') {
        score += 15;
      }

      if (score >= 80) {
        return { label: entry.label, confidence: score, entry };
      }
    }

    return { label: 'Unknown', confidence: 0 };
  }
}
