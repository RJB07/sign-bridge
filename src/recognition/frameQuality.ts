import type { Point3D } from '../types/signbridge';

export type FrameHint = {
  level: 'ok' | 'warn';
  message: string;
};

const TIPS = [4, 8, 12, 16, 20];

function palmSize(hand: Point3D[]): number {
  const wrist = hand[0];
  const middle = hand[9];
  return Math.hypot(wrist.x - middle.x, wrist.y - middle.y);
}

function outOfFrame(hand: Point3D[]): boolean {
  const watched = [0, ...TIPS];
  return watched.some((index) => {
    const point = hand[index];
    return point.x < 0.02 || point.x > 0.98 || point.y < 0.02 || point.y > 0.98;
  });
}

/** Coaching for distance, cropping, and light. A closed fist is not treated as a partial hand. */
export function assessFrame(input: {
  leftHand: Point3D[] | null;
  rightHand: Point3D[] | null;
  poseDetected: boolean;
}): FrameHint {
  const hands = [input.leftHand, input.rightHand].filter((hand): hand is Point3D[] => Boolean(hand && hand.length >= 21));
  if (hands.length === 0) {
    return input.poseDetected
      ? { level: 'warn', message: 'Face is in frame, hands are not. Raise a hand into the light.' }
      : { level: 'warn', message: 'Step into view and face a light source so the camera can see you.' };
  }
  if (hands.some(outOfFrame)) {
    return { level: 'warn', message: 'Keep the whole hand inside the frame, including the fingertips.' };
  }
  const size = Math.max(...hands.map(palmSize));
  if (size < 0.05) {
    return { level: 'warn', message: 'Move closer. The hand is too small for a stable sign.' };
  }
  if (size > 0.42) {
    return { level: 'warn', message: 'Move back a little so the full hand stays in view.' };
  }
  const which = input.leftHand && input.rightHand ? 'Both hands' : input.leftHand ? 'Left hand' : 'Right hand';
  return { level: 'ok', message: `${which} in frame. Sign one word, then pause.` };
}
