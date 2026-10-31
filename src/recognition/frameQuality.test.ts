import { describe, expect, it } from 'vitest';
import { assessFrame } from './frameQuality';
import { mirrorSequence } from './features';
import type { Point3D } from '../types/signbridge';

function handAt(x: number, scale = 0.12): Point3D[] {
  return Array.from({ length: 21 }, (_, index) => ({
    x: x + (index === 9 ? scale : 0),
    y: 0.4,
    z: 0,
  }));
}

describe('webcam coaching', () => {
  it('asks for light when nobody is visible', () => {
    expect(assessFrame({ leftHand: null, rightHand: null, poseDetected: false }).message).toMatch(/light/i);
  });

  it('asks for a hand when only the face is tracked', () => {
    expect(assessFrame({ leftHand: null, rightHand: null, poseDetected: true }).message).toMatch(/hand/i);
  });

  it('asks the signer to move closer when the palm is tiny', () => {
    expect(assessFrame({ leftHand: null, rightHand: handAt(0.5, 0.02), poseDetected: true }).message).toMatch(/closer/i);
  });

  it('asks for the fingertips to stay inside the frame', () => {
    const cropped = handAt(0.5);
    cropped[8] = { x: 1.2, y: 0.4, z: 0 };
    expect(assessFrame({ leftHand: cropped, rightHand: null, poseDetected: true }).message).toMatch(/frame/i);
  });

  it('accepts a left hand that is fully visible', () => {
    const hint = assessFrame({ leftHand: handAt(0.4), rightHand: null, poseDetected: true });
    expect(hint.level).toBe('ok');
    expect(hint.message).toMatch(/Left hand/);
  });
});

describe('mirror augmentation', () => {
  it('flips only x and is its own inverse', () => {
    const seq = [[1, 2, 3, -4, 5, 6]];
    const flipped = mirrorSequence(seq);
    expect(flipped[0]).toEqual([-1, 2, 3, 4, 5, 6]);
    expect(mirrorSequence(flipped)).toEqual(seq);
  });
});
