export function predictSignHeuristic(
  landmarks: Array<{ x: number; y: number; z: number }>
): string {
  if (!landmarks || landmarks.length !== 21) {
    return 'Unknown';
  }

  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const middleTip = landmarks[12];
  const ringTip = landmarks[16];
  const pinkyTip = landmarks[20];
  const indexMcp = landmarks[5];
  const middleMcp = landmarks[9];
  const ringMcp = landmarks[13];
  const pinkyMcp = landmarks[17];

  const fingerFolded = (
    tip: { x: number; y: number },
    mcp: { x: number; y: number }
  ) => tip.y > mcp.y;

  const allFingersFolded =
    fingerFolded(indexTip, indexMcp) &&
    fingerFolded(middleTip, middleMcp) &&
    fingerFolded(ringTip, ringMcp) &&
    fingerFolded(pinkyTip, pinkyMcp);

  const allFingersExtended =
    !fingerFolded(indexTip, indexMcp) &&
    !fingerFolded(middleTip, middleMcp) &&
    !fingerFolded(ringTip, ringMcp) &&
    !fingerFolded(pinkyTip, pinkyMcp);

  const thumbIndexDistance = Math.hypot(
    thumbTip.x - indexTip.x,
    thumbTip.y - indexTip.y
  );

  if (allFingersFolded) {
    return 'A';
  }

  if (allFingersExtended && thumbIndexDistance > 0.08) {
    return 'B';
  }

  if (
    thumbIndexDistance < 0.06 &&
    fingerFolded(middleTip, middleMcp) &&
    fingerFolded(ringTip, ringMcp) &&
    fingerFolded(pinkyTip, pinkyMcp)
  ) {
    return 'C';
  }

  return 'Unknown';
}
