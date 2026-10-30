/**
 * Landmark sequence features. Keep this aligned with training/features.py.
 * Dominant hand is mirrored into a right-hand frame, then wrist-centered and
 * scaled by shoulder width (image x/y) or palm size.
 */

export const FEATURE_DIM = 273;
/** Face-relative hands and pose, wrist-centered shape, and velocities. Keep aligned with training/features.py. */
export const RICH_FEATURE_DIM = 294;
export const SEQ_LEN = 32;
export const POSE_INDEXES = [0, 11, 12, 13, 14, 15, 16];

export type Landmark = { x: number; y: number; z: number };

export type FeatureFrame = {
  left: Landmark[] | null;
  right: Landmark[] | null;
  pose: Landmark[] | null;
};

export type HandSide = 'Left' | 'Right';

function xyDist(a: Landmark, b: Landmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function asPoints(value: Landmark[] | null | undefined, expected: number): Landmark[] | null {
  if (!value || value.length < expected) return null;
  for (let index = 0; index < expected; index += 1) {
    const point = value[index];
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) return null;
  }
  return value;
}

export function chooseDominant(frames: FeatureFrame[], forced?: HandSide | null): HandSide | null {
  if (forced === 'Left' || forced === 'Right') {
    const key = forced === 'Left' ? 'left' : 'right';
    if (frames.some((frame) => asPoints(frame[key], 21))) return forced;
  }

  const motion = { Left: 0, Right: 0 };
  const count = { Left: 0, Right: 0 };
  const prev: { Left: Landmark | null; Right: Landmark | null } = { Left: null, Right: null };
  for (const frame of frames) {
    const hands: Array<[HandSide, Landmark[] | null]> = [
      ['Left', asPoints(frame.left, 21)],
      ['Right', asPoints(frame.right, 21)],
    ];
    for (const [side, hand] of hands) {
      if (!hand) continue;
      count[side] += 1;
      if (prev[side]) motion[side] += xyDist(hand[0], prev[side]);
      prev[side] = hand[0];
    }
  }
  if (count.Left === 0 && count.Right === 0) return null;
  if (motion.Left > motion.Right + 1e-6) return 'Left';
  if (motion.Right > motion.Left + 1e-6) return 'Right';
  return count.Left >= count.Right ? 'Left' : 'Right';
}

function faceOriginAndScale(hand: Landmark[] | null, pose: Landmark[] | null): { origin: Landmark; scale: number } {
  let scale = 0;
  if (pose) scale = xyDist(pose[11], pose[12]);
  if (scale < 1e-4 && hand) scale = xyDist(hand[0], hand[9]);
  if (scale < 1e-4) scale = 1;
  let origin: Landmark;
  if (pose) origin = { ...pose[0] };
  else if (hand) origin = { ...hand[0] };
  else origin = { x: 0, y: 0, z: 0 };
  return { origin, scale };
}

function originAndScale(hand: Landmark[] | null, pose: Landmark[] | null): { origin: Landmark; scale: number } {
  let origin: Landmark;
  if (hand) origin = { ...hand[0] };
  else if (pose) {
    origin = {
      x: (pose[11].x + pose[12].x) / 2,
      y: (pose[11].y + pose[12].y) / 2,
      z: (pose[11].z + pose[12].z) / 2,
    };
  } else origin = { x: 0, y: 0, z: 0 };

  let scale = 0;
  if (pose) scale = xyDist(pose[11], pose[12]);
  if (scale < 1e-4 && hand) scale = xyDist(hand[0], hand[9]);
  if (scale < 1e-4) scale = 1;
  return { origin, scale };
}

function normalize(points: Landmark[], origin: Landmark, scale: number, mirror: boolean): number[] {
  const out: number[] = [];
  for (const point of points) {
    let x = (point.x - origin.x) / scale;
    const y = (point.y - origin.y) / scale;
    const z = (point.z - origin.z) / scale;
    if (mirror) x = -x;
    out.push(x, y, z);
  }
  return out;
}

function pack(points: Landmark[] | null, origin: Landmark, scale: number, mirror: boolean, width: number): number[] {
  if (!points) return new Array(width).fill(0);
  const values = normalize(points, origin, scale, mirror);
  if (values.length === width) return values;
  const out = new Array(width).fill(0);
  for (let i = 0; i < Math.min(width, values.length); i++) out[i] = values[i];
  return out;
}

export function sequenceFeatures(frames: FeatureFrame[], forced?: HandSide | null): number[][] {
  const dominant = chooseDominant(frames, forced);
  if (!dominant) return [];
  const mirror = dominant === 'Left';
  const domKey = dominant === 'Left' ? 'left' : 'right';
  const otherKey = dominant === 'Left' ? 'right' : 'left';
  const rows: number[][] = [];
  let prevDom: number[] | null = null;
  let prevOther: number[] | null = null;
  for (const frame of frames) {
    const hand = asPoints(frame[domKey], 21);
    const other = asPoints(frame[otherKey], 21);
    const pose = asPoints(frame.pose, 33);
    const { origin, scale } = originAndScale(hand, pose);
    const dom = pack(hand, origin, scale, mirror, 63);
    const nondom = pack(other, origin, scale, mirror, 63);
    const deltaDom = prevDom ? dom.map((value, index) => value - prevDom![index]) : new Array(63).fill(0);
    const deltaOther = prevOther ? nondom.map((value, index) => value - prevOther![index]) : new Array(63).fill(0);
    prevDom = dom;
    prevOther = nondom;
    const posePoints = pose ? POSE_INDEXES.map((index) => pose[index]) : null;
    const poseFeat = pack(posePoints, origin, scale, mirror, 21);
    rows.push([...dom, ...nondom, ...poseFeat, ...deltaDom, ...deltaOther]);
  }
  return rows;
}

export function sequenceFeaturesRich(frames: FeatureFrame[], forced?: HandSide | null): number[][] {
  const dominant = chooseDominant(frames, forced);
  if (!dominant) return [];
  const mirror = dominant === 'Left';
  const domKey = dominant === 'Left' ? 'left' : 'right';
  const otherKey = dominant === 'Left' ? 'right' : 'left';
  const rows: number[][] = [];
  let prevDom: number[] | null = null;
  let prevPose: number[] | null = null;
  for (const frame of frames) {
    const hand = asPoints(frame[domKey], 21);
    const other = asPoints(frame[otherKey], 21);
    const pose = asPoints(frame.pose, 33);
    const { origin, scale } = faceOriginAndScale(hand, pose);
    const domFace = pack(hand, origin, scale, mirror, 63);
    const otherFace = pack(other, origin, scale, mirror, 63);
    const posePoints = pose ? POSE_INDEXES.map((index) => pose[index]) : null;
    const poseFeat = pack(posePoints, origin, scale, mirror, 21);
    const shape = hand ? pack(hand, { ...hand[0] }, scale, mirror, 63) : new Array(63).fill(0);
    const velDom = prevDom ? domFace.map((value, index) => value - prevDom![index]) : new Array(63).fill(0);
    const velPose = prevPose ? poseFeat.map((value, index) => value - prevPose![index]) : new Array(21).fill(0);
    prevDom = domFace;
    prevPose = poseFeat;
    rows.push([...domFace, ...otherFace, ...poseFeat, ...shape, ...velDom, ...velPose]);
  }
  return rows;
}

export function mirrorSequence(sequence: number[][]): number[][] {
  return sequence.map((row) => row.map((value, index) => (index % 3 === 0 ? -value : value)));
}

export function resampleSequence(seq: number[][], length = SEQ_LEN): number[][] {
  if (seq.length === 0) return [];
  if (seq.length === length) return seq.map((row) => row.slice());
  if (seq.length === 1) return new Array(length).fill(0).map(() => seq[0].slice());
  const dims = seq[0].length;
  const out: number[][] = [];
  for (let i = 0; i < length; i++) {
    const dst = length === 1 ? 0 : i / (length - 1);
    const srcPos = dst * (seq.length - 1);
    const left = Math.floor(srcPos);
    const right = Math.min(seq.length - 1, left + 1);
    const mix = srcPos - left;
    const row = new Array(dims);
    for (let d = 0; d < dims; d++) {
      row[d] = seq[left][d] * (1 - mix) + seq[right][d] * mix;
    }
    out.push(row);
  }
  return out;
}
