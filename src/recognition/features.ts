/**
 * Landmark sequence features. Keep this aligned with training/features.py.
 * Dominant hand is mirrored into a right-hand frame, then wrist-centered and
 * scaled by shoulder width (image x/y) or palm size.
 */

export const FEATURE_DIM = 273;
/** Face-relative hands and pose, wrist-centered shape, and velocities. Keep aligned with training/features.py. */
export const RICH_FEATURE_DIM = 294;
/** Rich features, hand-to-face offsets, and five fingertip curls. Keep aligned with training/features.py. */
export const FINE_FEATURE_DIM = 389;
export const FINE_XYZ = 384;
export const LIP_INDEXES = [13, 14, 17, 61, 78, 81, 84, 87, 91, 146, 291, 308];
const FINE_HAND_INDEXES = [0, 4, 8, 12, 16, 20];
const FINE_CURLS: Array<[number, number, number]> = [[1, 2, 4], [5, 6, 8], [9, 10, 12], [13, 14, 16], [17, 18, 20]];
export const SEQ_LEN = 32;
export const POSE_INDEXES = [0, 11, 12, 13, 14, 15, 16];

export type Landmark = { x: number; y: number; z: number };

export type FeatureFrame = {
  left: Landmark[] | null;
  right: Landmark[] | null;
  pose: Landmark[] | null;
  lips?: Landmark[] | null;
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

function unitDot(a: Landmark, b: Landmark, c: Landmark): number {
  const toward = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
  const away = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const left = Math.hypot(toward.x, toward.y, toward.z);
  const right = Math.hypot(away.x, away.y, away.z);
  if (left < 1e-6 || right < 1e-6) return 0;
  return (toward.x * away.x + toward.y * away.y + toward.z * away.z) / (left * right);
}

function placePoint(point: Landmark, origin: Landmark, scale: number, mirror: boolean): Landmark {
  return {
    x: ((point.x - origin.x) / scale) * (mirror ? -1 : 1),
    y: (point.y - origin.y) / scale,
    z: (point.z - origin.z) / scale,
  };
}

export function sequenceFeaturesFine(frames: FeatureFrame[], forced?: HandSide | null): number[][] {
  const rich = sequenceFeaturesRich(frames, forced);
  if (rich.length === 0) return [];
  const dominant = chooseDominant(frames, forced);
  if (!dominant) return [];
  const mirror = dominant === 'Left';
  const domKey = dominant === 'Left' ? 'left' : 'right';
  return frames.map((frame, time) => {
    const hand = asPoints(frame[domKey], 21);
    const pose = asPoints(frame.pose, 33);
    const lips = asPoints(frame.lips ?? null, LIP_INDEXES.length);
    const { origin, scale } = faceOriginAndScale(hand, pose);
    const targets: Array<Landmark | null> = [];
    if (lips) {
      const mean = { x: 0, y: 0, z: 0 };
      for (const point of lips) {
        mean.x += point.x;
        mean.y += point.y;
        mean.z += point.z;
      }
      mean.x /= lips.length;
      mean.y /= lips.length;
      mean.z /= lips.length;
      targets.push(placePoint(mean, origin, scale, mirror));
    } else if (pose) {
      targets.push(placePoint({
        x: (pose[9].x + pose[10].x) / 2,
        y: (pose[9].y + pose[10].y) / 2,
        z: (pose[9].z + pose[10].z) / 2,
      }, origin, scale, mirror));
    } else targets.push(null);
    if (!pose) targets.push(null, null, null, null);
    else {
      targets.push(placePoint(pose[2], origin, scale, mirror));
      targets.push(placePoint(pose[5], origin, scale, mirror));
      targets.push(placePoint({
        x: (pose[3].x + pose[9].x) / 2,
        y: (pose[3].y + pose[9].y) / 2,
        z: (pose[3].z + pose[9].z) / 2,
      }, origin, scale, mirror));
      targets.push(placePoint({
        x: (pose[6].x + pose[10].x) / 2,
        y: (pose[6].y + pose[10].y) / 2,
        z: (pose[6].z + pose[10].z) / 2,
      }, origin, scale, mirror));
    }
    const offsets: number[] = [];
    if (!hand) {
      for (let i = 0; i < 90; i += 1) offsets.push(0);
    } else {
      for (const index of FINE_HAND_INDEXES) {
        const point = placePoint(hand[index], origin, scale, mirror);
        for (const target of targets) {
          if (!target) offsets.push(0, 0, 0);
          else offsets.push(point.x - target.x, point.y - target.y, point.z - target.z);
        }
      }
    }
    const curls = hand ? FINE_CURLS.map(([a, b, c]) => unitDot(hand[a], hand[b], hand[c])) : [0, 0, 0, 0, 0];
    return [...rich[time], ...offsets, ...curls];
  });
}

export function mirrorSequence(sequence: number[][], xyzDims?: number): number[][] {
  const width = xyzDims ?? (sequence[0]?.length === FINE_FEATURE_DIM ? FINE_XYZ : sequence[0]?.length ?? 0);
  return sequence.map((row) => row.map((value, index) => (index < width && index % 3 === 0 ? -value : value)));
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
