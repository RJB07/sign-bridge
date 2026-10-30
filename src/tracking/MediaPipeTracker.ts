import {
  FilesetResolver,
  HolisticLandmarker,
  type HolisticLandmarkerResult,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import { publicUrl } from '../publicUrl';
import { LIP_INDEXES } from '../recognition/features';
import type { FrameLandmarksData, Point3D, TrackingStatus } from '../types/signbridge';

const HOLISTIC_MODEL = publicUrl('/models/holistic_landmarker.task');

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const POSE_CONNECTIONS: Array<[number, number]> = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
];

function lipPoints(list: NormalizedLandmark[] | undefined): Point3D[] | null {
  if (!list || list.length <= Math.max(...LIP_INDEXES)) return null;
  const points = LIP_INDEXES.map((index) => list[index]);
  if (points.some((point) => !point || !Number.isFinite(point.x) || !Number.isFinite(point.y))) return null;
  return points.map((point) => ({ x: point.x, y: point.y, z: point.z ?? 0 }));
}

function toPoints(list: NormalizedLandmark[] | undefined, expected: number): Point3D[] | null {
  if (!list || list.length < expected) return null;
  return list.slice(0, expected).map((point) => ({ x: point.x, y: point.y, z: point.z ?? 0 }));
}

export class MediaPipeTracker {
  private landmarker: HolisticLandmarker | null = null;
  private stream: MediaStream | null = null;
  private running = false;
  private frameCount = 0;
  private lastFpsTime = performance.now();
  private currentFps = 0;
  private lastDetect = 0;
  private raf = 0;

  constructor(
    private readonly video: HTMLVideoElement,
    private readonly canvas: HTMLCanvasElement,
    private readonly onFrame: (data: FrameLandmarksData) => void,
    private readonly onStatus: (status: TrackingStatus) => void,
  ) {}

  async start(): Promise<void> {
    if (this.running) return;
    this.running = true;
    this.canvas.width = 640;
    this.canvas.height = 480;
    try {
      const fileset = await FilesetResolver.forVisionTasks(publicUrl('/vendor/mediapipe'));
      this.landmarker = await HolisticLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: HOLISTIC_MODEL },
        runningMode: 'VIDEO',
        outputFaceBlendshapes: false,
        outputPoseSegmentationMasks: false,
        minHandLandmarksConfidence: 0.35,
        minPoseDetectionConfidence: 0.35,
        minFaceDetectionConfidence: 0.35,
      });
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 },
        audio: false,
      });
      this.video.srcObject = this.stream;
      this.video.muted = true;
      this.video.playsInline = true;
      await this.video.play();
      this.onStatus({
        cameraActive: true,
        permissionGranted: true,
        handsCount: 0,
        poseDetected: false,
        message: 'Camera on. ASL gloss model runs on this device.',
        errorCode: 'NONE',
      });
      this.loop();
    } catch (error) {
      this.running = false;
      const denied = error instanceof DOMException && (error.name === 'NotAllowedError' || error.name === 'NotFoundError');
      this.onStatus({
        cameraActive: false,
        permissionGranted: !denied,
        handsCount: 0,
        poseDetected: false,
        message: denied
          ? 'Camera permission was denied or no camera is available. Held-out replay still works.'
          : `Tracking failed to start: ${error}`,
        errorCode: denied ? 'PERMISSION_DENIED' : 'LIBRARY_LOAD_ERROR',
      });
    }
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.landmarker?.close();
    this.landmarker = null;
  }

  private loop = (): void => {
    if (!this.running) return;
    this.raf = requestAnimationFrame(this.loop);
    const now = performance.now();
    if (!this.landmarker || this.video.readyState < 2 || now - this.lastDetect < 66) return;
    this.lastDetect = now;
    let result: HolisticLandmarkerResult;
    try {
      result = this.landmarker.detectForVideo(this.video, now);
    } catch {
      return;
    }
    this.paint(result);
    const leftHand = toPoints(result.leftHandLandmarks?.[0], 21);
    const rightHand = toPoints(result.rightHandLandmarks?.[0], 21);
    const pose = toPoints(result.poseLandmarks?.[0], 33);
    const lips = lipPoints(result.faceLandmarks?.[0]);
    const handsCount = Number(Boolean(leftHand)) + Number(Boolean(rightHand));
    this.frameCount += 1;
    if (now - this.lastFpsTime >= 1000) {
      this.currentFps = Math.round((this.frameCount * 1000) / (now - this.lastFpsTime));
      this.frameCount = 0;
      this.lastFpsTime = now;
    }
    this.onStatus({
      cameraActive: true,
      permissionGranted: true,
      handsCount,
      poseDetected: Boolean(pose),
      message: handsCount
        ? `${handsCount} hand(s) tracked`
        : 'Waiting for a hand in frame…',
      errorCode: 'NONE',
    });
    this.onFrame({
      leftHand,
      rightHand,
      pose,
      lips,
      faceDetected: Boolean(result.faceLandmarks?.[0]?.length),
      handsCount,
      timestamp: now,
      fps: this.currentFps,
    });
  };

  private paint(result: HolisticLandmarkerResult): void {
    const ctx = this.canvas.getContext('2d');
    if (!ctx) return;
    const width = this.canvas.width;
    const height = this.canvas.height;
    ctx.clearRect(0, 0, width, height);
    const pose = result.poseLandmarks?.[0];
    if (pose) this.stroke(ctx, pose, POSE_CONNECTIONS, 'rgba(255,255,255,0.85)', 3);
    this.drawHand(ctx, result.leftHandLandmarks?.[0], 'L');
    this.drawHand(ctx, result.rightHandLandmarks?.[0], 'R');
  }

  private stroke(
    ctx: CanvasRenderingContext2D,
    points: NormalizedLandmark[],
    links: Array<[number, number]>,
    color: string,
    width: number,
  ): void {
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    for (const [a, b] of links) {
      const start = points[a];
      const end = points[b];
      if (!start || !end) continue;
      ctx.beginPath();
      ctx.moveTo(start.x * w, start.y * h);
      ctx.lineTo(end.x * w, end.y * h);
      ctx.stroke();
    }
  }

  private drawHand(ctx: CanvasRenderingContext2D, points: NormalizedLandmark[] | undefined, label: string): void {
    if (!points || points.length < 21) return;
    this.stroke(ctx, points, HAND_CONNECTIONS, '#3b82f6', 3);
    const w = this.canvas.width;
    const h = this.canvas.height;
    const tips: Record<number, string> = { 4: 'T1', 8: 'I2', 12: 'M3', 16: 'R4', 20: 'P5' };
    points.forEach((point, index) => {
      const x = point.x * w;
      const y = point.y * h;
      ctx.beginPath();
      ctx.fillStyle = '#facc15';
      ctx.arc(x, y, tips[index] ? 6 : 3.5, 0, Math.PI * 2);
      ctx.fill();
      if (tips[index]) {
        ctx.fillStyle = '#fff';
        ctx.font = '700 12px ui-monospace, monospace';
        ctx.fillText(tips[index], x + 8, y - 6);
      }
    });
    ctx.fillStyle = '#fff';
    ctx.font = '700 13px ui-monospace, monospace';
    ctx.fillText(label, points[0].x * w + 10, points[0].y * h - 8);
  }
}
