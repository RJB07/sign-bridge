import type { Results } from '@mediapipe/hands';
import type { FrameLandmarksData, HeadGestureType, Point3D, TrackingStatus } from '../types/signbridge';

export class MediaPipeTracker {
  private videoElement: HTMLVideoElement;
  private canvasElement: HTMLCanvasElement;
  private onFrameCallback: (data: FrameLandmarksData) => void;
  private onStatusCallback: (status: TrackingStatus) => void;

  private handsInstance: any = null;
  private faceMeshInstance: any = null;
  private poseInstance: any = null;
  private showFaceOverlay: boolean = true;
  private cameraInstance: any = null;

  private isRunning = false;
  private frameCount = 0;
  private lastFpsTime = performance.now();
  private currentFps = 0;

  private handResultsRef: Results | null = null;
  private faceResultsRef: any = null;
  private poseResultsRef: any = null;

  private landmarkSmoothing: Record<string, Point3D[]> = {};
  private faceSmoothing: Point3D[] = [];
  private poseSmoothing: Point3D[] = [];

  // Trajectory trail history (Three.js / RDTR01 inspired spatial particle trail)
  private trajectoryTrail: Point3D[] = [];
  private readonly maxTrailLength = 24;

  // Head Gesture Motion History
  private headYHistory: number[] = [];
  private headXHistory: number[] = [];
  private currentHeadGesture: HeadGestureType = 'NONE';

  // Aesthetic colors (Oberon & RDTR01 Three.js theme)
  private readonly lineColor = '#3b82f6';
  private readonly pointColor = '#facc15';
  private readonly poseColor = 'rgba(255, 255, 255, 0.75)';
  private readonly faceColor = 'rgba(37, 244, 238, 0.4)'; // Cyan face mesh
  private readonly smoothingFactor = 0.35;

  constructor(
    videoElement: HTMLVideoElement,
    canvasElement: HTMLCanvasElement,
    onFrameCallback: (data: FrameLandmarksData) => void,
    onStatusCallback: (status: TrackingStatus) => void,
    options?: { showFaceOverlay?: boolean }
  ) {
    this.videoElement = videoElement;
    this.canvasElement = canvasElement;
    this.onFrameCallback = onFrameCallback;
    this.onStatusCallback = onStatusCallback;
    if (options && typeof options.showFaceOverlay === 'boolean') {
      this.showFaceOverlay = options.showFaceOverlay;
    }
  }

  public async start(): Promise<void> {
    if (this.isRunning) return;

    try {
      this.videoElement.width = 640;
      this.videoElement.height = 480;
      this.canvasElement.width = 640;
      this.canvasElement.height = 480;

      const HandsClass = (window as any).Hands;
      const FaceMeshClass = (window as any).FaceMesh;
      const PoseClass = (window as any).Pose;
      const CameraClass = (window as any).Camera;

      if (!HandsClass || !CameraClass) {
        this.onStatusCallback({
          cameraActive: false,
          permissionGranted: false,
          handsCount: 0,
          faceDetected: false,
          headGesture: 'NONE',
          isPrivacyMode: true,
          signerHandedness: 'AUTO',
          isOneHandedMode: false,
          message: 'MediaPipe Vision libraries missing. Check CDN dependencies.',
          errorCode: 'LIBRARY_LOAD_ERROR',
        });
        return;
      }

      // Initialize MediaPipe Hands
      this.handsInstance = new HandsClass({
        locateFile: (file: string) => {
          const name = file.split('/').pop()?.split('?')[0] || file;
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${name}`;
        },
      });

      this.handsInstance.setOptions({
        selfieMode: false,
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.75,
        minTrackingConfidence: 0.7,
      });

      this.handsInstance.onResults((results: Results) => {
        this.handResultsRef = results;
        this.processCombinedFrame();
      });

      // Initialize MediaPipe FaceMesh
      if (FaceMeshClass) {
        this.faceMeshInstance = new FaceMeshClass({
          locateFile: (file: string) => {
            const name = file.split('/').pop()?.split('?')[0] || file;
            return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${name}`;
          },
        });

        this.faceMeshInstance.setOptions({
          selfieMode: false,
          maxNumFaces: 1,
          refineLandmarks: true,
          minDetectionConfidence: 0.75,
          minTrackingConfidence: 0.7,
        });

        this.faceMeshInstance.onResults((results: any) => {
          this.faceResultsRef = results;
          this.processCombinedFrame();
        });
      }

      // Initialize MediaPipe Pose (33 upper body landmarks)
      if (PoseClass) {
        this.poseInstance = new PoseClass({
          locateFile: (file: string) => {
            const name = file.split('/').pop()?.split('?')[0] || file;
            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${name}`;
          },
        });

        this.poseInstance.setOptions({
          selfieMode: false,
          modelComplexity: 1,
          smoothLandmarks: true,
          minDetectionConfidence: 0.7,
          minTrackingConfidence: 0.7,
        });

        this.poseInstance.onResults((results: any) => {
          this.poseResultsRef = results;
          this.processCombinedFrame();
        });
      }

      // Camera Loop
      this.cameraInstance = new CameraClass(this.videoElement, {
        onFrame: async () => {
          if (!this.isRunning) return;
          try {
            await this.handsInstance.send({ image: this.videoElement });
          } catch {}
          if (this.faceMeshInstance?.send) {
            try {
              await this.faceMeshInstance.send({ image: this.videoElement });
            } catch {}
          }
          if (this.poseInstance?.send) {
            try {
              await this.poseInstance.send({ image: this.videoElement });
            } catch {}
          }
        },
        width: 640,
        height: 480,
      });

      await this.cameraInstance.start();
      this.isRunning = true;
      this.videoElement.muted = true;
      this.videoElement.playsInline = true;
      await this.videoElement.play().catch(() => {});

      this.onStatusCallback({
        cameraActive: true,
        permissionGranted: true,
        handsCount: 0,
        faceDetected: false,
        headGesture: 'NONE',
        isPrivacyMode: true,
        signerHandedness: 'AUTO',
        isOneHandedMode: false,
        message: 'Google DeepMind SL2T & MediaPipe Holistic 540+ tracking active.',
        errorCode: 'NONE',
      });
    } catch (error) {
      this.onStatusCallback({
        cameraActive: false,
        permissionGranted: false,
        handsCount: 0,
        faceDetected: false,
        headGesture: 'NONE',
        isPrivacyMode: true,
        signerHandedness: 'AUTO',
        isOneHandedMode: false,
        message: `Camera permission denied or device unavailable: ${error}`,
        errorCode: 'PERMISSION_DENIED',
      });
    }
  }

  public stop(): void {
    this.isRunning = false;
    this.cameraInstance?.stop();
    this.handsInstance?.close();
    this.faceMeshInstance?.close?.();
    this.poseInstance?.close?.();
  }

  private detectHeadGesture(faceLandmarks: Point3D[]): HeadGestureType {
    if (!faceLandmarks || faceLandmarks.length < 5) return 'NONE';

    const noseTip = faceLandmarks[1];
    this.headYHistory.push(noseTip.y);
    this.headXHistory.push(noseTip.x);

    if (this.headYHistory.length > 15) this.headYHistory.shift();
    if (this.headXHistory.length > 15) this.headXHistory.shift();

    if (this.headYHistory.length >= 10) {
      const maxY = Math.max(...this.headYHistory);
      const minY = Math.min(...this.headYHistory);
      const rangeY = maxY - minY;

      const maxX = Math.max(...this.headXHistory);
      const minX = Math.min(...this.headXHistory);
      const rangeX = maxX - minX;

      if (rangeY > 0.045 && rangeY > rangeX * 1.5) {
        return 'NOD_YES';
      }
      if (rangeX > 0.05 && rangeX > rangeY * 1.5) {
        return 'SHAKE_NO';
      }
    }

    return 'NONE';
  }

  private processCombinedFrame(): void {
    const ctx = this.canvasElement.getContext('2d');
    if (!ctx) return;

    this.frameCount++;
    const now = performance.now();
    if (now - this.lastFpsTime >= 1000) {
      this.currentFps = Math.round((this.frameCount * 1000) / (now - this.lastFpsTime));
      this.frameCount = 0;
      this.lastFpsTime = now;
    }

    ctx.save();
    ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
    ctx.drawImage(this.videoElement, 0, 0, this.canvasElement.width, this.canvasElement.height);

    // 1. Render Upper Body Pose Skeleton (Shoulders, Arms, Torso)
    const poseData = this.poseResultsRef?.poseLandmarks || null;
    let smoothedPose: Point3D[] | null = null;
    if (poseData) {
      smoothedPose = this.smoothPoints(this.poseSmoothing, poseData);
      this.poseSmoothing = smoothedPose;
      this.renderPoseSkeleton(ctx, smoothedPose);
    }

    // 2. Render Face Mesh 468 Landmarks
    const faceData = this.faceResultsRef?.multiFaceLandmarks?.[0] || null;
    let smoothedFace: Point3D[] | null = null;

    if (faceData) {
      smoothedFace = this.smoothPoints(this.faceSmoothing, faceData);
      this.faceSmoothing = smoothedFace;
      this.currentHeadGesture = this.detectHeadGesture(smoothedFace);
      if (this.showFaceOverlay) {
        this.renderFaceLandmarks(ctx, smoothedFace);
      }
    } else {
      this.currentHeadGesture = 'NONE';
    }

    // 3. Render 3D Hand Landmarks & Trajectory Particles
    const handsData = this.handResultsRef?.multiHandLandmarks || [];
    const handednessData = this.handResultsRef?.multiHandedness || [];
    const processedHands: Point3D[][] = [];
    const handednessInfo: Array<{ label: string; score: number }> = [];

    handsData.forEach((landmarks: any[], index: number) => {
      const rawLabel = ((handednessData[index] as any)?.label as string) || `Hand ${index + 1}`;
      const handLabel = rawLabel === 'Left' ? 'R' : rawLabel === 'Right' ? 'L' : rawLabel;
      const score = ((handednessData[index] as any)?.score as number) || 1.0;

      const smoothedHand = this.smoothPoints(this.landmarkSmoothing[handLabel], landmarks);
      this.landmarkSmoothing[handLabel] = smoothedHand;

      processedHands.push(smoothedHand);
      handednessInfo.push({ label: handLabel, score });

      // Track index tip (landmark 8) for spatial trajectory trail
      if (smoothedHand[8]) {
        this.trajectoryTrail.push({ ...smoothedHand[8] });
        if (this.trajectoryTrail.length > this.maxTrailLength) {
          this.trajectoryTrail.shift();
        }
      }

      this.renderHand(ctx, smoothedHand, handLabel);
    });

    // Render RDTR01 Three.js-style Spatial Trajectory Particle Trail
    this.renderTrajectoryTrail(ctx);

    ctx.restore();

    this.onStatusCallback({
      cameraActive: true,
      permissionGranted: true,
      handsCount: processedHands.length,
      faceDetected: smoothedFace !== null,
      headGesture: this.currentHeadGesture,
      isPrivacyMode: true,
      signerHandedness: 'AUTO',
      isOneHandedMode: processedHands.length === 1,
      message:
        processedHands.length > 0
          ? `${processedHands.length} hand(s) tracked | Holistic 540+ Active | Head: ${this.currentHeadGesture}`
          : 'Waiting for hand in frame...',
      errorCode: 'NONE',
    });

    this.onFrameCallback({
      hands: processedHands,
      handedness: handednessInfo,
      face: smoothedFace,
      pose: smoothedPose,
      headGesture: this.currentHeadGesture,
      timestamp: now,
      fps: this.currentFps,
    });
  }

  private smoothPoints(previous: Point3D[] | undefined, current: Point3D[]): Point3D[] {
    if (!previous || previous.length !== current.length) {
      return current.map((p) => ({ ...p }));
    }
    return current.map((p, idx) => ({
      x: previous[idx].x * (1 - this.smoothingFactor) + p.x * this.smoothingFactor,
      y: previous[idx].y * (1 - this.smoothingFactor) + p.y * this.smoothingFactor,
      z: previous[idx].z * (1 - this.smoothingFactor) + p.z * this.smoothingFactor,
    }));
  }

  private renderTrajectoryTrail(ctx: CanvasRenderingContext2D): void {
    if (this.trajectoryTrail.length < 2) return;
    const width = this.canvasElement.width;
    const height = this.canvasElement.height;

    ctx.save();
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';

    for (let i = 1; i < this.trajectoryTrail.length; i++) {
      const p1 = this.trajectoryTrail[i - 1];
      const p2 = this.trajectoryTrail[i];
      const alpha = i / this.trajectoryTrail.length;

      ctx.beginPath();
      ctx.moveTo(p1.x * width, p1.y * height);
      ctx.lineTo(p2.x * width, p2.y * height);
      ctx.strokeStyle = `rgba(250, 204, 21, ${alpha * 0.8})`; // Glowing yellow particle trail
      ctx.stroke();
    }
    ctx.restore();
  }

  private renderPoseSkeleton(ctx: CanvasRenderingContext2D, landmarks: Point3D[]): void {
    const width = this.canvasElement.width;
    const height = this.canvasElement.height;

    ctx.save();
    ctx.lineWidth = 3;
    ctx.strokeStyle = this.poseColor;

    // Key shoulder, arm, and torso connections (11: L Shoulder, 12: R Shoulder, 13: L Elbow, 14: R Elbow, 15: L Wrist, 16: R Wrist, 23: L Hip, 24: R Hip)
    const poseConnections = [
      [11, 12], [11, 13], [13, 15],
      [12, 14], [14, 16],
      [11, 23], [12, 24], [23, 24]
    ];

    poseConnections.forEach(([a, b]) => {
      if (landmarks[a] && landmarks[b]) {
        ctx.beginPath();
        ctx.moveTo(landmarks[a].x * width, landmarks[a].y * height);
        ctx.lineTo(landmarks[b].x * width, landmarks[b].y * height);
        ctx.stroke();
      }
    });

    ctx.fillStyle = '#ffffff';
    [11, 12, 13, 14, 15, 16].forEach((idx) => {
      if (landmarks[idx]) {
        ctx.beginPath();
        ctx.arc(landmarks[idx].x * width, landmarks[idx].y * height, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    ctx.restore();
  }

  private renderHand(ctx: CanvasRenderingContext2D, landmarks: Point3D[], label: string): void {
    const width = this.canvasElement.width;
    const height = this.canvasElement.height;

    const centerIndices = [0, 1, 5, 9, 13, 17];
    const palmCenter = centerIndices.reduce(
      (acc, index) => {
        acc.x += landmarks[index].x;
        acc.y += landmarks[index].y;
        return acc;
      },
      { x: 0, y: 0 }
    );
    palmCenter.x /= centerIndices.length;
    palmCenter.y /= centerIndices.length;

    ctx.save();
    const gradient = ctx.createRadialGradient(
      palmCenter.x * width,
      palmCenter.y * height,
      0,
      palmCenter.x * width,
      palmCenter.y * height,
      110
    );
    gradient.addColorStop(0, 'rgba(250,204,21,0.35)');
    gradient.addColorStop(1, 'rgba(59,130,246,0)');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(palmCenter.x * width, palmCenter.y * height, 110, 0, Math.PI * 2);
    ctx.fill();

    ctx.lineWidth = 4;
    ctx.strokeStyle = this.lineColor;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const drawConnectors = (window as any).drawConnectors;
    const HAND_CONNECTIONS = (window as any).HAND_CONNECTIONS;
    const extraPalmConnections = [
      [0, 9],
      [0, 13],
    ];

    if (drawConnectors && HAND_CONNECTIONS) {
      drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: this.lineColor, lineWidth: 4 });
      extraPalmConnections.forEach(([a, b]) => {
        const start = landmarks[a];
        const end = landmarks[b];
        ctx.beginPath();
        ctx.moveTo(start.x * width, start.y * height);
        ctx.lineTo(end.x * width, end.y * height);
        ctx.stroke();
      });
    }

    ctx.fillStyle = this.pointColor;
    ctx.strokeStyle = this.lineColor;
    const tipIndices = [4, 8, 12, 16, 20];
    const fingerLabels: Record<number, string> = {
      4: 'T1',
      8: 'I2',
      12: 'M3',
      16: 'R4',
      20: 'P5',
    };

    landmarks.forEach((landmark, index) => {
      const x = landmark.x * width;
      const y = landmark.y * height;
      const isTip = tipIndices.includes(index);
      const radius = index === 0 ? 10 : isTip ? 7 : 5;

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.stroke();

      if (isTip) {
        ctx.beginPath();
        ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(59,130,246,0.25)';
        ctx.stroke();
        ctx.strokeStyle = this.lineColor;

        const labelText = fingerLabels[index];
        if (labelText) {
          ctx.fillStyle = '#ffffff';
          ctx.font = '700 13px "Geist Mono", monospace';
          ctx.textBaseline = 'bottom';
          ctx.fillText(labelText, x + 10, y - 8);
        }
      }
    });

    const wrist = landmarks[0];
    ctx.fillStyle = '#ffffff';
    ctx.font = '700 14px "Geist Mono", monospace';
    ctx.fillText(`[ HAND ${label} ]`, wrist.x * width + 12, wrist.y * height - 10);
    ctx.restore();
  }

  private renderFaceLandmarks(ctx: CanvasRenderingContext2D, landmarks: Point3D[]): void {
    const drawConnectors = (window as any).drawConnectors;
    const FACEMESH_TESSELATION = (window as any).FACEMESH_TESSELATION;
    const FACEMESH_RIGHT_EYE = (window as any).FACEMESH_RIGHT_EYE;
    const FACEMESH_LEFT_EYE = (window as any).FACEMESH_LEFT_EYE;
    const FACEMESH_LIPS = (window as any).FACEMESH_LIPS;
    const FACEMESH_FACE_OVAL = (window as any).FACEMESH_FACE_OVAL;

    ctx.save();
    ctx.lineWidth = 1.5;

    if (drawConnectors && FACEMESH_TESSELATION) {
      drawConnectors(ctx, landmarks, FACEMESH_TESSELATION, { color: 'rgba(37, 244, 238, 0.15)', lineWidth: 1 });
    }
    if (drawConnectors && FACEMESH_RIGHT_EYE) {
      drawConnectors(ctx, landmarks, FACEMESH_RIGHT_EYE, { color: 'rgba(250,204,21,0.5)', lineWidth: 1 });
    }
    if (drawConnectors && FACEMESH_LEFT_EYE) {
      drawConnectors(ctx, landmarks, FACEMESH_LEFT_EYE, { color: 'rgba(250,204,21,0.5)', lineWidth: 1 });
    }
    if (drawConnectors && FACEMESH_LIPS) {
      drawConnectors(ctx, landmarks, FACEMESH_LIPS, { color: 'rgba(37, 244, 238, 0.8)', lineWidth: 1.5 });
    }
    if (drawConnectors && FACEMESH_FACE_OVAL) {
      drawConnectors(ctx, landmarks, FACEMESH_FACE_OVAL, { color: 'rgba(37, 244, 238, 0.4)', lineWidth: 1 });
    }

    ctx.fillStyle = this.faceColor; // Cyan face mesh points
    const width = this.canvasElement.width;
    const height = this.canvasElement.height;
    landmarks.forEach((landmark) => {
      const x = landmark.x * width;
      const y = landmark.y * height;
      ctx.beginPath();
      ctx.arc(x, y, 1.2, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.restore();
  }
}
