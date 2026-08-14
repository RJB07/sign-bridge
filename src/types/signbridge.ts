// Unified TypeScript Interfaces for SignBridge System Architecture
// Enhanced with Google DeepMind SL2T (Sign-Language-to-Text) Standards

export type Point3D = {
  x: number;
  y: number;
  z: number;
};

export type HandLandmarks = Point3D[];
export type FaceLandmarks = Point3D[];
export type PoseLandmarks = Point3D[];

export type HeadGestureType = 'NOD_YES' | 'SHAKE_NO' | 'TILT' | 'EYEBROWS_RAISED' | 'EYEBROWS_SQUEEZED' | 'NONE';

export type FrameLandmarksData = {
  hands: HandLandmarks[];
  handedness: Array<{ label: string; score: number }>;
  face: FaceLandmarks | null;
  pose: PoseLandmarks | null;
  headGesture: HeadGestureType;
  timestamp: number;
  fps: number;
};

export type TrackingStatus = {
  cameraActive: boolean;
  permissionGranted: boolean;
  handsCount: number;
  faceDetected: boolean;
  headGesture: HeadGestureType;
  message: string;
  isPrivacyMode: boolean; // DeepMind SL2T Privacy: Video frames discarded; coordinate streaming only
  signerHandedness: 'AUTO' | 'LEFT' | 'RIGHT';
  isOneHandedMode: boolean; // Adaptive 1-hand mobile dictation
  errorCode: 'NONE' | 'PERMISSION_DENIED' | 'NO_CAMERA' | 'LOW_CONFIDENCE' | 'LIBRARY_LOAD_ERROR';
};

export type RecognitionMode = 'FINGERSPELLING' | 'COMMON_WORDS' | 'HYBRID' | 'SL2T_DIRECT_STREAM';

export type RecognitionResult = {
  text: string;
  confidence: number;
  mode: RecognitionMode;
  isWordBoundary: boolean;
  timestamp: number;
};

export type ConversationMessage = {
  id: string;
  sender: 'SIGNER' | 'HEARING';
  text: string;
  timestamp: string;
  confidence?: number;
  grammarRuleApplied?: string;
};
