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
  leftHand: Point3D[] | null;
  rightHand: Point3D[] | null;
  pose: Point3D[] | null;
  faceDetected: boolean;
  handsCount: number;
  timestamp: number;
  fps: number;
};

export type TrackingStatus = {
  cameraActive: boolean;
  permissionGranted: boolean;
  handsCount: number;
  poseDetected: boolean;
  message: string;
  errorCode: 'NONE' | 'PERMISSION_DENIED' | 'NO_CAMERA' | 'LIBRARY_LOAD_ERROR' | 'MODEL_LOAD_ERROR';
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
