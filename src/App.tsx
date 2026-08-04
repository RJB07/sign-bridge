import { useEffect, useMemo, useRef, useState } from 'react';
import type { Results } from '@mediapipe/hands';

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [predictedSign, setPredictedSign] = useState('Waiting for hand...');
  const [message, setMessage] = useState('Initializing camera...');
  const [handsDetected, setHandsDetected] = useState(0);
  const [faceDetected, setFaceDetected] = useState(false);
  const [translationLog, setTranslationLog] = useState<string[]>([]);

  const handResultsRef = useRef<Results | null>(null);
  const faceResultsRef = useRef<any>(null);
  const faceLandmarkSmoothing = useRef<Array<{ x: number; y: number; z: number }>>([]);

  const lineColor = '#3b82f6';
  const pointColor = '#facc15';
  const smoothingFactor = 0.35;
  const landmarkSmoothing = useRef<Record<string, Array<{ x: number; y: number; z: number }>>>({});

  const smoothLandmarks = (
    previous: Array<{ x: number; y: number; z: number }> | undefined,
    current: Array<{ x: number; y: number; z: number }>
  ) => {
    if (!previous || previous.length !== current.length) {
      return current.map((landmark) => ({ ...landmark }));
    }

    return current.map((landmark, index) => ({
      x: previous[index].x * (1 - smoothingFactor) + landmark.x * smoothingFactor,
      y: previous[index].y * (1 - smoothingFactor) + landmark.y * smoothingFactor,
      z: previous[index].z * (1 - smoothingFactor) + landmark.z * smoothingFactor,
    }));
  };

  const renderHand = (
    ctx: CanvasRenderingContext2D,
    landmarks: Array<{ x: number; y: number; z: number }>,
    label: string
  ) => {
    const width = canvasRef.current?.width || 640;
    const height = canvasRef.current?.height || 480;
    const connections = [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [0, 5],
      [5, 6],
      [6, 7],
      [7, 8],
      [5, 9],
      [9, 10],
      [10, 11],
      [11, 12],
      [9, 13],
      [13, 14],
      [14, 15],
      [15, 16],
      [13, 17],
      [17, 18],
      [18, 19],
      [19, 20],
      [0, 17],
    ];

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
      120
    );
    gradient.addColorStop(0, 'rgba(250,204,21,0.35)');
    gradient.addColorStop(1, 'rgba(59,130,246,0)');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(palmCenter.x * width, palmCenter.y * height, 120, 0, Math.PI * 2);
    ctx.fill();

    ctx.lineWidth = 4;
    ctx.strokeStyle = lineColor;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowColor = 'rgba(59,130,246,0.25)';
    ctx.shadowBlur = 8;

    const drawConnectors = (window as any).drawConnectors;
    const HAND_CONNECTIONS = (window as any).HAND_CONNECTIONS;
    const extraPalmConnections = [
      [0, 9],
      [0, 13],
    ];

    if (drawConnectors && HAND_CONNECTIONS) {
      drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: lineColor, lineWidth: 4 });
      extraPalmConnections.forEach(([a, b]) => {
        const start = landmarks[a];
        const end = landmarks[b];
        ctx.beginPath();
        ctx.moveTo(start.x * width, start.y * height);
        ctx.lineTo(end.x * width, end.y * height);
        ctx.stroke();
      });
    } else {
      connections.forEach(([a, b]) => {
        const start = landmarks[a];
        const end = landmarks[b];
        ctx.beginPath();
        ctx.moveTo(start.x * width, start.y * height);
        ctx.lineTo(end.x * width, end.y * height);
        ctx.stroke();
      });
    }

    ctx.shadowBlur = 0;
    ctx.fillStyle = pointColor;
    ctx.strokeStyle = lineColor;
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
        ctx.strokeStyle = lineColor;

        const labelText = fingerLabels[index];
        if (labelText) {
          ctx.fillStyle = '#ffffff';
          ctx.font = '13px Inter, system-ui, sans-serif';
          ctx.textBaseline = 'bottom';
          ctx.fillText(labelText, x + 10, y - 8);
        }
      }
    });

    const wrist = landmarks[0];
    ctx.fillStyle = '#ffffff';
    ctx.font = '14px Inter, system-ui, sans-serif';
    ctx.fillText(label, wrist.x * width + 12, wrist.y * height - 10);
    ctx.restore();
  };

  const renderFaceLandmarks = (ctx: CanvasRenderingContext2D, landmarks: Array<{ x: number; y: number; z: number }>) => {
    const drawConnectors = (window as any).drawConnectors;
    const FACEMESH_TESSELATION = (window as any).FACEMESH_TESSELATION;
    const FACEMESH_RIGHT_EYE = (window as any).FACEMESH_RIGHT_EYE;
    const FACEMESH_LEFT_EYE = (window as any).FACEMESH_LEFT_EYE;
    const FACEMESH_LIPS = (window as any).FACEMESH_LIPS;
    const FACEMESH_FACE_OVAL = (window as any).FACEMESH_FACE_OVAL;

    const width = canvasRef.current?.width || 640;
    const height = canvasRef.current?.height || 480;

    ctx.save();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';

    if (drawConnectors && FACEMESH_TESSELATION) {
      drawConnectors(ctx, landmarks, FACEMESH_TESSELATION, { color: 'rgba(255,255,255,0.12)', lineWidth: 1 });
    }
    if (drawConnectors && FACEMESH_RIGHT_EYE) {
      drawConnectors(ctx, landmarks, FACEMESH_RIGHT_EYE, { color: 'rgba(250,204,21,0.45)', lineWidth: 1 });
    }
    if (drawConnectors && FACEMESH_LEFT_EYE) {
      drawConnectors(ctx, landmarks, FACEMESH_LEFT_EYE, { color: 'rgba(250,204,21,0.45)', lineWidth: 1 });
    }
    if (drawConnectors && FACEMESH_LIPS) {
      drawConnectors(ctx, landmarks, FACEMESH_LIPS, { color: 'rgba(250,204,21,0.7)', lineWidth: 1.5 });
    }
    if (drawConnectors && FACEMESH_FACE_OVAL) {
      drawConnectors(ctx, landmarks, FACEMESH_FACE_OVAL, { color: 'rgba(250,204,21,0.3)', lineWidth: 1 });
    }

    ctx.fillStyle = 'rgba(250,204,21,0.4)';
    landmarks.forEach((landmark) => {
      const x = landmark.x * width;
      const y = landmark.y * height;
      ctx.beginPath();
      ctx.arc(x, y, 1.2, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.restore();
  };

  const hands = useMemo(() => {
    const HandsClass = (window as any).Hands;
    if (!HandsClass) {
      return null;
    }

    const locateFile = (file: string) => {
      const name = file.split('/').pop()?.split('?')[0] || file;
      return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${name}`;
    };

    const instance = new HandsClass({ locateFile });
    instance.setOptions({
      selfieMode: false,
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.75,
      minTrackingConfidence: 0.7,
    });

    return instance;
  }, []);

  const faceMesh = useMemo(() => {
    const FaceMeshClass = (window as any).FaceMesh;
    if (!FaceMeshClass) {
      return null;
    }

    const locateFile = (file: string) => {
      const name = file.split('/').pop()?.split('?')[0] || file;
      return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${name}`;
    };

    const instance = new FaceMeshClass({ locateFile });
    instance.setOptions({
      selfieMode: false,
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.75,
      minTrackingConfidence: 0.7,
    });

    return instance;
  }, []);

  useEffect(() => {
    if (!hands || !faceMesh || !videoRef.current || !canvasRef.current) {
      return;
    }

    const CameraClass = (window as any).Camera;
    if (!CameraClass) {
      setMessage('MediaPipe camera library is not loaded.');
      return;
    }

    const videoElement = videoRef.current;
    const canvasElement = canvasRef.current;
    videoElement.width = 640;
    videoElement.height = 480;
    canvasElement.width = 640;
    canvasElement.height = 480;

    hands.setOptions({ selfieMode: false });
    if (faceMesh) {
      faceMesh.setOptions({ selfieMode: false });
    }

    const getErrorMessage = (error: unknown) =>
      error instanceof Error ? error.message : typeof error === 'string' ? error : JSON.stringify(error);

    const renderCombined = (handResults: Results | null, faceResults: any | null) => {
      const ctx = canvasElement.getContext('2d');
      if (!ctx) return;

      ctx.save();
      ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

      if (faceResults?.multiFaceLandmarks?.length > 0) {
        const smoothedFace = smoothLandmarks(faceLandmarkSmoothing.current, faceResults.multiFaceLandmarks[0]);
        faceLandmarkSmoothing.current = smoothedFace;
        renderFaceLandmarks(ctx, smoothedFace);
      }

      const handsData = handResults?.multiHandLandmarks;
      if (handsData && handsData.length > 0) {
        const handedness = handResults?.multiHandedness || [];

        handsData.forEach((landmarks, handIndex) => {
          const rawLabel = ((handedness[handIndex] as any)?.label as string) || `Hand ${handIndex + 1}`;
          const handLabel = rawLabel === 'Left' ? 'R' : rawLabel === 'Right' ? 'L' : rawLabel;

          const smoothedHand = smoothLandmarks(landmarkSmoothing.current[handLabel], landmarks);
          landmarkSmoothing.current[handLabel] = smoothedHand;
          renderHand(ctx, smoothedHand, handLabel);
        });
      }

      ctx.restore();
    };

    const camera = new CameraClass(videoElement, {
      onFrame: async () => {
        try {
          await hands.send({ image: videoElement });
        } catch (error) {
          console.error('MediaPipe hands send error:', error);
          setMessage(`Hand frame error: ${getErrorMessage(error)}`);
          return;
        }

        if (faceMesh?.send) {
          try {
            await faceMesh.send({ image: videoElement });
          } catch (error) {
            console.error('MediaPipe faceMesh send error:', error);
            setMessage(`Face frame error: ${getErrorMessage(error)}`);
          }
        }
      },
      width: 640,
      height: 480,
    });

    const handsCallback = (results: Results) => {
      handResultsRef.current = results;
      setHandsDetected(results.multiHandLandmarks?.length || 0);
      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        setPredictedSign('Sign Detected');
        setMessage(`${results.multiHandLandmarks.length} hand(s) detected. Translating...`);
      } else {
        setPredictedSign('No hand detected');
        setMessage('Move your hand into view to translate.');
      }
      renderCombined(results, faceResultsRef.current);
    };

    const faceCallback = (results: any) => {
      faceResultsRef.current = results;
      setFaceDetected(results.multiFaceLandmarks?.length > 0);
      renderCombined(handResultsRef.current, results);
    };

    hands.onResults(handsCallback);
    faceMesh.onResults(faceCallback);

    camera
      .start()
      .then(() => {
        setMessage('Camera active. Show your hand to translate.');
        videoElement.muted = true;
        videoElement.playsInline = true;
        videoElement.setAttribute('playsinline', '');
        videoElement.play().catch(() => {
          // autoplay may require user interaction
        });
      })
      .catch(() => {
        setMessage('Unable to start camera. Check permissions.');
      });

    return () => {
      camera.stop();
      hands.close();
      faceMesh?.close?.();
    };
  }, [hands, faceMesh]);

  return (
    <div className="app-shell">
      <section className="hero">
        <p className="eyebrow">Real-time sign language bridge</p>
        <h1>Translate hand gestures into text with live AI tracking</h1>
        <p className="hero-copy">
          Your camera becomes a sign interpreter with face and hand landmark overlay, powered by MediaPipe.
        </p>
        <div className="hero-actions">
          <button className="btn-primary">Start translation</button>
          <button className="btn-secondary">How it works</button>
        </div>
      </section>

      <div className="layout">
        <div className="video-box">
          <div className="video-header">
            <span>Camera capture</span>
            <span className="status-pill">Live</span>
          </div>
          <div className="video-frame">
            <video
              ref={videoRef}
              className="video"
              autoPlay
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              className="video"
            />
          </div>
          <div className="video-footer">{message}</div>
        </div>

        <div className="chat-box">
          <div className="chat-header">
            <div>Translation dashboard</div>
          </div>

          <div className="chat-status">
            <div>
              <span>Predicted</span>
              <strong>{predictedSign}</strong>
            </div>
            <div>
              <span>Hands</span>
              <strong>{handsDetected}</strong>
            </div>
          </div>

          <div className="chat-list">
            {translationLog.length === 0 ? (
              <div className="empty">No signs translated yet.</div>
            ) : (
              translationLog.map((item, index) => (
                <div key={index} className="chat-item">
                  {item}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
