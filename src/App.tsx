import { useEffect, useRef, useState } from 'react';
import { ASL_ENGLISH_PACK } from './languages/aslEnglish';
import { LiveInterpreter, type LiveSnapshot } from './recognition/liveInterpreter';
import { decodeF32, glossToEnglish, modelFromJson, predictSequence, type SequenceModel } from './recognition/sequenceModel';
import { exportDatasetJSON, parseImportedDatasetJSON, type SignDataset } from './signModel';
import { loadSavedDataset, saveDataset } from './storage';
import { SignTextAssembler } from './textAssembly/SignTextAssembler';
import { MediaPipeTracker } from './tracking/MediaPipeTracker';
import { HearingChannel } from './reverseChannel/HearingChannel';
import type { ConversationMessage, FrameLandmarksData, TrackingStatus } from './types/signbridge';

type DemoClip = { label: string; features: string };

function classifierFor(model: SequenceModel) {
  return {
    labels: model.labels,
    emitThreshold: model.emitThreshold,
    marginThreshold: model.marginThreshold,
    predict: (sequence: number[][]) => predictSequence(model, sequence),
  };
}

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);
  const assemblerRef = useRef<SignTextAssembler | null>(null);
  const hearingRef = useRef<HearingChannel | null>(null);
  const interpreterRef = useRef<LiveInterpreter | null>(null);
  const demoClipsRef = useRef<DemoClip[]>([]);
  const demoIndexRef = useRef(0);
  const uiStampRef = useRef(0);

  const [fps, setFps] = useState(0);
  const [forcedHand, setForcedHand] = useState<'AUTO' | 'LEFT' | 'RIGHT'>('AUTO');
  const [trackingStatus, setTrackingStatus] = useState<TrackingStatus>({
    cameraActive: false,
    permissionGranted: false,
    handsCount: 0,
    poseDetected: false,
    message: 'Loading the on-device ASL model…',
    errorCode: 'NONE',
  });
  const [predictedSign, setPredictedSign] = useState('Loading model…');
  const [matchConfidence, setMatchConfidence] = useState(0);
  const [topGuesses, setTopGuesses] = useState<string>('');
  const [phase, setPhase] = useState<LiveSnapshot['phase']>('idle');
  const [currentPhrase, setCurrentPhrase] = useState<string[]>([]);
  const [modelLabels, setModelLabels] = useState<string[]>([]);
  const [modelSummary, setModelSummary] = useState('Model not loaded');
  const [conversation, setConversation] = useState<ConversationMessage[]>([
    {
      id: 'welcome',
      sender: 'SIGNER',
      text: 'Sign an isolated ASL gloss. SignBridge shows the English word on this device.',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [isMicListening, setIsMicListening] = useState(false);
  const [interimHearingText, setInterimHearingText] = useState('');
  const [typedReplyText, setTypedReplyText] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [dataset, setDataset] = useState<SignDataset>(() => loadSavedDataset());

  const speechRef = useRef(speechEnabled);
  speechRef.current = speechEnabled;

  const speak = (text: string) => {
    if (!speechRef.current || !text || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    window.speechSynthesis.speak(utterance);
  };

  useEffect(() => {
    assemblerRef.current = new SignTextAssembler(
      (phrase) => setCurrentPhrase(phrase),
      (message) => {
        setConversation((prev) => [...prev, message]);
        speak(message.text);
      },
    );
    hearingRef.current = new HearingChannel(
      (message) => setConversation((prev) => [...prev, message]),
      (listening, interim) => {
        setIsMicListening(listening);
        setInterimHearingText(interim || '');
      },
    );
  }, []);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [modelResponse, demoResponse] = await Promise.all([
          fetch(ASL_ENGLISH_PACK.modelUrl),
          fetch('/models/demo-clips.json'),
        ]);
        if (!modelResponse.ok) throw new Error(`model HTTP ${modelResponse.status}`);
        const payload = await modelResponse.json();
        const model = modelFromJson(payload);
        if (cancelled) return;
        interpreterRef.current = new LiveInterpreter(classifierFor(model));
        setModelLabels(model.labels.map(glossToEnglish));
        if (payload.metrics) {
          const metrics = payload.metrics;
          setModelSummary(
            `${metrics.classes} English glosses · test top-1 ${(metrics.testTop1 * 100).toFixed(1)}% · top-5 ${(metrics.testTop5 * 100).toFixed(1)}%`,
          );
        }
        if (demoResponse.ok) {
          const demo = await demoResponse.json();
          demoClipsRef.current = demo.clips ?? [];
        }
        setPredictedSign('Show a sign');
        setTrackingStatus((prev) => ({ ...prev, message: 'Model ready. Allow the camera, or replay a held-out sign.' }));
      } catch (error) {
        if (cancelled) return;
        setPredictedSign('Model unavailable');
        setTrackingStatus((prev) => ({
          ...prev,
          message: `Could not load the ASL model (${error}).`,
          errorCode: 'MODEL_LOAD_ERROR',
        }));
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!videoRef.current || !canvasRef.current) return;
    const applySnapshot = (snap: LiveSnapshot) => {
      setPhase(snap.phase);
      if (snap.committed) {
        setPredictedSign(snap.committed.text);
        setMatchConfidence(snap.committed.confidence * 100);
        assemblerRef.current?.pushSign(snap.committed.text, snap.committed.confidence);
      } else if (snap.candidate) {
        setPredictedSign(snap.candidate);
        setMatchConfidence(snap.confidence * 100);
      } else if (snap.phase === 'signing') {
        setPredictedSign('Signing…');
      }
      if (snap.top.length > 0) {
        setTopGuesses(snap.top.map((item) => `${item.label} ${(item.probability * 100).toFixed(0)}%`).join(' · '));
      }
    };
    const onFrame = (data: FrameLandmarksData) => {
      setFps(data.fps);
      const interpreter = interpreterRef.current;
      if (!interpreter) return;
      const snap = interpreter.push({
        timeMs: data.timestamp,
        left: data.leftHand,
        right: data.rightHand,
        pose: data.pose,
      });
      const now = performance.now();
      if (snap.committed || now - uiStampRef.current > 150) {
        uiStampRef.current = now;
        applySnapshot(snap);
      }
    };
    const tracker = new MediaPipeTracker(videoRef.current, canvasRef.current, onFrame, setTrackingStatus);
    tracker.start();
    return () => tracker.stop();
  }, []);

  useEffect(() => {
    const hand = forcedHand === 'LEFT' ? 'Left' : forcedHand === 'RIGHT' ? 'Right' : null;
    interpreterRef.current?.setForcedHand(hand);
  }, [forcedHand]);

  useEffect(() => {
    chatScrollRef.current?.scrollTo({ top: chatScrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [conversation, interimHearingText]);

  const replayHeldOut = () => {
    const interpreter = interpreterRef.current;
    const clips = demoClipsRef.current;
    if (!interpreter || clips.length === 0) return;
    const clip = clips[demoIndexRef.current % clips.length];
    demoIndexRef.current += 1;
    const flat = decodeF32(clip.features);
    const sequence: number[][] = [];
    for (let t = 0; t < flat.length / 147; t++) {
      sequence.push(Array.from(flat.slice(t * 147, (t + 1) * 147)));
    }
    const result = interpreter.classifyPrepared(sequence);
    setPredictedSign(result.label);
    setMatchConfidence(result.confidence * 100);
    setPhase('idle');
    setTopGuesses('');
    assemblerRef.current?.pushSign(result.label, result.confidence);
  };

  const filtered = modelLabels.filter((label) => label.toLowerCase().includes(searchQuery.toLowerCase()));

  return (
    <div className="app-shell">
      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        style={{ display: 'none' }}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (!file) return;
          const reader = new FileReader();
          reader.onload = () => {
            const imported = parseImportedDatasetJSON(String(reader.result ?? ''));
            if (imported.length === 0) return;
            const merged = [...dataset, ...imported];
            setDataset(merged);
            saveDataset(merged);
          };
          reader.readAsText(file);
        }}
      />
      <div className="layout">
        <div className="video-box">
          <span className="corner-cross">+</span>
          <div className="video-header">
            <span>ASL TO ENGLISH</span>
            <div className="telemetry-badges">
              <span className="status-pill">{fps} FPS</span>
              <span className="status-pill highlight">{phase.toUpperCase()}</span>
              <span className="status-pill">ON DEVICE</span>
              <button
                className="status-pill-btn"
                type="button"
                onClick={() => setForcedHand(forcedHand === 'AUTO' ? 'LEFT' : forcedHand === 'LEFT' ? 'RIGHT' : 'AUTO')}
              >
                HAND: {forcedHand}
              </button>
            </div>
          </div>
          <div className="video-frame mirror-view">
            <video ref={videoRef} className="video" autoPlay playsInline muted />
            <canvas ref={canvasRef} className="video" />
          </div>
          <div className="video-footer">
            <div>{trackingStatus.message}</div>
            <div className="grammar-badge">{modelSummary}</div>
          </div>
        </div>
        <div className="side-stack">
          <div className="chat-box">
            <span className="corner-cross">+</span>
            <div className="chat-header">
              <div>ENGLISH CAPTIONS</div>
              <button className="tts-toggle-btn" type="button" onClick={() => setSpeechEnabled(!speechEnabled)}>
                {speechEnabled ? 'SPEECH ON' : 'SPEECH OFF'}
              </button>
            </div>
            <div className="chat-status">
              <div>
                <span>English gloss</span>
                <strong>{predictedSign}</strong>
              </div>
              <div>
                <span>Confidence</span>
                <strong>{matchConfidence > 0 ? `${matchConfidence.toFixed(0)}%` : '—'}</strong>
              </div>
            </div>
            <div className="phrase-builder-box">
              <div className="phrase-header">
                <span>PHRASE</span>
                <div className="phrase-actions">
                  <button className="btn-phrase-speak" type="button" onClick={() => speak(currentPhrase.join(' '))} disabled={currentPhrase.length === 0}>
                    SPEAK
                  </button>
                  <button className="btn-phrase-clear" type="button" onClick={() => assemblerRef.current?.clearBuffer()} disabled={currentPhrase.length === 0}>
                    CLEAR
                  </button>
                  <button id="replay-held-out" className="btn-phrase-mock" type="button" onClick={replayHeldOut}>
                    REPLAY HELD-OUT SIGN
                  </button>
                </div>
              </div>
              <div className="phrase-display">
                {currentPhrase.length === 0 ? (
                  <span className="phrase-empty">Recognized English words collect here. A pause finishes the line.</span>
                ) : (
                  currentPhrase.join(' ')
                )}
              </div>
              {topGuesses && <div className="grammar-badge">Top guesses: {topGuesses}</div>}
            </div>
            <div className="chat-list" ref={chatScrollRef}>
              {conversation.map((msg) => (
                <div key={msg.id} className={`chat-item ${msg.sender === 'HEARING' ? 'chat-item-hearing' : ''}`}>
                  <div className="chat-msg-header">
                    <span className="chat-msg-sender">{msg.sender === 'SIGNER' ? 'SIGNER' : 'HEARING PERSON'}</span>
                    <span className="chat-msg-time">{msg.timestamp}</span>
                  </div>
                  <div className="chat-msg-body">{msg.text}</div>
                </div>
              ))}
              {interimHearingText && (
                <div className="chat-item chat-item-hearing interim">
                  <span className="chat-msg-sender">HEARING PERSON</span>
                  <div className="chat-msg-body">{interimHearingText}</div>
                </div>
              )}
            </div>
            <div className="reverse-channel-bar">
              <button className={`btn-mic-toggle ${isMicListening ? 'active' : ''}`} type="button" onClick={() => hearingRef.current?.toggleMicrophone()}>
                {isMicListening ? 'LISTENING…' : 'SPEAK REPLY'}
              </button>
              <form
                className="typed-reply-form"
                onSubmit={(event) => {
                  event.preventDefault();
                  if (!typedReplyText.trim()) return;
                  hearingRef.current?.sendTypedReply(typedReplyText);
                  setTypedReplyText('');
                }}
              >
                <label className="sr-only" htmlFor="reply-input">
                  Reply for the signer to read
                </label>
                <input
                  id="reply-input"
                  className="typed-reply-input"
                  placeholder="Type an English reply…"
                  value={typedReplyText}
                  onChange={(event) => setTypedReplyText(event.target.value)}
                />
                <button className="btn-send-reply" type="submit">
                  SEND
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
      <section className="dict-box">
        <span className="corner-cross">+</span>
        <div className="dict-header">
          <div>ASL GLOSSES THIS MODEL CAN SAY IN ENGLISH</div>
          <div className="dataset-actions-row">
            <button className="btn-dataset-action" type="button" onClick={() => dataset.length && exportDatasetJSON(dataset)}>
              EXPORT SAMPLES
            </button>
            <button className="btn-dataset-action" type="button" onClick={() => fileInputRef.current?.click()}>
              IMPORT SAMPLES
            </button>
          </div>
        </div>
        <div className="dict-controls">
          <input
            className="dict-search-input"
            type="search"
            placeholder="Search the trained English glosses"
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
          />
        </div>
        <div className="dict-grid">
          {filtered.map((label) => (
            <div key={label} className="dict-card">
              <div className="dict-card-head">
                <span className="dict-card-title">{label}</span>
                <span className="dict-card-category">ASL</span>
              </div>
              <p className="dict-card-desc">Isolated sign. Output is this English gloss, not a full sentence translation.</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export default App;
