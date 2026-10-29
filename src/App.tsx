import { useEffect, useRef, useState } from 'react';
import { ASL_ENGLISH_PACK } from './languages/aslEnglish';
import { LiveInterpreter, type LiveSnapshot } from './recognition/liveInterpreter';
import { assessFrame } from './recognition/frameQuality';
import { letterModelFromJson, predictLetter } from './recognition/letterModel';
import { decodeF32, glossToEnglish, modelFromJson, predictSequence, type SequenceModel } from './recognition/sequenceModel';
import { exportDatasetJSON, parseImportedDatasetJSON, type SignDataset } from './dataset';
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
      classThresholds: model.classThresholds,
      predict: (sequence: number[][]) => predictSequence(model, sequence),
      predictPreview: (sequence: number[][]) => predictSequence(model, sequence, { members: 1, tta: false }),
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
  const featureDimRef = useRef(273);
  const letterPredictRef = useRef<((landmarks: import('./recognition/features').Landmark[]) => { label: string; confidence: number } | null) | null>(null);
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
  const [frameHint, setFrameHint] = useState('Sign one word, then pause. Keep the whole hand in frame.');
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
  const [speechEnabled, setSpeechEnabled] = useState(() => {
    try {
      return localStorage.getItem('signbridge-speech') === 'on';
    } catch {
      return false;
    }
  });
  const [theme, setTheme] = useState<'light' | 'dark' | 'contrast'>(() => {
    const saved = localStorage.getItem('signbridge-theme');
    if (saved === 'light' || saved === 'dark' || saved === 'contrast') return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });
  const [textSize, setTextSize] = useState<'sm' | 'md' | 'lg' | 'xl'>(() => {
    const saved = localStorage.getItem('signbridge-text');
    if (saved === 'sm' || saved === 'md' || saved === 'lg' || saved === 'xl') return saved;
    return 'lg';
  });
  const [words, setWords] = useState<string[]>([]);
  const [spelling, setSpelling] = useState<string[]>([]);
  const [justAdded, setJustAdded] = useState('');
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
      () => {
        const assembler = assemblerRef.current;
        if (!assembler) return;
        setWords(assembler.getWords());
        setSpelling(assembler.getSpelling());
        setCurrentPhrase(assembler.getCurrentPhraseText());
      },
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
        const [modelResponse, demoResponse, letterResponse] = await Promise.all([
          fetch(ASL_ENGLISH_PACK.modelUrl),
          fetch('/models/demo-clips.json'),
          fetch('/models/asl-fingerspelling.json'),
        ]);
        if (!modelResponse.ok) throw new Error(`model HTTP ${modelResponse.status}`);
        const payload = await modelResponse.json();
        const model = modelFromJson(payload);
        if (cancelled) return;
        featureDimRef.current = model.featureDim;
        if (letterResponse.ok) {
          const letters = letterModelFromJson(await letterResponse.json());
          letterPredictRef.current = (hand) => predictLetter(letters, hand);
        }
        interpreterRef.current = new LiveInterpreter(classifierFor(model), {
          letterPredict: (hand) => letterPredictRef.current?.(hand) ?? null,
        });
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
        setJustAdded(snap.committed.text);
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
      setFrameHint(assessFrame({
        leftHand: data.leftHand,
        rightHand: data.rightHand,
        poseDetected: Boolean(data.pose),
      }).message);
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
    if (!justAdded) return;
    const timer = window.setTimeout(() => setJustAdded(''), 900);
    return () => window.clearTimeout(timer);
  }, [justAdded]);

  useEffect(() => {
    localStorage.setItem('signbridge-theme', theme);
    localStorage.setItem('signbridge-text', textSize);
    localStorage.setItem('signbridge-speech', speechEnabled ? 'on' : 'off');
  }, [theme, textSize, speechEnabled]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest('input, textarea')) return;
      const undoKey = event.key === 'Backspace' || ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z');
      if (!undoKey) return;
      event.preventDefault();
      assemblerRef.current?.undo();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const replayHeldOut = () => {
    const interpreter = interpreterRef.current;
    const clips = demoClipsRef.current;
    if (!interpreter || clips.length === 0) return;
    const clip = clips[demoIndexRef.current % clips.length];
    demoIndexRef.current += 1;
    const flat = decodeF32(clip.features);
    const dim = featureDimRef.current;
    const sequence: number[][] = [];
    for (let t = 0; t < flat.length / dim; t++) {
      sequence.push(Array.from(flat.slice(t * dim, (t + 1) * dim)));
    }
    const result = interpreter.classifyPrepared(sequence);
    setPredictedSign(result.label);
    setMatchConfidence(result.confidence * 100);
    setPhase('idle');
    setTopGuesses('');
    setJustAdded(result.label);
    assemblerRef.current?.pushSign(result.label, result.confidence);
  };

  const filtered = modelLabels.filter((label) => label.toLowerCase().includes(searchQuery.toLowerCase()));
  const latestHearing = [...conversation].reverse().find((message) => message.sender === 'HEARING');
  const hearingNow = interimHearingText || latestHearing?.text || '';
  const handCount = trackingStatus.handsCount;
  const handStatus = handCount === 0 ? 'No hand' : handCount === 1 ? 'Hand seen' : 'Two hands';

  return (
    <div className="app-shell" data-theme={theme} data-text={textSize}>
      <a className="skip-link" href="#live-caption">Skip to captions</a>
      <header className="top-bar">
        <div>
          <h1>SignBridge</h1>
          <p>ASL to English, on this device. You do not need sound.</p>
        </div>
        <div className="top-controls" role="toolbar" aria-label="Display settings">
          <div className="control-group" role="group" aria-label="Caption size">
            {(['sm', 'md', 'lg', 'xl'] as const).map((size) => (
              <button key={size} type="button" aria-pressed={textSize === size} className={textSize === size ? 'is-on' : ''} onClick={() => setTextSize(size)}>
                {size === 'sm' ? 'A' : size === 'md' ? 'A+' : size === 'lg' ? 'A++' : 'A+++'}
              </button>
            ))}
          </div>
          <div className="control-group" role="group" aria-label="Theme">
            {([
              ['light', 'Light'],
              ['dark', 'Dark'],
              ['contrast', 'Contrast'],
            ] as const).map(([id, label]) => (
              <button key={id} type="button" aria-pressed={theme === id} className={theme === id ? 'is-on' : ''} onClick={() => setTheme(id)}>
                {label}
              </button>
            ))}
          </div>
          <button
            type="button"
            className={speechEnabled ? 'is-on' : ''}
            aria-pressed={speechEnabled}
            onClick={() => setSpeechEnabled(!speechEnabled)}
          >
            {speechEnabled ? 'Speaking aloud: on' : 'Speaking aloud: off'}
          </button>
        </div>
      </header>

      <section id="live-caption" className={`caption-stage ${justAdded ? 'is-committed' : ''}`} aria-live="polite">
        <p className="caption-kicker">Your words</p>
        <div className="caption-line">
          {words.length === 0 && spelling.length === 0 ? (
            <span className="caption-placeholder">Signs show up here as English.</span>
          ) : (
            words.map((word, index) => (
              <button key={`${word}-${index}`} type="button" className="caption-word" onClick={() => assemblerRef.current?.removeAt(index)} aria-label={`Remove ${word}`}>
                {word}
              </button>
            ))
          )}
        </div>
        <div className="spell-row" aria-label="Fingerspelling">
          {spelling.length === 0 ? (
            <span className="spell-empty">Fingerspelled letters appear here.</span>
          ) : (
            spelling.map((letter, index) => (
              <span key={`${letter}-${index}`} className="spell-tile">{letter}</span>
            ))
          )}
        </div>
        {justAdded && <p className="commit-flash" role="status">Added {justAdded}</p>}
        <div className="status-row">
          <span className={`status-chip ${handCount > 0 ? 'is-live' : ''}`}>{handStatus}</span>
          <span className="status-chip">{phase === 'fingerspelling' ? 'Fingerspelling' : phase === 'signing' ? 'Signing' : 'Waiting'}</span>
          <span className="status-chip">{forcedHand === 'AUTO' ? 'Either hand' : `${forcedHand} hand`}</span>
          <div className="meter" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(matchConfidence)} aria-label="Confidence">
            <div style={{ width: `${Math.max(0, Math.min(100, matchConfidence))}%` }} />
            <span>{matchConfidence > 0 ? `${matchConfidence.toFixed(0)}%` : '—'}</span>
          </div>
        </div>
        <p className="now-reading">{predictedSign}</p>
        <p className={frameHint.match(/frame|closer|light|back|Raise/) ? 'frame-hint warn' : 'frame-hint'}>{frameHint}</p>
        {topGuesses && <p className="top-guesses">Other guesses: {topGuesses}</p>}
        <div className="caption-actions">
          <button type="button" onClick={() => assemblerRef.current?.undo()} disabled={words.length + spelling.length === 0}>Undo</button>
          <button type="button" onClick={() => assemblerRef.current?.clearBuffer()} disabled={words.length + spelling.length === 0}>Clear</button>
          <button type="button" className="primary" onClick={() => assemblerRef.current?.commitLine()} disabled={words.length + spelling.length === 0}>Send line</button>
          <button id="replay-held-out" type="button" onClick={replayHeldOut}>Replay a sample</button>
        </div>
      </section>

      <div className="layout">
        <section className="video-box" aria-label="Camera">
          <div className="video-header">
            <span>Camera</span>
            <div className="telemetry-badges">
              <span className="status-pill">{fps} FPS</span>
              <button
                className="status-pill-btn"
                type="button"
                title="AUTO follows the moving hand, including a left-handed signer. LEFT or RIGHT forces that hand."
                onClick={() => setForcedHand(forcedHand === 'AUTO' ? 'LEFT' : forcedHand === 'LEFT' ? 'RIGHT' : 'AUTO')}
              >
                Hand: {forcedHand}
              </button>
            </div>
          </div>
          <div className="video-frame mirror-view">
            <video ref={videoRef} className="video" autoPlay playsInline muted />
            <canvas ref={canvasRef} className="video" />
          </div>
          <div className="video-footer">
            <div>{trackingStatus.message}</div>
            <div>{modelSummary}</div>
          </div>
        </section>

        <section className="hearing-panel" aria-label="Hearing person">
          <h2>Their words</h2>
          <div className="hearing-hero" aria-live="polite">
            {hearingNow || 'Type, or turn on speech-to-text. Their reply shows here.'}
          </div>
          <div className="caption-actions">
            <button type="button" className={isMicListening ? 'is-on' : ''} aria-pressed={isMicListening} onClick={() => hearingRef.current?.toggleMicrophone()}>
              {isMicListening ? 'Listening…' : 'Speech to text'}
            </button>
          </div>
          <form
            className="typed-reply-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (!typedReplyText.trim()) return;
              hearingRef.current?.sendTypedReply(typedReplyText);
              setTypedReplyText('');
            }}
          >
            <label htmlFor="reply-input">Type to the signer</label>
            <input
              id="reply-input"
              className="typed-reply-input"
              placeholder="English reply"
              value={typedReplyText}
              onChange={(event) => setTypedReplyText(event.target.value)}
            />
            <button className="primary" type="submit">Show text</button>
          </form>
        </section>
      </div>

      <section className="chat-box" aria-label="Conversation">
        <h2>Conversation</h2>
        <div className="chat-list" ref={chatScrollRef}>
          {conversation.map((msg) => (
            <article key={msg.id} className={`chat-item ${msg.sender === 'HEARING' ? 'chat-item-hearing' : ''}`}>
              <header className="chat-msg-header">
                <span>{msg.sender === 'SIGNER' ? 'You signed' : 'They said'}</span>
                <time>{msg.timestamp}</time>
              </header>
              <p>{msg.text}</p>
            </article>
          ))}
        </div>
      </section>

      <details className="dict-box">
        <summary>Words this model knows ({modelLabels.length})</summary>
        <div className="dict-controls">
          <label className="sr-only" htmlFor="gloss-search">Search glosses</label>
          <input id="gloss-search" className="dict-search-input" type="search" placeholder="Search" value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} />
          <button type="button" onClick={() => dataset.length && exportDatasetJSON(dataset)}>Export samples</button>
          <button type="button" onClick={() => fileInputRef.current?.click()}>Import samples</button>
        </div>
        <ul className="gloss-list">
          {filtered.map((label) => (
            <li key={label}>{label}</li>
          ))}
        </ul>
      </details>
      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        className="sr-only"
        aria-label="Import dataset JSON"
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
    </div>
  );
}


export default App;
