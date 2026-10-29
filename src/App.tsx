import { useEffect, useRef, useState } from 'react';
import { ASL_ENGLISH_PACK } from './languages/aslEnglish';
import { LiveInterpreter, type LiveSnapshot } from './recognition/liveInterpreter';
import { assessFrame } from './recognition/frameQuality';
import { letterModelFromJson, predictLetter } from './recognition/letterModel';
import { decodeF32, glossToEnglish, modelFromJson, predictSequence, type Prediction, type SequenceModel } from './recognition/sequenceModel';
import {
  applyPersonalLabel,
  loadPersonalSamples,
  persistPersonalSamples,
  sampleFromSequence,
  type PersonalSample,
} from './personalize/personalSigns';
import { publicUrl } from './publicUrl';
import { glossesToEnglish } from './textAssembly/glossEnglish';
import { SignTextAssembler } from './textAssembly/SignTextAssembler';
import { MediaPipeTracker } from './tracking/MediaPipeTracker';
import { HearingChannel } from './reverseChannel/HearingChannel';
import type { ConversationMessage, FrameLandmarksData, TrackingStatus } from './types/signbridge';

type DemoClip = { label: string; features: string };

function classifierFor(
  model: SequenceModel,
  personal: () => PersonalSample[],
  rememberSequence: (sequence: number[][]) => void,
  onPersonal: (active: boolean) => void,
) {
  const withPersonal = (sequence: number[][], prediction: Prediction) => {
    const next = applyPersonalLabel(prediction, sequence, personal());
    onPersonal(next !== prediction);
    return next;
  };
  return {
    labels: model.labels,
    emitThreshold: model.emitThreshold,
    marginThreshold: model.marginThreshold,
    classThresholds: model.classThresholds,
    predict: (sequence: number[][]) => {
      rememberSequence(sequence);
      return withPersonal(sequence, predictSequence(model, sequence));
    },
    predictPreview: (sequence: number[][]) => withPersonal(sequence, predictSequence(model, sequence, { members: 1, tta: false })),
  };
}

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
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
      text: 'Sign a gloss. SignBridge keeps the gloss and writes a short English sentence on this device.',
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
  const [sttNote, setSttNote] = useState('');
  const [personalSamples, setPersonalSamples] = useState<PersonalSample[]>(() => loadPersonalSamples());
  const [personalLabel, setPersonalLabel] = useState('');
  const [personalMessage, setPersonalMessage] = useState('Sign or replay a sample, then save it here.');
  const [personalHit, setPersonalHit] = useState(false);
  const personalRef = useRef<PersonalSample[]>([]);
  const lastSequenceRef = useRef<number[][] | null>(null);
  personalRef.current = personalSamples;

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
          fetch(publicUrl(ASL_ENGLISH_PACK.modelUrl)),
          fetch(publicUrl('/models/demo-clips.json')),
          fetch(publicUrl('/models/asl-fingerspelling.json')),
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
        interpreterRef.current = new LiveInterpreter(classifierFor(model, () => personalRef.current, (sequence) => {
          lastSequenceRef.current = sequence;
        }, setPersonalHit), {
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
    const tracker = new MediaPipeTracker(videoRef.current, canvasRef.current, onFrame, (status) => {
      setTrackingStatus((prev) => (
        prev.cameraActive === status.cameraActive
        && prev.permissionGranted === status.permissionGranted
        && prev.handsCount === status.handsCount
        && prev.poseDetected === status.poseDetected
        && prev.message === status.message
        && prev.errorCode === status.errorCode
          ? prev
          : status
      ));
    });
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

  const previewTokens = spelling.length > 0 ? [...words, spelling.join('')] : words;
  const englishPreview = glossesToEnglish(previewTokens);
  const saveLastSign = () => {
    const sequence = lastSequenceRef.current;
    const label = personalLabel.trim() || (predictedSign !== 'Show a sign' && predictedSign !== 'Loading model…' ? predictedSign.trim() : '');
    if (!sequence || sequence.length === 0 || !label) {
      setPersonalMessage('Sign or replay a sample, and give it a label.');
      return;
    }
    const next = [...personalSamples, sampleFromSequence(label, sequence)].slice(-40);
    personalRef.current = next;
    setPersonalSamples(next);
    persistPersonalSamples(next);
    setPersonalMessage(`Saved “${label}” in this browser. It is not uploaded.`);
  };
  const removePersonal = (id: string) => {
    const next = personalSamples.filter((sample) => sample.id !== id);
    personalRef.current = next;
    setPersonalSamples(next);
    persistPersonalSamples(next);
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
        <p className="caption-kicker">English</p>
        <p id="english-line" className="english-line">
          {englishPreview.english || 'Signs become an English sentence here.'}
        </p>
        {englishPreview.note && <p className="english-note">{englishPreview.note}</p>}
        <p className="caption-kicker">Glosses</p>
        <div className="caption-line">
          {words.length === 0 && spelling.length === 0 ? (
            <span className="caption-placeholder caption-placeholder-small">Recognized glosses stay here. Tap one to remove it.</span>
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
          {personalHit && <span className="status-chip is-live">Your sign</span>}
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
            <video ref={videoRef} className="video" autoPlay playsInline muted aria-hidden="true" />
            <canvas ref={canvasRef} className="video" aria-hidden="true" />
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
            <button
              type="button"
              className={isMicListening ? 'is-on' : ''}
              aria-pressed={isMicListening}
              aria-describedby="stt-note"
              onClick={() => {
                const result = hearingRef.current?.toggleMicrophone();
                if (result === 'on') {
                  setSttNote('Microphone audio goes to your browser’s speech service, not to SignBridge. The camera stays on this device.');
                } else if (result === 'unsupported') {
                  setSttNote('Speech to text is not available in this browser. Type instead. The camera stays on this device.');
                } else {
                  setSttNote('');
                }
              }}
            >
              {isMicListening ? 'Listening…' : 'Speech to text'}
            </button>
          </div>
          <p id="stt-note" className="privacy-note">{sttNote || 'Speech to text is off. The camera is never uploaded.'}</p>
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
              {msg.glosses && <p className="chat-glosses">Glosses: {msg.glosses}</p>}
            </article>
          ))}
        </div>
      </section>

      <section className="personal-box" aria-label="Your signs">
        <h2>Your signs</h2>
        <p>These stay in this browser. Video is not stored or uploaded. A close match can replace the shared guess for you.</p>
        <form
          className="personal-form"
          onSubmit={(event) => {
            event.preventDefault();
            saveLastSign();
          }}
        >
          <label htmlFor="personal-label">Label</label>
          <input
            id="personal-label"
            className="dict-search-input"
            list="known-glosses"
            placeholder="hello"
            value={personalLabel}
            onChange={(event) => setPersonalLabel(event.target.value)}
          />
          <button id="save-personal" className="primary" type="submit">Save last sign</button>
          <button
            id="clear-personal"
            type="button"
            onClick={() => {
              personalRef.current = [];
              setPersonalSamples([]);
              persistPersonalSamples([]);
              setPersonalMessage('Cleared the signs saved in this browser.');
            }}
            disabled={personalSamples.length === 0}
          >
            Clear
          </button>
        </form>
        <datalist id="known-glosses">
          {modelLabels.map((label) => (
            <option key={label} value={label} />
          ))}
        </datalist>
        <p id="personal-message">{personalMessage}</p>
        <ul className="personal-list">
          {personalSamples.map((sample) => (
            <li key={sample.id}>
              <span>{sample.label}</span>
              <button type="button" onClick={() => removePersonal(sample.id)} aria-label={`Remove saved ${sample.label}`}>
                Remove
              </button>
            </li>
          ))}
        </ul>
      </section>

      <details className="dict-box">
        <summary>Words this model knows ({modelLabels.length})</summary>
        <div className="dict-controls">
          <label className="sr-only" htmlFor="gloss-search">Search glosses</label>
          <input id="gloss-search" className="dict-search-input" type="search" placeholder="Search" value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} />
        </div>
        <ul className="gloss-list">
          {filtered.map((label) => (
            <li key={label}>{label}</li>
          ))}
        </ul>
      </details>
      <p className="privacy-note">Camera video stays on this device and is not uploaded. Saved signs are landmark numbers in this browser only.</p>
    </div>
  );
}


export default App;
