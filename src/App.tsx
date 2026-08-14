import { useEffect, useRef, useState } from 'react';
import { MediaPipeTracker } from './tracking/MediaPipeTracker';
import { SignTextAssembler } from './textAssembly/SignTextAssembler';
import { HearingChannel } from './reverseChannel/HearingChannel';
import { SignLanguageMatcher } from './recognition/SignLanguageMatcher';
import { OPEN_ASL_DICTIONARY } from './aslDictionary';
import { exportDatasetJSON, parseImportedDatasetJSON, type SignDataset } from './signModel';
import { loadSavedDataset, saveDataset } from './storage';
import type { ConversationMessage, FrameLandmarksData, HeadGestureType, TrackingStatus } from './types/signbridge';

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);

  // Tracking Telemetry State (Layer 1) & Google DeepMind SL2T Options
  const [fps, setFps] = useState<number>(0);
  const [headGesture, setHeadGesture] = useState<HeadGestureType>('NONE');
  const [signerHandedness, setSignerHandedness] = useState<'AUTO' | 'LEFT' | 'RIGHT'>('AUTO');
  const [isOneHandedMode, setIsOneHandedMode] = useState<boolean>(false);
  const [isPrivacyMode] = useState<boolean>(true); // DeepMind SL2T: Landmark coordinate streaming only

  const [trackingStatus, setTrackingStatus] = useState<TrackingStatus>({
    cameraActive: false,
    permissionGranted: false,
    handsCount: 0,
    faceDetected: false,
    headGesture: 'NONE',
    message: 'Initializing Google DeepMind SL2T & MediaPipe Vision...',
    isPrivacyMode: true,
    signerHandedness: 'AUTO',
    isOneHandedMode: false,
    errorCode: 'NONE',
  });

  // Layer 2 & 3 Recognition & Text Assembly State
  const [predictedSign, setPredictedSign] = useState<string>('Waiting for hand...');
  const [matchConfidence, setMatchConfidence] = useState<number>(0);
  const [currentPhrase, setCurrentPhrase] = useState<string[]>([]);
  const [activeGrammarRule, setActiveGrammarRule] = useState<string>('Rule #8: SVO Standard');

  // Layer 4 & 5 Unified 2-Way Conversation Stream
  const [conversation, setConversation] = useState<ConversationMessage[]>([
    {
      id: 'welcome-1',
      sender: 'SIGNER',
      text: 'SignBridge Google DeepMind SL2T Active. Perform ASL gestures to translate live.',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      grammarRuleApplied: 'DeepMind SL2T Benchmark Active',
    },
  ]);

  // Layer 5 Reverse Channel State
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [isMicListening, setIsMicListening] = useState(false);
  const [interimHearingText, setInterimHearingText] = useState('');
  const [typedReplyText, setTypedReplyText] = useState('');

  // Dictionary & Book Source Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');
  const [selectedBookSource, setSelectedBookSource] = useState<string>('ALL');
  const [dataset, setDataset] = useState<SignDataset>(() => loadSavedDataset());

  // Class Instances Refs
  const trackerRef = useRef<MediaPipeTracker | null>(null);
  const assemblerRef = useRef<SignTextAssembler | null>(null);
  const hearingChannelRef = useRef<HearingChannel | null>(null);
  const lastSpokenRef = useRef<string>('');

  // Sanitize TTS text so single uppercase letters (e.g. 'A', 'B') are spoken directly as 'a', 'b' without saying 'Capital A', while preserving pronoun 'I'
  const formatTextForSpeech = (text: string): string => {
    if (!text) return '';
    const trimmed = text.trim();
    if (/^[A-Z]$/.test(trimmed)) {
      return trimmed.toLowerCase();
    }
    return text.replace(/\b([A-HJ-Z])\b/g, (_, letter) => letter.toLowerCase());
  };



  // Real-time Text-to-Speech engine for Signer -> Hearing
  const speakWord = (word: string) => {
    if (!speechEnabled || typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    if (!word || word === 'Waiting for hand...' || word === 'No hand detected' || word === 'Unknown' || word === 'Translating...') return;
    if (lastSpokenRef.current === word) return;

    lastSpokenRef.current = word;
    window.speechSynthesis.cancel();

    const spokenText = formatTextForSpeech(word);
    const utterance = new SpeechSynthesisUtterance(spokenText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  };

  const speakPhrase = (phraseText: string) => {
    if (!speechEnabled || typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();

    const spokenText = formatTextForSpeech(phraseText);
    const utterance = new SpeechSynthesisUtterance(spokenText);
    window.speechSynthesis.speak(utterance);
  };


  // Initialize Layer 3 Assembler & Layer 5 Reverse Channel
  useEffect(() => {
    assemblerRef.current = new SignTextAssembler(
      (phrase: string[]) => setCurrentPhrase(phrase),
      (message: ConversationMessage) => {
        setConversation((prev) => [...prev, message]);
        if (message.grammarRuleApplied) {
          setActiveGrammarRule(message.grammarRuleApplied);
        }
        speakPhrase(message.text);
      }
    );

    hearingChannelRef.current = new HearingChannel(
      (message: ConversationMessage) => {
        setConversation((prev) => [...prev, message]);
      },
      (listening: boolean, interim?: string) => {
        setIsMicListening(listening);
        setInterimHearingText(interim || '');
      }
    );
  }, [speechEnabled]);

  // Initialize Layer 1 Tracking & S-Tier Visual Feature Matcher
  useEffect(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const onFrame = (data: FrameLandmarksData) => {
      setFps(data.fps);
      setHeadGesture(data.headGesture);

      if (data.hands && data.hands.length > 0) {
        const primaryHand = data.hands[0];
        const match = SignLanguageMatcher.match(primaryHand, data.headGesture);

        const rawSign = match.confidence > 75 ? match.label : 'Translating...';
        const displaySign = typeof rawSign === 'string'
          ? rawSign.replace(/\s*\(.*\)/, '').split('/')[0].trim()
          : rawSign;

        setPredictedSign(displaySign);
        setMatchConfidence(match.confidence);

        if (match.confidence > 75) {
          speakWord(displaySign);
          assemblerRef.current?.pushSign(displaySign, match.confidence);
        }
      } else {
        setPredictedSign('No hand detected');
        setMatchConfidence(0);
        assemblerRef.current?.checkPauseBoundary();
      }
    };

    const onStatus = (status: TrackingStatus) => {
      setTrackingStatus({
        ...status,
        isPrivacyMode,
        signerHandedness,
        isOneHandedMode,
      });
    };

    trackerRef.current = new MediaPipeTracker(
      videoRef.current,
      canvasRef.current,
      onFrame,
      onStatus
    );

    trackerRef.current.start();

    return () => {
      trackerRef.current?.stop();
    };
  }, [isPrivacyMode, signerHandedness, isOneHandedMode]);

  // Auto-scroll conversation chat log
  useEffect(() => {
    chatScrollRef.current?.scrollTo({
      top: chatScrollRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [conversation, interimHearingText]);

  // Hearing Person Reply Handlers
  const handleToggleMic = () => {
    hearingChannelRef.current?.toggleMicrophone();
  };

  const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));

  const runMockTest = async () => {
    setConversation((prev) => [
      ...prev,
      {
        id: 'mock-start-' + Date.now().toString(36),
        sender: 'HEARING',
        text: 'Running SL2T ASL Test: "Yesterday we went to the store."',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        grammarRuleApplied: 'Rule #2: Time-First Tense',
      },
    ]);

    const seq: Array<{ sign: string; delay: number }> = [
      { sign: 'Yesterday', delay: 0 },
      { sign: 'we', delay: 700 },
      { sign: 'go', delay: 700 },
      { sign: 'to', delay: 500 },
      { sign: 'store', delay: 700 },
    ];

    for (const item of seq) {
      await sleep(item.delay);
      assemblerRef.current?.pushSign(item.sign, 96);
      setPredictedSign(item.sign);
      setMatchConfidence(96);
    }

    await sleep(2000);
    assemblerRef.current?.checkPauseBoundary();
  };

  const handleSendTypedReply = (e: React.FormEvent) => {
    e.preventDefault();
    if (!typedReplyText.trim()) return;
    hearingChannelRef.current?.sendTypedReply(typedReplyText);
    setTypedReplyText('');
  };

  // Free Dataset JSON Export & Import
  const handleExportDataset = () => {
    if (dataset.length === 0) {
      alert('No recorded dataset samples to export. Record or capture some samples first!');
      return;
    }
    exportDatasetJSON(dataset);
  };

  const handleImportJSONClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      const imported = parseImportedDatasetJSON(content);
      if (imported.length > 0) {
        const merged = [...dataset, ...imported];
        setDataset(merged);
        saveDataset(merged);
        alert(`Successfully imported ${imported.length} gesture samples into dataset!`);
      } else {
        alert('Invalid dataset JSON file format.');
      }
    };
    reader.readAsText(file);
  };

  // Filter ASL Dictionary Entries
  const filteredDictionary = OPEN_ASL_DICTIONARY.filter((item) => {
    const matchesCategory = selectedCategory === 'ALL' || item.category === selectedCategory;
    const matchesBook = selectedBookSource === 'ALL' || item.bookSource === selectedBookSource;
    if (!matchesCategory || !matchesBook) return false;

    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return (
      item.label.toLowerCase().includes(q) ||
      item.category.toLowerCase().includes(q) ||
      item.bookSource.toLowerCase().includes(q) ||
      item.description.toLowerCase().includes(q)
    );
  });

  return (
    <div className="app-shell">
      {/* Hidden Dataset File Input */}
      <input
        type="file"
        ref={fileInputRef}
        accept=".json"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      {/* Main 2-Way Oberon Layout */}
      <div className="layout">
        {/* Left Viewport: Signer Camera Capture & Multimodal Tracking */}
        <div className="video-box">
          <span className="corner-cross">+</span>
          <div className="video-header">
            <span>⚡ GOOGLE DEEPMIND SL2T HOLISTIC VIEWPORT</span>
            <div className="telemetry-badges">
              <span className="status-pill">{fps} FPS</span>
              <span className="status-pill highlight">HEAD: {headGesture}</span>
              <span className="status-pill highlight">SL2T PRIVACY: ON</span>
              <button
                className="status-pill-btn"
                onClick={() => setIsOneHandedMode(!isOneHandedMode)}
              >
                {isOneHandedMode ? '📱 1-HAND: ON' : '🤲 2-HAND: DUAL'}
              </button>
              <button
                className="status-pill-btn"
                onClick={() => setSignerHandedness(signerHandedness === 'AUTO' ? 'LEFT' : signerHandedness === 'LEFT' ? 'RIGHT' : 'AUTO')}
              >
                HANDS: {signerHandedness}
              </button>
            </div>
          </div>

          <div className="video-frame">
            <video ref={videoRef} className="video" autoPlay playsInline muted />
            <canvas ref={canvasRef} className="video" />
          </div>

          <div className="video-footer">
            <div>{trackingStatus.message}</div>
            <div className="grammar-badge">Active Grammar Engine: {activeGrammarRule}</div>
          </div>
        </div>

        {/* Right Viewport: Shared 2-Way Conversation Feed */}
        <div className="side-stack">
          <div className="chat-box">
            <span className="corner-cross">+</span>
            <div className="chat-header">
              <div>💬 2-WAY SL2T CONVERSATION STREAM</div>
              <button
                className="tts-toggle-btn"
                onClick={() => setSpeechEnabled(!speechEnabled)}
              >
                {speechEnabled ? '🔊 SPEECH ON' : '🔇 SPEECH OFF'}
              </button>
            </div>

            <div className="chat-status">
              <div>
                <span>Predicted Sign</span>
                <strong>{predictedSign}</strong>
              </div>
              <div>
                <span>Match Confidence</span>
                <strong>{matchConfidence > 0 ? `${matchConfidence.toFixed(1)}%` : '---'}</strong>
              </div>
            </div>

            {/* Current Active Phrase Assembly */}
            <div className="phrase-builder-box">
              <div className="phrase-header">
                <span>SL2T STREAMING PHRASE BUFFER:</span>
                <div className="phrase-actions">
                  <button
                    className="btn-phrase-speak"
                    onClick={() => speakPhrase(currentPhrase.join(' '))}
                    disabled={currentPhrase.length === 0}
                  >
                    🔊 SPEAK
                  </button>
                  <button
                    className="btn-phrase-clear"
                    onClick={() => assemblerRef.current?.clearBuffer()}
                    disabled={currentPhrase.length === 0}
                  >
                    CLEAR
                  </button>
                  <button
                    className="btn-phrase-mock"
                    onClick={() => runMockTest()}
                  >
                    TEST SL2T PHRASE
                  </button>
                </div>
              </div>
              <div className="phrase-display">
                {currentPhrase.length === 0 ? (
                  <span className="phrase-empty">Live coordinate landmark sequences buffer here into spoken text...</span>
                ) : (
                  currentPhrase.join(' ')
                )}
              </div>
            </div>

            {/* Scrollable Conversation Stream */}
            <div className="chat-list" ref={chatScrollRef}>
              {conversation.map((msg) => (
                <div
                  key={msg.id}
                  className={`chat-item ${msg.sender === 'HEARING' ? 'chat-item-hearing' : ''}`}
                >
                  <div className="chat-msg-header">
                    <span className="chat-msg-sender">
                      {msg.sender === 'SIGNER' ? '🤟 DEAF SIGNER' : '🗣️ HEARING PERSON'}
                    </span>
                    <span className="chat-msg-time">{msg.timestamp}</span>
                  </div>
                  <div className="chat-msg-body">{msg.text}</div>
                  {msg.grammarRuleApplied && (
                    <div className="chat-msg-rule">📐 Rule: {msg.grammarRuleApplied}</div>
                  )}
                </div>
              ))}
              {interimHearingText && (
                <div className="chat-item chat-item-hearing interim">
                  <span className="chat-msg-sender">🗣️ HEARING PERSON (LISTENING...)</span>
                  <div className="chat-msg-body">{interimHearingText}</div>
                </div>
              )}
            </div>

            {/* Layer 5 Reverse Channel: Hearing Person Reply Bar */}
            <div className="reverse-channel-bar">
              <button
                className={`btn-mic-toggle ${isMicListening ? 'active' : ''}`}
                onClick={handleToggleMic}
              >
                {isMicListening ? '🎙️ LISTENING...' : '🎤 SPEAK REPLY'}
              </button>

              <form onSubmit={handleSendTypedReply} className="typed-reply-form">
                <input
                  type="text"
                  className="typed-reply-input"
                  placeholder="Type reply for Deaf person to read..."
                  value={typedReplyText}
                  onChange={(e) => setTypedReplyText(e.target.value)}
                />
                <button type="submit" className="btn-send-reply">
                  SEND
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>

      {/* Open-Domain ASL Sign Language Library Explorer */}
      <section className="dict-box">
        <span className="corner-cross">+</span>
        <div className="dict-header">
          <div>📚 OPEN-DOMAIN ASL KNOWLEDGE BASE & SIGN DICTIONARY</div>
          <div className="dataset-actions-row">
            <button className="btn-dataset-action" onClick={handleExportDataset}>
              💾 EXPORT DATASET (.JSON)
            </button>
            <button className="btn-dataset-action" onClick={handleImportJSONClick}>
              📂 IMPORT DATASET (.JSON)
            </button>
          </div>
        </div>

        <div className="dict-controls">
          {/* Category Filters */}
          <div className="category-filters">
            {['ALL', 'Common Word', 'Alphabet', 'Number', 'Classroom', 'Verbs & Actions', 'Time & Calendar', 'Adjectives & Feelings', 'Classifiers', 'Emergency / Care'].map((cat) => (
              <button
                key={cat}
                className={`filter-pill ${selectedCategory === cat ? 'active' : ''}`}
                onClick={() => setSelectedCategory(cat)}
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Book Source Filters */}
          <div className="category-filters" style={{ marginTop: '8px' }}>
            <span style={{ fontSize: '11px', color: '#6b7280', alignSelf: 'center', marginRight: '6px' }}>BOOK SOURCE:</span>
            {['ALL', 'ASL Classroom Book', 'Juicy English Starters', 'ASL The Easy Way', 'ACR GCD Storybook Standards'].map((book) => (
              <button
                key={book}
                className={`filter-pill ${selectedBookSource === book ? 'active' : ''}`}
                onClick={() => setSelectedBookSource(book)}
              >
                {book}
              </button>
            ))}
          </div>

          <input
            type="text"
            className="dict-search-input"
            style={{ marginTop: '12px' }}
            placeholder="Search 63+ open-domain ASL signs, handshapes, grammar rules, classifiers (cl-1, cl-3, cl-55)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div className="dict-grid">
          {filteredDictionary.map((entry, idx) => (
            <div key={idx} className="dict-card">
              <div className="dict-card-head">
                <span className="dict-card-title">{entry.label}</span>
                <span className="dict-card-category">{entry.category}</span>
              </div>
              <p className="dict-card-desc">{entry.description}</p>
              <div className="dict-card-hint">✋ {entry.handshapeHint}</div>
              <div className="dict-card-book">📖 Source: {entry.bookSource}</div>
              {entry.headGestureRule && entry.headGestureRule !== 'NONE' && (
                <div className="dict-card-headrule">🗣️ Head Rule: {entry.headGestureRule}</div>
              )}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export default App;
