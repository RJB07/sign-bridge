import type { ConversationMessage } from '../types/signbridge';
import { glossesToEnglish } from './glossEnglish';

export class SignTextAssembler {
  private currentWordBuffer: string[] = [];
  private fullPhraseBuffer: string[] = [];
  private lastSign: string = '';
  private lastSignTimestamp: number = 0;
  
  // Settings & SL2T Anti-Hallucination Thresholds
  private debounceMs: number = 600; // Ignore duplicate sign within 600ms
  private wordPauseMs: number = 1800; // Pause > 1.8s triggers word boundary
  private sentencePauseMs: number = 3500; // Pause > 3.5s completes sentence

  private onPhraseUpdate: (phrase: string[]) => void;
  private onSentenceComplete: (message: ConversationMessage) => void;
  private now: () => number;

  constructor(
    onPhraseUpdate: (phrase: string[]) => void,
    onSentenceComplete: (message: ConversationMessage) => void,
    now: () => number = () => performance.now(),
  ) {
    this.onPhraseUpdate = onPhraseUpdate;
    this.onSentenceComplete = onSentenceComplete;
    this.now = now;
  }

  public pushSign(signText: string, confidence: number = 90): void {
    // SL2T Anti-Hallucination Filter: Ignore resting hands / non-signing noise
    if (!signText || signText === 'Unknown' || signText === 'Waiting for hand...' || signText === 'No hand detected' || signText === 'Translating...') {
      this.checkPauseBoundary();
      return;
    }

    const now = this.now();

    // Debounce duplicate frame signs within debounceMs
    if (signText === this.lastSign && now - this.lastSignTimestamp < this.debounceMs) {
      this.lastSignTimestamp = now;
      return;
    }

    this.lastSign = signText;
    this.lastSignTimestamp = now;

    // Check single character (fingerspelling) vs full word sign
    if (signText.length === 1 && /[A-Z0-9]/i.test(signText)) {
      this.currentWordBuffer.push(signText);
    } else {
      // If we had buffered fingerspelled letters, flush them as a completed word
      if (this.currentWordBuffer.length > 0) {
        this.fullPhraseBuffer.push(this.currentWordBuffer.join(''));
        this.currentWordBuffer = [];
      }
      this.fullPhraseBuffer.push(signText);
    }

    this.onPhraseUpdate(this.getCurrentPhraseText());
  }

  public checkPauseBoundary(): void {
    if (this.lastSignTimestamp === 0) return;
    const elapsed = this.now() - this.lastSignTimestamp;

    // Word boundary flush for fingerspelling
    if (elapsed > this.wordPauseMs && this.currentWordBuffer.length > 0) {
      this.fullPhraseBuffer.push(this.currentWordBuffer.join(''));
      this.currentWordBuffer = [];
      this.onPhraseUpdate(this.getCurrentPhraseText());
    }

    if (elapsed > this.sentencePauseMs && (this.fullPhraseBuffer.length > 0 || this.currentWordBuffer.length > 0)) {
      this.commitLine();
    }
  }

  private lineMessage(): ConversationMessage {
    const line = glossesToEnglish(this.fullPhraseBuffer);
    return {
      id: Math.random().toString(36).substring(2, 9),
      sender: 'SIGNER',
      text: line.english,
      glosses: line.glosses,
      grammarRuleApplied: line.note,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    };
  }

  public getWords(): string[] {
    return [...this.fullPhraseBuffer];
  }

  public getSpelling(): string[] {
    return [...this.currentWordBuffer];
  }

  public undo(): void {
    if (this.currentWordBuffer.length > 0) {
      this.currentWordBuffer.pop();
    } else if (this.fullPhraseBuffer.length > 0) {
      this.fullPhraseBuffer.pop();
    } else {
      return;
    }
    this.lastSign = '';
    this.onPhraseUpdate(this.getCurrentPhraseText());
  }

  public removeAt(index: number): void {
    if (index < 0 || index >= this.fullPhraseBuffer.length) return;
    this.fullPhraseBuffer.splice(index, 1);
    this.lastSign = '';
    this.onPhraseUpdate(this.getCurrentPhraseText());
  }

  public commitLine(): void {
    if (this.currentWordBuffer.length > 0) {
      this.fullPhraseBuffer.push(this.currentWordBuffer.join(''));
      this.currentWordBuffer = [];
    }
    if (this.fullPhraseBuffer.length === 0) return;
    this.onSentenceComplete(this.lineMessage());
    this.clearBuffer();
  }

  public getCurrentPhraseText(): string[] {
    const active = [...this.fullPhraseBuffer];
    if (this.currentWordBuffer.length > 0) {
      active.push(this.currentWordBuffer.join(''));
    }
    return active;
  }

  public clearBuffer(): void {
    this.currentWordBuffer = [];
    this.fullPhraseBuffer = [];
    this.lastSign = '';
    this.lastSignTimestamp = 0;
    this.onPhraseUpdate([]);
  }
}
