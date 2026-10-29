import type { ConversationMessage } from '../types/signbridge';

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

    // Complete sentence flush with ASL Grammar Rule Translation & SL2T direct formatting
    if (elapsed > this.sentencePauseMs && this.fullPhraseBuffer.length > 0) {
      const { translatedText, ruleName } = this.applyASLGrammarRules(this.fullPhraseBuffer);
      
      const msg: ConversationMessage = {
        id: Math.random().toString(36).substring(2, 9),
        sender: 'SIGNER',
        text: translatedText,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        grammarRuleApplied: ruleName,
      };

      this.onSentenceComplete(msg);
      this.clearBuffer();
    }
  }

  /**
   * Light English formatting only. Isolated glosses are not a full ASL grammar translation.
   */
  private applyASLGrammarRules(tokens: string[]): { translatedText: string; ruleName: string } {
    if (tokens.length === 0) return { translatedText: '', ruleName: 'None' };

    const raw = tokens.join(' ');
    const whWords = ['who', 'what', 'when', 'where', 'why', 'which', 'how'];
    const isWhQuestion = whWords.includes(tokens[tokens.length - 1]?.toLowerCase());
    let result = raw.charAt(0).toUpperCase() + raw.slice(1);
    let ruleName = 'English formatting';
    if (isWhQuestion) {
      ruleName = 'Question mark';
      if (!result.endsWith('?')) result = `${result}?`;
    } else if (!/[.!?]$/.test(result)) {
      result += '.';
    }
    return { translatedText: result, ruleName };
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
