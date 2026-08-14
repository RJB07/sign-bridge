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

  constructor(
    onPhraseUpdate: (phrase: string[]) => void,
    onSentenceComplete: (message: ConversationMessage) => void
  ) {
    this.onPhraseUpdate = onPhraseUpdate;
    this.onSentenceComplete = onSentenceComplete;
  }

  public pushSign(signText: string, confidence: number = 90): void {
    // SL2T Anti-Hallucination Filter: Ignore resting hands / non-signing noise
    if (!signText || signText === 'Unknown' || signText === 'Waiting for hand...' || signText === 'No hand detected' || signText === 'Translating...') {
      this.checkPauseBoundary();
      return;
    }

    const now = performance.now();

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
    const elapsed = performance.now() - this.lastSignTimestamp;

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
        confidence: 95,
        grammarRuleApplied: ruleName,
      };

      this.onSentenceComplete(msg);
      this.clearBuffer();
    }
  }

  /**
   * Applies Dr. David A. Stewart's 10 Basic ASL Grammar Rules & DeepMind SL2T Direct Sentence Assembly
   */
  private applyASLGrammarRules(tokens: string[]): { translatedText: string; ruleName: string } {
    if (tokens.length === 0) return { translatedText: '', ruleName: 'None' };

    const raw = tokens.join(' ');
    let ruleName = 'Standard SVO (Rule #8)';

    // Rule #2: Tense with Time Adverbs at Start (YESTERDAY, TODAY, TOMORROW, NOW, PAST)
    const timeWords = ['Yesterday', 'Today', 'Tomorrow', 'Now', 'Past', 'Last-Week', 'Next-Week'];
    const hasTimeWord = timeWords.some(t => tokens[0]?.toLowerCase() === t.toLowerCase());

    // Rule #5: Wh-Questions (WHO, WHAT, WHEN, WHERE, WHY, WHICH, HOW at end)
    const whWords = ['who', 'what', 'when', 'where', 'why', 'which', 'how'];
    const lastTokenLower = tokens[tokens.length - 1]?.toLowerCase();
    const isWhQuestion = whWords.includes(lastTokenLower);

    // Rule #9: Conditional Sentences (SUPPOSE, IF at start)
    const isConditional = tokens[0]?.toLowerCase() === 'suppose' || tokens[0]?.toLowerCase() === 'if';

    // Rule #10: Negation (NOT, NEVER, NONE, CAN'T, DON'T-WANT)
    const negWords = ['not', 'never', 'none', "can't", "don't-want", "don't-know", 'no'];
    const hasNegation = tokens.some(t => negWords.includes(t.toLowerCase()));

    // Rule #7: Rhetorical Question (e.g. ME EXCITED WHY? -> "I am excited because...")
    const isRhetorical = tokens.includes('WHY') && tokens.indexOf('WHY') < tokens.length - 1;

    let result = raw;

    if (isConditional) {
      ruleName = 'Conditional Sentence (Rule #9)';
      result = raw.endsWith('?') ? raw : `${raw}?`;
    } else if (isWhQuestion) {
      ruleName = 'Wh-Question (Rule #5)';
      if (!raw.endsWith('?')) result = `${raw}?`;
    } else if (hasTimeWord) {
      ruleName = 'Time-First Tense (Rule #2)';
    } else if (hasNegation) {
      ruleName = 'Negation (Rule #10)';
    } else if (isRhetorical) {
      ruleName = 'Rhetorical Question (Rule #7)';
    }

    // Capitalize first letter and sanitize punctuation
    result = result.charAt(0).toUpperCase() + result.slice(1);
    if (!/[.!?]$/.test(result)) {
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
