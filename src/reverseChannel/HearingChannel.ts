import type { ConversationMessage } from '../types/signbridge';

export class HearingChannel {
  private recognitionInstance: any = null;
  private isListening = false;
  private onMessageCallback: (message: ConversationMessage) => void;
  private onStatusCallback: (listening: boolean, text?: string) => void;

  constructor(
    onMessageCallback: (message: ConversationMessage) => void,
    onStatusCallback: (listening: boolean, text?: string) => void
  ) {
    this.onMessageCallback = onMessageCallback;
    this.onStatusCallback = onStatusCallback;
    this.initBrowserSTT();
  }

  private initBrowserSTT(): void {
    const SpeechRecognitionClass =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognitionClass) {
      this.onStatusCallback(false, 'Browser Speech Recognition not supported. Keyboard input available.');
      return;
    }

    this.recognitionInstance = new SpeechRecognitionClass();
    this.recognitionInstance.continuous = true;
    this.recognitionInstance.interimResults = true;
    this.recognitionInstance.lang = 'en-US';

    this.recognitionInstance.onresult = (event: any) => {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      if (interimTranscript) {
        this.onStatusCallback(true, interimTranscript);
      }

      if (finalTranscript.trim()) {
        this.sendHearingMessage(finalTranscript.trim());
      }
    };

    this.recognitionInstance.onerror = (event: any) => {
      this.isListening = false;
      this.onStatusCallback(false, `Speech error: ${event.error}`);
    };

    this.recognitionInstance.onend = () => {
      if (this.isListening) {
        try {
          this.recognitionInstance.start();
        } catch {
          this.isListening = false;
          this.onStatusCallback(false);
        }
      } else {
        this.onStatusCallback(false);
      }
    };
  }

  public toggleMicrophone(): 'on' | 'off' | 'unsupported' {
    if (!this.recognitionInstance) {
      this.onStatusCallback(false, 'Speech to text is not available in this browser. Type instead.');
      return 'unsupported';
    }

    if (this.isListening) {
      this.isListening = false;
      this.recognitionInstance.stop();
      this.onStatusCallback(false);
      return 'off';
    } else {
      try {
        this.isListening = true;
        this.recognitionInstance.start();
        this.onStatusCallback(true, '');
        return 'on';
      } catch {
        this.isListening = false;
        this.onStatusCallback(false);
        return 'off';
      }
    }
  }

  public sendTypedReply(typedText: string): void {
    if (!typedText.trim()) return;
    this.sendHearingMessage(typedText.trim());
  }

  private sendHearingMessage(text: string): void {
    const msg: ConversationMessage = {
      id: Math.random().toString(36).substring(2, 9),
      sender: 'HEARING',
      text: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    };

    this.onMessageCallback(msg);
    this.onStatusCallback(this.isListening, '');
  }
}
