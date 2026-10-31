type FramingGuideProps = {
  onDone: () => void;
};

/** First-run camera framing. Drawings and text only. No sound. */
export function FramingGuide({ onDone }: FramingGuideProps) {
  return (
    <div className="framing-guide" role="dialog" aria-modal="true" aria-labelledby="framing-title">
      <div className="framing-card">
        <h2 id="framing-title">Set up the camera</h2>
        <p>You do not need sound. Follow the picture.</p>
        <div className="framing-stage" aria-hidden="true">
          <div className="framing-box">
            <span className="framing-face" />
            <span className="framing-hand" />
          </div>
          <div className="framing-light" />
        </div>
        <ol className="framing-steps">
          <li>Keep your face and one hand inside the bright box.</li>
          <li>Face a lamp or a window so the hand is easy to see.</li>
          <li>If a fingertip leaves the box, the caption says so. Move it back in.</li>
        </ol>
        <button id="framing-start" type="button" className="framing-start" onClick={onDone}>
          Start signing
        </button>
      </div>
    </div>
  );
}
