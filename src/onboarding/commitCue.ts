/** A short vibration when the browser offers it. The yellow flash is the cue that always runs. */
export function commitCue(): void {
  try {
    navigator.vibrate?.(40);
  } catch {
    /* A detached vibrate call throws. The flash still shows. */
  }
}
