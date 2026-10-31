export type CaptionSize = 'sm' | 'md' | 'lg' | 'xl';

const SIZES: CaptionSize[] = ['sm', 'md', 'lg', 'xl'];

/** Phones open on the largest caption step unless the signer already chose one. */
export function defaultCaptionSize(width: number, saved: string | null): CaptionSize {
  if (saved === 'sm' || saved === 'md' || saved === 'lg' || saved === 'xl') return saved;
  return width <= 720 ? 'xl' : 'lg';
}

export function captionSizeLabel(size: CaptionSize): string {
  return SIZES.includes(size) ? size : 'lg';
}
