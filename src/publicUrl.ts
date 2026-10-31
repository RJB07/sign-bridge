/** Prefix a public file with Vite's base, so GitHub Pages and local preview both resolve. */
export function publicUrl(path: string): string {
  const base = import.meta.env.BASE_URL || '/';
  return `${base}${path.replace(/^\//, '')}`;
}
