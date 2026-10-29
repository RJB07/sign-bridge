export type SignSample = {
  label: string;
  landmarks: number[];
};

export type SignDataset = SignSample[];

export function normalizeLandmarks(landmarks: Array<{ x: number; y: number; z: number }>): number[] {
  const wrist = landmarks[0];
  const centered = landmarks.map(({ x, y, z }) => [x - wrist.x, y - wrist.y, z - wrist.z]);
  const maxAbs = Math.max(1e-6, ...centered.flatMap(([x, y, z]) => [Math.abs(x), Math.abs(y), Math.abs(z)]));
  return centered.flatMap(([x, y, z]) => [x / maxAbs, y / maxAbs, z / maxAbs]);
}

export function exportDatasetJSON(dataset: SignDataset, filename = 'asl_signbridge_dataset.json'): void {
  const jsonStr = JSON.stringify(dataset, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function parseImportedDatasetJSON(jsonString: string): SignDataset {
  try {
    const parsed = JSON.parse(jsonString);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item) => typeof item.label === 'string' && Array.isArray(item.landmarks) && item.landmarks.length === 63,
    );
  } catch {
    return [];
  }
}
