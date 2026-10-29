import type { SignDataset } from './dataset';

const DATASET_KEY = 'signbridge-dataset';
const MODEL_LABELS_KEY = 'signbridge-model-labels';

export function loadSavedDataset(): SignDataset {
  if (typeof window === 'undefined') {
    return [];
  }

  const raw = window.localStorage.getItem(DATASET_KEY);
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw) as SignDataset;
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveDataset(dataset: SignDataset): void {
  window.localStorage.setItem(DATASET_KEY, JSON.stringify(dataset));
}

export function clearSavedDataset(): void {
  window.localStorage.removeItem(DATASET_KEY);
  window.localStorage.removeItem(MODEL_LABELS_KEY);
}

export function saveModelLabels(labels: string[]): void {
  window.localStorage.setItem(MODEL_LABELS_KEY, JSON.stringify(labels));
}

export function loadModelLabels(): string[] {
  const raw = window.localStorage.getItem(MODEL_LABELS_KEY);
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw) as string[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}
