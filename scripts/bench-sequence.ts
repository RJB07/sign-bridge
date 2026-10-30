/**
 * Time the browser recognizer. One preview pass is a single member and no
 * mirror. The commit pass uses every member and the model's mirror flag.
 */
import { readFileSync } from 'node:fs';
import { modelFromJson, predictSequence } from '../src/recognition/sequenceModel';

const modelPath = process.argv[2];
const runs = Number(process.argv[3] ?? 20);
if (!modelPath) {
  throw new Error('usage: bench-sequence <model.json> [runs]');
}
const model = modelFromJson(JSON.parse(readFileSync(modelPath, 'utf8')));
const sequence = Array.from({ length: model.seqLen }, (_, time) =>
  Array.from({ length: model.featureDim }, (_, dim) => ((time + 1) * (dim + 1) % 17) / 50),
);
predictSequence(model, sequence, { members: 1, tta: false });
predictSequence(model, sequence);

function millis(body: () => void, count: number): number {
  const start = performance.now();
  for (let index = 0; index < count; index += 1) body();
  return (performance.now() - start) / count;
}

const previewMs = millis(() => predictSequence(model, sequence, { members: 1, tta: false }), runs);
const commitMs = millis(() => predictSequence(model, sequence), runs);
console.log(JSON.stringify({
  previewMs: Number(previewMs.toFixed(2)),
  commitMs: Number(commitMs.toFixed(2)),
  members: model.members.length,
  layers: model.layers,
  dModel: model.dModel,
  featureDim: model.featureDim,
  featureLayout: model.featureLayout,
  tta: model.tta,
}));
