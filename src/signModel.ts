import * as tf from '@tensorflow/tfjs';

export type SignSample = {
  label: string;
  landmarks: number[];
};

export type SignDataset = SignSample[];

export function normalizeLandmarks(
  landmarks: Array<{ x: number; y: number; z: number }>
): number[] {
  const wrist = landmarks[0];
  const centered = landmarks.map(({ x, y, z }) => [x - wrist.x, y - wrist.y, z - wrist.z]);
  const maxAbs = Math.max(
    1e-6,
    ...centered.flatMap(([x, y, z]) => [Math.abs(x), Math.abs(y), Math.abs(z)])
  );
  return centered.flatMap(([x, y, z]) => [x / maxAbs, y / maxAbs, z / maxAbs]);
}

export function buildSignModel(numClasses: number): tf.Sequential {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [63],
      units: 128,
      activation: 'relu',
    })
  );
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(
    tf.layers.dense({
      units: 64,
      activation: 'relu',
    })
  );
  model.add(
    tf.layers.dense({
      units: numClasses,
      activation: 'softmax',
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

export function createDatasetTensors(
  dataset: SignDataset,
  labelMap: string[]
): { xs: tf.Tensor2D; ys: tf.Tensor2D } {
  const xs = tf.tensor2d(dataset.map((sample) => sample.landmarks));
  const ys = tf.tensor2d(
    dataset.map((sample) => {
      const labelVector = new Array(labelMap.length).fill(0);
      const index = labelMap.indexOf(sample.label);
      if (index >= 0) {
        labelVector[index] = 1;
      }
      return labelVector;
    })
  );
  return { xs, ys };
}

export async function trainSignModel(
  model: tf.Sequential,
  dataset: SignDataset,
  labelMap: string[],
  epochs = 30,
  batchSize = 16,
  onBatchEnd?: (batch: number, logs?: tf.Logs) => void
): Promise<tf.History> {
  const { xs, ys } = createDatasetTensors(dataset, labelMap);

  const history = await model.fit(xs, ys, {
    epochs,
    batchSize,
    shuffle: true,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        if (onBatchEnd) {
          onBatchEnd(batch, logs);
        }
        await tf.nextFrame();
      },
    },
  });

  xs.dispose();
  ys.dispose();
  return history;
}

export function predictLabelSync(
  model: tf.LayersModel,
  landmarks: number[],
  labelMap: string[]
): string {
  if (!model || labelMap.length === 0) {
    return 'Unknown';
  }
  const input = tf.tensor2d([landmarks]);
  const output = model.predict(input) as tf.Tensor;
  const scores = output.dataSync();
  input.dispose();
  output.dispose();

  const topIndex = scores.indexOf(Math.max(...scores));
  return labelMap[topIndex] ?? 'Unknown';
}

export async function saveModelToStorage(
  model: tf.LayersModel,
  modelName: string
): Promise<tf.io.SaveResult> {
  return model.save(`localstorage://${modelName}`);
}

export async function loadModelFromStorage(
  modelName: string
): Promise<tf.LayersModel> {
  return tf.loadLayersModel(`localstorage://${modelName}`);
}

export async function deleteModelFromStorage(
  modelName: string
): Promise<void> {
  await tf.io.removeModel(`localstorage://${modelName}`);
}
