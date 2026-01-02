import fs from "node:fs";
import { Tensor, Tape } from "../structures/Tensor";

const shuffle = (array: string[]): string[] => {
  let result = array;
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
};

const randomNormal = (): number => {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

const names = fs.readFileSync("./names.txt", "utf8");
const words = shuffle(names.split("\n"));
console.log(words.length + " words");
const SPECIAL = ".";
const uniqChars = [SPECIAL, ...Array.from(new Set(words.join(""))).sort()];
console.log(uniqChars.length + " unique characters");
const stoi = (s: string) => uniqChars.indexOf(s);
const itos = (i: number) => uniqChars[i];

const BLOCK_SIZE = 3;
type DataSet = [Tensor, Tensor];
const flatten2D = (values: number[][]): number[] => {
  const flattened = new Array(values.length * values[0].length);
  let cursor = 0;
  for (const row of values) {
    for (const value of row) {
      flattened[cursor++] = value;
    }
  }
  return flattened;
};

function buildDataset(data: string[], block_size: number): DataSet {
  const x: number[][] = [];
  const y: number[] = [];
  for (const item of data) {
    let context: number[] = new Array(BLOCK_SIZE).fill(0);
    const chars = item.split("").concat([SPECIAL]);
    for (const char of chars) {
      const idx = stoi(char);
      x.push(context);
      y.push(idx);
      context = context.slice(1).concat([idx]);
    }
  }
  const xTensor = Tensor.fromArray([x.length, block_size], flatten2D(x));
  const yTensor = Tensor.fromArray([y.length], y);
  return [xTensor, yTensor];
}

const [xTrain, yTrain] = buildDataset(
  words.slice(0, 0.8 * words.length),
  BLOCK_SIZE
);
const [xDev, yDev] = buildDataset(
  words.slice(0.8 * words.length, 0.9 * words.length),
  BLOCK_SIZE
);
const [xTest, yTest] = buildDataset(
  words.slice(0.9 * words.length, words.length),
  BLOCK_SIZE
);
console.log(xTrain.shape, xDev.shape, xTest.shape);

const trainX = xTrain;
const trainY = yTrain;
const valX = xDev;
const valY = yDev;
const shouldTrain = process.argv.includes("--train");
const shouldSample = process.argv.some((arg) => arg.startsWith("--sample"));
const shouldUseWasm = process.argv.includes("--wasm");
const sampleArg = process.argv.find((arg) => arg.startsWith("--sample="));
const parsedSampleCount = sampleArg ? Number(sampleArg.split("=")[1]) : NaN;
const sampleCount = Number.isFinite(parsedSampleCount) ? parsedSampleCount : 5;
const modelPath = "model.json";

if (shouldUseWasm) {
  process.env.TENSOR_WASM_MATMUL = "1";
}

type TensorPayload = { shape: number[]; data: number[] };

type ModelPayload = {
  lookupTable: TensorPayload;
  weights: TensorPayload;
  biases: TensorPayload;
  secondWeights: TensorPayload;
  secondBiases: TensorPayload;
};

type ModelParams = {
  lookupTable: Tensor;
  weights: Tensor;
  biases: Tensor;
  secondWeights: Tensor;
  secondBiases: Tensor;
};

const tensorToJSON = (tensor: Tensor): TensorPayload => ({
  shape: [...tensor.shape],
  data: Array.from(tensor.data)
});

const tensorFromJSON = (payload: TensorPayload, tape: Tape): Tensor =>
  Tensor.fromArray(payload.shape, payload.data, true, tape);

const createParam = (shape: number[], tape: Tape): Tensor => {
  const size = shape.reduce((acc, cur) => acc * cur, 1);
  const data = new Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = randomNormal();
  }
  return Tensor.fromArray(shape, data, true, tape);
};

const loadModel = (path: string, tape: Tape): ModelParams => {
  const payload = JSON.parse(fs.readFileSync(path, "utf8")) as ModelPayload;
  return {
    lookupTable: tensorFromJSON(payload.lookupTable, tape),
    weights: tensorFromJSON(payload.weights, tape),
    biases: tensorFromJSON(payload.biases, tape),
    secondWeights: tensorFromJSON(payload.secondWeights, tape),
    secondBiases: tensorFromJSON(payload.secondBiases, tape)
  };
};

const tape = new Tape();
const initialParams = fs.existsSync(modelPath)
  ? loadModel(modelPath, tape)
  : {
      lookupTable: createParam([uniqChars.length, 10], tape),
      weights: createParam([30, 200], tape),
      biases: createParam([200], tape),
      secondWeights: createParam([200, uniqChars.length], tape),
      secondBiases: createParam([uniqChars.length], tape)
    };
const { lookupTable, weights, biases, secondWeights, secondBiases } =
  initialParams;
const params = [lookupTable, weights, biases, secondWeights, secondBiases];
const numParams = params.reduce((acc, cur) => (acc += cur.size), 0);
console.log(`${numParams} params`);

const detach = (tensor: Tensor): Tensor =>
  new Tensor(tensor.shape, {
    data: tensor.data,
    requiresGrad: false,
    reuseData: true
  });

const detachParams = (p: ModelParams): ModelParams => ({
  lookupTable: detach(p.lookupTable),
  weights: detach(p.weights),
  biases: detach(p.biases),
  secondWeights: detach(p.secondWeights),
  secondBiases: detach(p.secondBiases)
});

const getBatch = (
  x: Tensor,
  y: Tensor,
  batchSize: number
): [Tensor, Tensor] => {
  const size = x.shape[0];
  const indices: number[] = new Array(batchSize);
  for (let i = 0; i < batchSize; i++) {
    indices[i] = Math.floor(Math.random() * size);
  }
  const indexTensor = Tensor.fromArray([batchSize], indices);
  return [x.gather(0, indexTensor), y.gather(0, indexTensor)];
};

const oneHot = (indices: Tensor, numClasses: number): Tensor => {
  const batch = indices.size;
  const data = new Float32Array(batch * numClasses);
  for (let i = 0; i < batch; i++) {
    const idx = indices.data[i];
    if (!Number.isInteger(idx)) {
      throw new Error(`Target index ${idx} is not an integer`);
    }
    if (idx < 0 || idx >= numClasses) {
      throw new Error(`Target index ${idx} is out of bounds`);
    }
    data[i * numClasses + idx] = 1;
  }
  return new Tensor([batch, numClasses], {
    data,
    reuseData: true
  });
};

const forwardLogits = (x: Tensor, p: ModelParams): Tensor => {
  const embeddingLookup = p.lookupTable.gather(0, x);
  const batch = x.shape[0];
  const embeddingFlat = embeddingLookup.view([batch, BLOCK_SIZE * 10]);
  const hiddenLayer = embeddingFlat.matmul(p.weights).add(p.biases).tanh();
  return hiddenLayer.matmul(p.secondWeights).add(p.secondBiases);
};

const crossEntropy = (logits: Tensor, target: Tensor): Tensor => {
  const logProbs = logits.logsoftmax(1);
  const targetOneHot = oneHot(target, logits.shape[1]);
  const picked = logProbs.mul(targetOneHot).sum(1);
  return picked.sum().mulScalar(-1 / logits.shape[0]);
};

const forward = (x: Tensor, y: Tensor, p: ModelParams) => {
  const logits = forwardLogits(x, p);
  return crossEntropy(logits, y);
};

const batchSize = 256;
const iters = shouldTrain ? 10000 : 0;
for (let i = 0; i < iters; i++) {
  const [batchX, batchY] = getBatch(trainX, trainY, batchSize);
  const loss = forward(batchX, batchY, initialParams);

  for (const paramTensor of params) {
    paramTensor.zeroGrad();
  }

  loss.backward();

  for (const paramTensor of params) {
    for (let j = 0; j < paramTensor.size; j++) {
      paramTensor.data[j] += -0.01 * paramTensor.grad[j];
    }
  }

  tape.clear();

  if ((i + 1) % 100 === 0) {
    const evalParams = detachParams(initialParams);
    const valLoss = forward(valX, valY, evalParams);
    console.log(`${i} train ${loss.data[0]} dev ${valLoss.data[0]}`);
  }
}

const sampleNextIndex = (probs: Tensor): number => {
  const vocab = probs.shape[probs.shape.length - 1];
  const offset = 0;
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < vocab; i++) {
    cumulative += probs.data[offset + i];
    if (r <= cumulative) {
      return i;
    }
  }
  return vocab - 1;
};

const sampleWords = (count: number) => {
  const evalParams = detachParams(initialParams);
  for (let n = 0; n < count; n++) {
    let context = new Array(BLOCK_SIZE).fill(0);
    let word = "";
    while (true) {
      const x = Tensor.fromArray([1, BLOCK_SIZE], context);
      const logits = forwardLogits(x, evalParams);
      const probs = logits.softmax(1);
      const idx = sampleNextIndex(probs);
      if (idx === 0) break;
      word += itos(idx);
      context = context.slice(1).concat([idx]);
    }
    console.log(word);
  }
};

if (shouldTrain) {
  const modelPayload: ModelPayload = {
    lookupTable: tensorToJSON(lookupTable),
    weights: tensorToJSON(weights),
    biases: tensorToJSON(biases),
    secondWeights: tensorToJSON(secondWeights),
    secondBiases: tensorToJSON(secondBiases)
  };
  fs.writeFileSync(modelPath, JSON.stringify(modelPayload));
}

if (shouldSample) {
  sampleWords(sampleCount);
}

// 2.642340480827716
