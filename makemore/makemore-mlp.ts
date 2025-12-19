import fs from "node:fs";
import { Tensor, Value } from "../structures";
import { crossEntropy, multiply, randomNormal, shuffle, softmax } from "../matt-torch";

const names = fs.readFileSync("./names.txt", "utf8");
const words = shuffle(names.split("\n"));
console.log(words.length + ' words');
const SPECIAL = ".";
const uniqChars = [SPECIAL, ...Array.from(new Set(words.join(""))).sort()];
console.log(uniqChars.length + ' unique characters');
const stoi = (s: string) => uniqChars.indexOf(s);
const itos = (i: number) => uniqChars[i];

const BLOCK_SIZE = 3;
type DataSet = [Tensor, Tensor];
function buildDataset(data: string[], block_size: number): DataSet {
  let x: number[][] = [];
  let y: number[] = [];
  for (const item of data) {
    let context: number[] = new Array(BLOCK_SIZE).fill(0);
    const chars = item.split('').concat([SPECIAL]);
    for (const char of chars) {
      const idx = stoi(char);
      x.push(context);
      y.push(idx);
      context = context.slice(1).concat([idx]);
    }
  }
  const xTensor = Tensor.fromNestedArray([x.length, block_size], x);
  const yTensor = Tensor.fromNestedArray([y.length], y);
  return [ xTensor, yTensor];
}

const [xTrain, yTrain] = buildDataset(words.slice(0, 0.8 * words.length), BLOCK_SIZE);
const [xDev, yDev] = buildDataset(words.slice(0.8 * words.length, 0.9 * words.length), BLOCK_SIZE);
const [xTest, yTest] = buildDataset(words.slice(0.9 * words.length, words.length), BLOCK_SIZE);
xTrain.shape()
xDev.shape()
xTest.shape()

const trainX = xTrain;
const trainY = yTrain;
const valX = xDev;
const valY = yDev;
const shouldTrain = process.argv.includes("--train");
const shouldSample = process.argv.some((arg) => arg.startsWith("--sample"));
const sampleArg = process.argv.find((arg) => arg.startsWith("--sample="));
const parsedSampleCount = sampleArg ? Number(sampleArg.split("=")[1]) : NaN;
const sampleCount = Number.isFinite(parsedSampleCount) ? parsedSampleCount : 5;
const modelPath = "model.json";
const loadModel = (path: string) => {
  const payload = JSON.parse(fs.readFileSync(path, "utf8"));
  return {
    lookupTable: Tensor.fromJSON(payload.lookupTable),
    weights: Tensor.fromJSON(payload.weights),
    biases: Tensor.fromJSON(payload.biases),
    secondWeights: Tensor.fromJSON(payload.secondWeights),
    secondBiases: Tensor.fromJSON(payload.secondBiases),
  };
};

// parameters
const initialParams = fs.existsSync(modelPath)
  ? loadModel(modelPath)
  : {
      lookupTable: new Tensor([uniqChars.length, 10], () => randomNormal()),
      weights: new Tensor([30, 200], () => randomNormal()),
      biases: new Tensor([200], () => randomNormal()),
      secondWeights: new Tensor([200, uniqChars.length], () => randomNormal()),
      secondBiases: new Tensor([uniqChars.length], () => randomNormal()),
    };
const { lookupTable, weights, biases, secondWeights, secondBiases } = initialParams;
const params = [lookupTable, weights, biases, secondWeights, secondBiases];
const numParams = params.reduce((acc, cur) => acc += cur.size, 0);
console.log(`${numParams} params`)

const getBatch = (x: Tensor, y: Tensor, batchSize: number): [Tensor, Tensor] => {
  const size = x.dims[0];
  const indices: number[] = new Array(batchSize);
  for (let i = 0; i < batchSize; i++) {
    indices[i] = Math.floor(Math.random() * size);
  }
  const indexTensor = Tensor.fromNestedArray([batchSize], indices);
  return [x.gather(0, indexTensor), y.gather(0, indexTensor)];
};

const forwardLogits = (x: Tensor) => {
  const embeddingLookup = lookupTable.gather(0, x).view(-1, 30);
  const hiddenLayer = multiply(embeddingLookup, weights)
  hiddenLayer.forEach((val, loc) => val.add(biases.at([loc[1]])).tanh());
  const logits = multiply(hiddenLayer, secondWeights);
  logits.forEach((val, loc) => val.add(secondBiases.at([loc[1]])));
  return logits;
};

const forward = (x: Tensor, y: Tensor) => {
  const logits = forwardLogits(x);
  return crossEntropy(logits, y);
};

const batchSize = 256;
const iters = shouldTrain ? 10000 : 0;
for (let i = 0; i < iters; i++) {
  // forward pass
  const [batchX, batchY] = getBatch(trainX, trainY, batchSize);
  const loss = forward(batchX, batchY);
  // console.log(loss.data)
  // backward pass
  // zero grad
  for (const paramTensor of params) {
    paramTensor.zero_grad();
  }

  loss.backward();

  // update
  for (const paramTensor of params) {
    paramTensor.forEach(item => item.data += -0.03 * item.grad);
  }

  if ((i + 1) % 100 === 0) {
    const valLoss = Value.withNoGrad(() => forward(valX, valY));
    console.log(`${i} train ${loss.data} dev ${valLoss.data}`);
  }
}

const sampleNextIndex = (probs: Tensor): number => {
  const row = probs.vrow([0]);
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < row.length; i++) {
    cumulative += row[i].data;
    if (r <= cumulative) {
      return i;
    }
  }
  return row.length - 1;
};

const sampleWords = (count: number) => {
  for (let n = 0; n < count; n++) {
    let context = new Array(BLOCK_SIZE).fill(0);
    let word = "";
    while (true) {
      const x = Tensor.fromNestedArray([1, BLOCK_SIZE], [context]);
      const idx = Value.withNoGrad(() => {
        const logits = forwardLogits(x);
        const probs = softmax(logits);
        return sampleNextIndex(probs);
      });
      if (idx === 0) break;
      word += itos(idx);
      context = context.slice(1).concat([idx]);
    }
    console.log(word);
  }
};

if (shouldTrain) {
  const modelPayload = {
    lookupTable: lookupTable.toJSON(),
    weights: weights.toJSON(),
    biases: biases.toJSON(),
    secondWeights: secondWeights.toJSON(),
    secondBiases: secondBiases.toJSON(),
  };
  fs.writeFileSync("model.json", JSON.stringify(modelPayload));
}

if (shouldSample) {
  sampleWords(sampleCount);
}




// 2.642340480827716
