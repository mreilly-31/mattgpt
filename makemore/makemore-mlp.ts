import fs from "node:fs";
import { Tensor } from "../structures";
import { crossEntropy, multiply, randomNormal, shuffle } from "../matt-torch";

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
// parameters
const lookupTable = new Tensor([uniqChars.length, 2], () => randomNormal());
const weights = new Tensor([6, 100], () => randomNormal());
const biases = new Tensor([100], () => randomNormal());
const secondWeights = new Tensor([100, uniqChars.length], () => randomNormal());
const secondBiases = new Tensor([uniqChars.length], () => randomNormal());
const params = [lookupTable, weights, biases, secondWeights, secondBiases];
const numParams = params.reduce((acc, cur) => acc += cur.size, 0);
console.log(numParams)
console.log('STARTING TRAINNIG')
for (let i = 0; i < 100; i++) {
  // forward pass
  const embeddingLookup = lookupTable.gather(0, xDev).view(-1, 6);
  const hiddenLayer = multiply(embeddingLookup, weights)
  hiddenLayer.forEach((val, loc) => val.add(biases.at([loc[1]])).tanh());
  const logits = multiply(hiddenLayer, secondWeights);
  logits.forEach((val, loc) => val.add(secondBiases.at([loc[1]])));
  const loss = crossEntropy(logits, yDev);
  console.log(loss.data)
  // backward pass
  // zero grad
  for (const paramTensor of params) {
    paramTensor.zero_grad();
  }

  loss.backward();

  // update
  for (const paramTensor of params) {
    paramTensor.forEach(item => item.data += -0.1 * item.grad);
  }
}





// 2.642340480827716