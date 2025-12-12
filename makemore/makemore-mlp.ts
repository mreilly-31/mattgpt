import fs from "node:fs";
import { Tensor } from "../structures";
import { multiply, oneHot, randomNormal, softmax } from "../matt-torch";

const names = fs.readFileSync("./names.txt", "utf8");
const words = names.split("\n");
console.log(words.length + ' words');
const SPECIAL = ".";
const uniqChars = [SPECIAL, ...Array.from(new Set(words.join(""))).sort()];
console.log(uniqChars.length + ' unique characters');
const stoi = (s: string) => uniqChars.indexOf(s);
const itos = (i: number) => uniqChars[i];

const BLOCK_SIZE = 3;

let X: number[][] = [];
let Y: number[] = [];

for (const word of words.slice(0,5)) {
  // console.log(word);
  let context: number[] = new Array(BLOCK_SIZE).fill(0);
  const chars = word.split('').concat([SPECIAL]);
  for (const char of chars) {
    const idx = stoi(char);
    X.push(context);
    Y.push(idx);
    // console.log(`${context.map(item => itos(item)).join('')} ---> ${itos(idx)}`);
    context = context.slice(1).concat([idx]);
  }
}


const xT = Tensor.fromNestedArray([X.length, BLOCK_SIZE], X);
console.log('XT Shape ' + xT.shape())
const yT = Tensor.fromNestedArray([Y.length], Y);
// yT.show()

const lookupTable = new Tensor([uniqChars.length, 2], () => randomNormal());
// lookupTable.show()

const oneHotT = Tensor.fromNestedArray(
  [1, uniqChars.length],
  oneHot([5], uniqChars.length)
);
// oneHotT.show()
console.log('================')
const product = multiply(oneHotT, lookupTable);
product.show()
const r = lookupTable.row([5]);
console.log(r)

const sub = lookupTable.slice(0, { start: 5, end: 8 });
sub.show();

const embeddingLookup = lookupTable.gather(0, xT);
console.log('Embedding lookup shape', embeddingLookup.shape());

const contextBlocks = embeddingLookup.unbind(1);
contextBlocks.forEach((block, idx) => {
  console.log(`Block ${idx} shape`, block.shape());
});

const flattenedEmbeddings = embeddingLookup.view(
  embeddingLookup.dims[0] * embeddingLookup.dims[1],
  embeddingLookup.dims[2]
);
console.log('Flattened embedding shape', flattenedEmbeddings.shape());
const temp = embeddingLookup.row([13, 2]);
// console.log(temp)
// console.log(lookupTable.row([1]))

const weights = new Tensor([6, 100], () => randomNormal());
const biases = new Tensor([100], () => randomNormal());
const newView = embeddingLookup.view(-1, 6);

const hiddenLayer = multiply(newView, weights).map((val, loc) => val.add(biases.at([loc[1]]))).map((val) => val.tanh());

hiddenLayer.show()

const secondWeights = new Tensor([100, uniqChars.length], () => randomNormal());
const secondBiases = new Tensor([uniqChars.length], () => randomNormal());

const logits = multiply(hiddenLayer, secondWeights).map((val, loc) => val.add(secondBiases.at([loc[1]])));
const sMax = softmax(logits)
sMax.shape()