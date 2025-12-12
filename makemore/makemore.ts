import fs from "node:fs";
import {
  arrange,
  multinomial,
  multiply,
  oneHot,
  randomNormal,
  softmax,
  sum,
  weightedRandomSample,
  zip,
} from "../matt-torch";
import { Tensor } from "../structures/Tensor";
import { Value } from "../structures";

const SPECIAL = ".";
const bigramKey = (a: string, b: string): string => `${a}:${b}`;
const bigramVals = (key: string): string[] => key.split(":");
const names = fs.readFileSync("./names.txt", "utf8");
const words = names.split("\n");

console.log(words.length);
// // build map of bigrams and their frequencies
const bigramCounter = new Map<string, number>();
for (const word of words) {
  const wordArr = [SPECIAL, ...word.split(""), SPECIAL];
  const comparator = zip(wordArr, wordArr.slice(1));
  for (const [a, b] of comparator) {
    const key = bigramKey(a, b);
    const val = bigramCounter.get(key) || 0;
    bigramCounter.set(key, val + 1);
  }
}

const uniqChars = [SPECIAL, ...Array.from(new Set(words.join(""))).sort()];
// console.log(uniqChars);
const stoi = (s: string) => uniqChars.indexOf(s);
const itos = (i: number) => uniqChars[i];
const MODEL_SMOOTHING_INT = 1;
// create a table matrix to represent the bigram counter map
// 28 by 28 tensor to represent every char plus start and end
// MODEL SMOOTHING: set fill to 1 instead of 0 to ensure log likelihood is never - infinity
// the more we start with, the smoother the model (this isn't necessarily good, smooth != low loss)
const bigramTensor = new Tensor([uniqChars.length, uniqChars.length] as const, MODEL_SMOOTHING_INT);

[...bigramCounter.entries()].map(([k, v]) => {
  const chars = bigramVals(k);
  const x = stoi(chars[0]);
  const y = stoi(chars[1]);

  bigramTensor.set([x, y], v);
});

// console.table(bigramTensor);
const normalizedTensor = bigramTensor.normalize();
// console.log(normalizedTensor)
// console.log(sum(normalizedTensor[0]))

// manually sample a full word from the distribution
for (let iter = 0; iter < 20; iter++) {
  let index = 0;
  let builtStr = [];
  while (true) {
    const normalizedP = normalizedTensor.vrow([index]).map((item) => item.data);

    index = weightedRandomSample(normalizedP, 1)[0];
    builtStr.push(itos(index));
    if (index === 0) {
      break;
    }
  }

  console.log(builtStr.join(""));
}
let counter = 0;
let log_likelihood = 0.0;
for (const word of words.slice(0,1)) {
  const wordArr = [SPECIAL, ...word.split(""), SPECIAL];
  const comparator = zip(wordArr, wordArr.slice(1));
  for (const [a, b] of comparator) {
    const idx1 = stoi(a);
    const idx2 = stoi(b);
    log_likelihood += Math.log(normalizedTensor.at([idx1, idx2]).data);
    counter++;
  }
}

// console.log(log_likelihood)
// const neg_log_likelihood = -log_likelihood;
// console.log('LOSS', neg_log_likelihood / counter)


// GOAL: maximize log likelihood which is the same as minimizing the negative log likelihood
// which is the same as minimizing the average negative log likelihood
// log(a+b+c) = log(a) + log(b) + log(c)

// CREATE TRAINING SET
// console.log(words.length);
// build map of bigrams and their frequencies
const xs: number[] = [];
const ys: number[] = [];
for (const word of words.slice(0, 50)) {
  const wordArr = [SPECIAL, ...word.split(""), SPECIAL];
  const comparator = zip(wordArr, wordArr.slice(1));
  for (const [a, b] of comparator) {
    xs.push(stoi(a));
    ys.push(stoi(b));
  }
}

let weights = new Tensor([uniqChars.length, uniqChars.length], () => randomNormal());
for (let i = 0; i < 100; i++) {
  console.time(`TRAINING_ITERATION_${i}`);
  // forward pass
  const xEncoded = Tensor.fromNestedArray([xs.length, uniqChars.length], oneHot(xs, uniqChars.length));
  // LOG COUNTS also called logits
  console.timeLog(`TRAINING_ITERATION_${i}`)
  const logits = multiply(xEncoded, weights);
  console.timeLog(`TRAINING_ITERATION_${i}`)
  // SOFTMAX - take an entry from a layer, exponentiate, and then normalize into a probability
  const sMax = softmax(logits); // this is ypred from micrograd\
  // sMax.show()
  const t = arrange(xs.length);
  const loss = t.map((item, _) => sMax.at([item.data, ys[item.data]]).log().negative());
  const weightLimiter = weights.map((item) => item.pow(2)).sum().divide(new Value(weights.size)).multiply(new Value(0.01));
  const avgLoss = loss.sum().divide(new Value(loss.size)).add(weightLimiter);
  console.log(`${i} loss: ${avgLoss.data}`);
  weights.zero_grad();
  avgLoss.backward();

  weights.forEach((item) => item.data += -15 * item.grad);
  console.timeEnd(`TRAINING_ITERATION_${i}`)
}

// SAMPLE from the model
// for (let i = 0; i < 5; i++) {
//   let result: string[] = [];
//   let idx = 0;
//   while (true) {
//     const xEncoded = Tensor.fromNestedArray([1, uniqChars.length], oneHot([idx], uniqChars.length));
//     // LOG COUNTS also called logits
//     const logits = multiply(xEncoded, weights);
//     // SOFTMAX - take an entry from a layer, exponentiate, and then normalize into a probability
//     const sMax = softmax(logits); // this is ypred from micrograd
//     const probabilities = sMax.row([0]);
//     idx = multinomial(probabilities, 1)[0];
//     result.push(itos(idx));
//     if (idx === 0) {
//       break;
//     }
//   }
//   console.log(result.join(''))
// }