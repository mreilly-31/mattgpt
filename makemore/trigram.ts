import fs from "node:fs";
import { weightedRandomSample } from "../matt-torch";
import { normalize, tensor } from "../structures/Tensor";

const SPECIAL = ".";
const START_CONTEXT = `${SPECIAL}${SPECIAL}`;
const trigramKey = (a: string, b: string, c: string): string => `${a}${b}:${c}`;
const names = fs.readFileSync("./names.txt", "utf8");
const words = names.split("\n");
console.log(words.length);
// // build map of bigrams and their frequencies
const trigramCounter = new Map<string, number>();
for (const word of words) {
  const wordArr = [SPECIAL, SPECIAL, ...word.split(""), SPECIAL, SPECIAL];
  for (let i = 0; i < wordArr.length - 2; i++) {
    const key = trigramKey(wordArr[i], wordArr[i + 1], wordArr[i + 2]);
    const val = trigramCounter.get(key) || 0;
    trigramCounter.set(key, val + 1);
  }
}

const uniqChars = [SPECIAL, ...Array.from(new Set(words.join(""))).sort()];
const uniqTokens = uniqChars.flatMap(a => uniqChars.map(b => `${a}${b}`));
const k_to_idx = (k: string) =>  uniqTokens.indexOf(k);
const v_to_idx = (v: string) => uniqChars.indexOf(v);
const itos = (i: number) => uniqChars[i];

const MODEL_SMOOTHING_INT = 1;
const trigramTensor = tensor([uniqTokens.length, uniqChars.length] as const, MODEL_SMOOTHING_INT);

[...trigramCounter.entries()].map(([k, v]) => {
  const chars = k.split(':');
  const x = k_to_idx(chars[0]);
  const y = v_to_idx(chars[1]);
  trigramTensor[x][y] = v;
});

const normalizedTensor = normalize<[number, number]>(trigramTensor);

let counter = 0;
let log_likelihood = 0.0;
for (const word of words) {
  const wordArr = [SPECIAL, SPECIAL, ...word.split(""), SPECIAL, SPECIAL];
  for (let i = 0; i < wordArr.length - 2; i++) {
    const idx1 = k_to_idx(`${wordArr[i]}${wordArr[i + 1]}`);
    const idx2 = v_to_idx(wordArr[i + 2]);

    log_likelihood += Math.log(normalizedTensor[idx1][idx2]);
    counter++;
  }
}

console.log(log_likelihood)
const neg_log_likelihood = -log_likelihood;
console.log(neg_log_likelihood / counter)

// manually sample a full word from the distribution
for (let iter = 0; iter < 20; iter++) {
  let context = START_CONTEXT;
  let builtStr: string[] = [];
  while (true) {
    const contextIdx = k_to_idx(context);
    const normalizedP = normalizedTensor[contextIdx];
    const nextIndex = weightedRandomSample(normalizedP, 1)[0];
    const nextChar = itos(nextIndex);
    if (nextChar === SPECIAL) {
      break;
    }
    builtStr.push(nextChar);
    context = `${context.slice(1)}${nextChar}`;
  }

  console.log(builtStr.join(""));
}


// // GOAL: maximize log likelihood which is the same as minimizing the negative log likelihood
// // which is the same as minimizing the average negative log likelihood
// // log(a+b+c) = log(a) + log(b) + log(c)
