import fs from "node:fs";
import { BPETokenizer } from "./tokenizers/BPETokenizer";
import { MattGPT, Tape, Tensor } from "./structures";
import { crossEntropy } from "./matt-torch";

const dataPath = "shakespeare.txt";
const vocabSize = 256;
const contextLength = 64;
const batchSize = 16;
const baseSteps = 2000;
const lr = 1e-3;
const overfit = process.argv.includes("--overfit");
const debugStats = process.argv.includes("--debug-stats") || overfit;
const debugTokenizer = process.argv.includes("--debug-tokenizer");
const maxSteps = overfit ? 500 : baseSteps;

console.log("[setup] loading dataset");
const rawText = fs.readFileSync(dataPath, "utf8");

const tokenizerPath = "checkpoints/tokenizer.json";
let tokenizer: BPETokenizer;
if (fs.existsSync(tokenizerPath)) {
  console.log("[tokenizer] loading saved tokenizer");
  const saved = JSON.parse(fs.readFileSync(tokenizerPath, "utf8"));
  tokenizer = BPETokenizer.fromState(saved);
} else {
  console.log("[tokenizer] training");
  tokenizer = new BPETokenizer();
  tokenizer.train(rawText, vocabSize);
  fs.mkdirSync("checkpoints", { recursive: true });
  fs.writeFileSync(tokenizerPath, JSON.stringify(tokenizer.getState()));
}

const encodedPath = "checkpoints/shakespeare.tokens.json";
let encoded: number[];
if (fs.existsSync(encodedPath)) {
  console.log("[tokenizer] loading encoded tokens");
  encoded = JSON.parse(fs.readFileSync(encodedPath, "utf8"));
} else {
  console.log("[tokenizer] encoding dataset");
  encoded = tokenizer.encode(rawText);
  fs.mkdirSync("checkpoints", { recursive: true });
  fs.writeFileSync(encodedPath, JSON.stringify(encoded));
}
console.log(`[tokenizer] encoded length=${encoded.length}`);
if (debugTokenizer) {
  console.log("[tokenizer] debug checks");
  let minId = Infinity;
  let maxId = -Infinity;
  let outOfRange = 0;
  let nonInt = 0;
  const freqs = new Array(vocabSize).fill(0);
  for (let i = 0; i < encoded.length; i++) {
    const id = encoded[i];
    if (!Number.isInteger(id)) {
      nonInt += 1;
      continue;
    }
    if (id < 0 || id >= vocabSize) {
      outOfRange += 1;
      continue;
    }
    if (id < minId) minId = id;
    if (id > maxId) maxId = id;
    freqs[id] += 1;
  }
  const safeMin = minId === Infinity ? NaN : minId;
  const safeMax = maxId === -Infinity ? NaN : maxId;
  const totalValid = freqs.reduce((acc, cur) => acc + cur, 0);
  const top = freqs
    .map((count, id) => ({ id, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 5);
  console.log(
    `[tokenizer] id_range=${safeMin}-${safeMax} out_of_range=${outOfRange} non_int=${nonInt}`
  );
  if (totalValid > 0) {
    const topStr = top
      .map((entry) => `${entry.id}:${((entry.count / totalValid) * 100).toFixed(2)}%`)
      .join(" ");
    console.log(`[tokenizer] top_ids=${topStr}`);
  }

  const sampleOffsets = [
    0,
    Math.max(0, Math.floor(rawText.length / 2) - 100),
    Math.max(0, rawText.length - 200)
  ];
  sampleOffsets.forEach((start, idx) => {
    const slice = rawText.slice(start, start + 200);
    const enc = tokenizer.encode(slice);
    const dec = tokenizer.decode(enc);
    const exact = dec === slice;
    let mismatch = -1;
    if (!exact) {
      const len = Math.min(slice.length, dec.length);
      for (let i = 0; i < len; i++) {
        if (slice[i] !== dec[i]) {
          mismatch = i;
          break;
        }
      }
      if (mismatch === -1 && slice.length !== dec.length) {
        mismatch = len;
      }
    }
    const label = exact ? "exact" : `mismatch_at=${mismatch}`;
    console.log(
      `[tokenizer] sample${idx + 1} chars=${slice.length} tokens=${enc.length} ${label}`
    );
  });
}

const tape = new Tape();
const modelConfig = {
  vocabSize,
  contextLength,
  embedDim: 128,
  numHeads: 4,
  numLayers: 2,
  dropout: overfit ? 0 : 0.1
};
const checkpointPath = "checkpoints/mattgpt.json";
let model: MattGPT;
if (fs.existsSync(checkpointPath)) {
  try {
    const checkpoint = JSON.parse(fs.readFileSync(checkpointPath, "utf8"));
    const savedConfig = checkpoint?.model?.config;
    const configMatches =
      savedConfig &&
      savedConfig.vocabSize === modelConfig.vocabSize &&
      savedConfig.contextLength === modelConfig.contextLength &&
      savedConfig.embedDim === modelConfig.embedDim &&
      savedConfig.numHeads === modelConfig.numHeads &&
      savedConfig.numLayers === modelConfig.numLayers &&
      savedConfig.dropout === modelConfig.dropout;
    if (configMatches) {
      model = MattGPT.fromState(checkpoint.model, tape);
      console.log("[setup] loaded model checkpoint");
    } else {
      console.log("[setup] checkpoint config mismatch, starting fresh");
      model = new MattGPT({ ...modelConfig, tape });
    }
  } catch (err) {
    console.warn("[setup] failed to load checkpoint, starting fresh");
    model = new MattGPT({ ...modelConfig, tape });
  }
} else {
  model = new MattGPT({ ...modelConfig, tape });
}
model.train();
console.log("[setup] model initialized");
if (overfit) {
  console.log("[setup] overfit mode enabled");
}

const makeBatch = (fixedStart?: number): { inputs: Tensor; targets: Tensor } => {
  const inputs = new Array(batchSize * contextLength);
  const targets = new Array(batchSize * contextLength);
  const maxStart = encoded.length - contextLength - 1;
  if (maxStart <= 0) {
    throw new Error("Dataset is too small for the chosen context length.");
  }
  for (let b = 0; b < batchSize; b++) {
    const start =
      fixedStart === undefined
        ? Math.floor(Math.random() * maxStart)
        : Math.min(fixedStart, maxStart);
    for (let t = 0; t < contextLength; t++) {
      const idx = b * contextLength + t;
      inputs[idx] = encoded[start + t];
      targets[idx] = encoded[start + t + 1];
    }
  }
  return {
    inputs: Tensor.fromArray([batchSize, contextLength], inputs),
    targets: Tensor.fromArray([batchSize, contextLength], targets)
  };
};

const stats = (values: Float32Array | number[]) => {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumAbs = 0;
  for (let i = 0; i < values.length; i++) {
    const val = values[i];
    if (val < min) min = val;
    if (val > max) max = val;
    sum += val;
    sumAbs += Math.abs(val);
  }
  const mean = values.length ? sum / values.length : 0;
  const meanAbs = values.length ? sumAbs / values.length : 0;
  return { min, max, mean, meanAbs };
};

const nowMs = (): number => Number(process.hrtime.bigint()) / 1e6;
const prompt = "To be, or not to be";
const maxNewTokens = 50;

const greedySample = (
  promptText: string,
  tokensToGenerate: number,
  restoreTrain: boolean
): string => {
  model.eval();
  let promptIds = tokenizer.encode(promptText);
  for (let step = 0; step < tokensToGenerate; step++) {
    const context = promptIds.slice(-contextLength);
    const input = Tensor.fromArray([1, context.length], context);
    const logits = model.forward(input);
    const lastIndex = context.length - 1;
    const lastLogits = logits.view([context.length, vocabSize]).data;
    const offset = lastIndex * vocabSize;
    let bestId = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      const val = lastLogits[offset + i];
      if (val > bestVal) {
        bestVal = val;
        bestId = i;
      }
    }
    promptIds.push(bestId);
  }
  const generated = tokenizer.decode(promptIds);
  if (restoreTrain) {
    model.train();
  }
  return generated;
};

console.log("[train] starting");
const params = model.parameters();
const numParams = params.reduce((acc, cur) => acc + cur.size, 0);
console.log(`[train] parameters=${numParams}`);
const startTime = nowMs();
let stepTimeMs = 0;
for (let step = 0; step < maxSteps; step++) {
  const stepStart = nowMs();
  const { inputs, targets } = makeBatch(overfit ? 0 : undefined);
  tape.clear();
  const logits = model.forward(inputs);
  const flatLogits = logits.view([batchSize * contextLength, vocabSize]);
  const flatTargets = targets.view([batchSize * contextLength]);
  const loss = crossEntropy(flatLogits, flatTargets);

  model.parameters().forEach((p) => p.zeroGrad());
  loss.backward();

  for (const p of model.parameters()) {
    for (let i = 0; i < p.size; i++) {
      p.data[i] -= lr * p.grad[i];
    }
  }

  stepTimeMs += nowMs() - stepStart;
  const logEvery = 100;
  if (step % logEvery === 0) {
    const avgStepMs = stepTimeMs / (step + 1);
    const elapsedMs = nowMs() - startTime;
    console.log(
      `[train] step=${step} loss=${loss.data[0].toFixed(4)} avg_step_ms=${avgStepMs.toFixed(2)} elapsed_s=${(elapsedMs / 1000).toFixed(1)}`
    );
    if (debugStats) {
      const logitStats = stats(flatLogits.data);
      const gradStats = stats(params[0].grad);
      console.log(
        `[stats] logits min=${logitStats.min.toFixed(4)} max=${logitStats.max.toFixed(4)} mean=${logitStats.mean.toFixed(4)} meanAbs=${logitStats.meanAbs.toFixed(4)} grad0_meanAbs=${gradStats.meanAbs.toFixed(6)}`
      );
      if (step === 0) {
        const targetStats = stats(flatTargets.data);
        console.log(
          `[stats] targets min=${targetStats.min.toFixed(0)} max=${targetStats.max.toFixed(0)}`
        );
      }
    }
    if (!overfit) {
      const sample = greedySample(prompt, maxNewTokens, true);
      console.log("[infer] sample:");
      console.log(sample);
    }
  }
}

console.log("[train] done, saving checkpoint");
const checkpoint = {
  model: model.getState(),
  tokenizer: tokenizer.getState()
};
fs.mkdirSync("checkpoints", { recursive: true });
fs.writeFileSync("checkpoints/mattgpt.json", JSON.stringify(checkpoint));

console.log("[infer] running greedy sampling");
const generated = greedySample(prompt, maxNewTokens, false);
console.log("[infer] output:");
console.log(generated);
