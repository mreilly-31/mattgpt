import { Tensor } from "./structures/Tensor";
import { Value } from "./structures/Value";

// super basic random sample algorithm, I'm sure its good enough
export function weightedRandomSample(
  probabilities: number[],
  numSamples: number
): number[] {
  const allowedResultValues = Array.from(
    { length: probabilities.length },
    (_, index) => index
  );

  let result: number[] = [];
  for (let step = 0; step < numSamples; step++) {
    const randomNumber = randFloat(0, 1);
    let cursor = 0;
    for (let i = 0; i < probabilities.length; i++) {
      cursor += probabilities[i];
      if (cursor >= randomNumber) {
        result.push(allowedResultValues[i]);
        break;
      }
    }
  }
  return result;
}

export function shuffle(array: any[]): any[] {
  let result = array;
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

export const ninf = -1e9;

/**
 * Samples indices from a categorical distribution defined by `values`.
 * - `values` must be a 1-D tensor containing non-negative weights (not necessarily normalized).
 * - Returns `numSamples` draws with replacement.
 */
export function multinomial(values: Tensor, numSamples: number): number[] {
  if (values.shape.length !== 1) {
    throw new Error("multinomial currently only supports 1-D tensors");
  }
  if (numSamples <= 0) {
    return [];
  }

  const weights = values.data;
  let totalWeight = 0;
  for (let i = 0; i < weights.length; i++) {
    totalWeight += weights[i];
  }
  if (totalWeight <= 0) {
    throw new Error("Sum of weights must be positive to sample");
  }

  const probabilities = new Array(weights.length);
  for (let i = 0; i < weights.length; i++) {
    probabilities[i] = weights[i] / totalWeight;
  }
  const draws: number[] = [];

  for (let sample = 0; sample < numSamples; sample++) {
    const r = randFloat(0, 1);
    let cursor = 0;
    for (let i = 0; i < probabilities.length; i++) {
      cursor += probabilities[i];
      if (r <= cursor) {
        draws.push(i);
        break;
      }
    }
  }

  return draws;
}

export const randInt = (low: number, high: number): number =>
  Math.floor(
    Math.random() * (Math.floor(high) - Math.ceil(low) + 1) + Math.ceil(low)
  );
export const randFloat = (low: number, high: number): number =>
  Math.random() * (high - low) + low;
export const sum = (arr: number[], start?: number): number =>
  arr.reduce((acc, cur) => (acc += cur), start ?? 0);
export const zip = (a: any[], b: any[]): any[][] => {
  let result = [];
  const maxCompatLength = Math.min(a.length, b.length);
  for (let i = 0; i < maxCompatLength; i++) {
    result.push([a[i], b[i]]);
  }

  return result;
};

type DeepCastToValue<T> = T extends number
  ? Value
  : T extends Array<infer U>
  ? DeepCastToValue<U>[]
  : never;

export function valuize<T>(data: T): DeepCastToValue<T> {
  if (Array.isArray(data)) {
    return data.map((item) => valuize(item)) as DeepCastToValue<T>;
  }

  return new Value(data as number) as DeepCastToValue<T>;
}

export const loss = (predictions: Value[], targets: Value[]) => {
  const combined = zip(targets, predictions);
  let result: Value[] = [];
  for (const combo of combined) {
    // calculate distance
    result.push(combo[0].subtract(combo[1]).pow(2));
  }
  return result.reduce((acc: Value, cur: Value) => {
    return acc.add(cur);
  }, new Value(0));
};

export const oneHot = (vals: number[], num_classes: number): number[][] => {
  let result: number[][] = [];
  for (let i = 0; i < vals.length; i++) {
    const zerosRow: number[] = new Array(num_classes).fill(0.0);
    zerosRow[vals[i]] = 1.0;
    result.push(zerosRow);
  }

  return result;
};

// box muller transform to get random numbers over a normal gaussian distribution
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
export const randomNormal = (): number => {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

export const multiply = (a: Tensor, b: Tensor): Tensor => a.matmul(b);

export const softmax = (t: Tensor, dim?: number): Tensor => t.softmax(dim);

export const arrange = (count: number): Tensor => {
  const filler = new Array(count);
  for (let i = 0; i < count; i++) {
    filler[i] = i;
  }
  return Tensor.fromArray([count], filler);
};

export const oneHotTensor = (
  indices: number[] | Tensor,
  numClasses: number
): Tensor => {
  const values = Array.isArray(indices) ? indices : Array.from(indices.data);
  const batch = values.length;
  const data = new Array(batch * numClasses).fill(0);
  for (let i = 0; i < batch; i++) {
    const idx = values[i];
    if (!Number.isInteger(idx)) {
      throw new Error(`Target index ${idx} is not an integer`);
    }
    if (idx < 0 || idx >= numClasses) {
      throw new Error(`Target index ${idx} is out of bounds`);
    }
    data[i * numClasses + idx] = 1;
  }
  return Tensor.fromArray([batch, numClasses], data);
};

export const crossEntropy = (logits: Tensor, target: Tensor): Tensor => {
  const logProbs = logits.logsoftmax(1);
  const targetOneHot = oneHotTensor(target, logits.shape[1]);
  const picked = logProbs.mul(targetOneHot).sum(1);
  return picked.sum().mulScalar(-1 / logits.shape[0]);
};

export const triu = (input: Tensor, diagonal = 0, grad: boolean = true): Tensor => {
  if (input.shape.length !== 2) {
    throw new Error("triu currently supports 2D tensors only");
  }
  const [rows, cols] = input.shape;
  const data = new Array(input.size).fill(0);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      if (j - i >= diagonal) {
        const index = i * input.strides[0] + j * input.strides[1];
        data[index] = 1;
      }
    }
  }
  const mask = Tensor.fromArray([rows, cols], data, grad);
  return input.mul(mask);
};

export const ones = (shape: number[], grad: boolean = true): Tensor => {
  if (shape.length === 0) {
    return Tensor.fromArray([], [1]);
  }
  const size = shape.reduce((acc, cur) => acc * cur, 1);
  const data = new Array(size).fill(1);
  return Tensor.fromArray(shape, data, grad);
};

/**
 * Kaiming uniform-ish init (common default in deep learning).
 * For demo: uniform in [-bound, bound], with bound ~ sqrt(1 / inFeatures)
 * https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/
 * https://arxiv.org/pdf/1502.01852
 *
 * (PyTorch uses kaiming_uniform_ with a=sqrt(5); this is a close, simple version.)
 */
export const initWeights = (
  in_features: number,
  out_features: number
): Tensor => {
  const bound = Math.sqrt(1 / in_features);
  const size = in_features * out_features;
  const data = new Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = randFloat(-bound, bound);
  }
  return Tensor.fromArray([in_features, out_features], data);
};

/**
 * Bias uniform in [-bound, bound] where bound = 1/sqrt(fan_in),
 * matching the typical PyTorch linear bias initialization idea.
 */
export const initBias = (in_features: number, out_features: number): Tensor => {
  const bound = 1 / Math.sqrt(in_features);
  const data = new Array(out_features);
  for (let i = 0; i < out_features; i++) {
    data[i] = randFloat(-bound, bound);
  }
  return Tensor.fromArray([out_features], data);
}
