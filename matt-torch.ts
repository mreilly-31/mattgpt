import { Tensor, Value } from "./structures";

// super basic random sample algorithm, I'm sure its good enough
export function weightedRandomSample(
  probabilities: number[],
  numSamples: number
): number[] {
  const allowedResultValues = Array.from({ length: probabilities.length }, (_, index) => index);

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

export const randInt = (low: number, high: number): number => Math.floor(Math.random() * (Math.floor(high) - Math.ceil(low) + 1) + Math.ceil(low));
export const randFloat = (low: number, high: number) : number => Math.random() * (high - low) + low;
export const sum = (arr: number[], start?: number): number => arr.reduce((acc, cur) => acc += cur, start ?? 0);
export const zip = (a: any[], b: any[]): any[][] => {
  let result = [];
  const maxCompatLength = Math.min(a.length, b.length);
  for (let i = 0; i < maxCompatLength; i++) {
    result.push([a[i], b[i]]);
  }

  return result;
};


type DeepCastToValue<T> = 
  T extends number
    ? Value
    : T extends Array<infer U>
      ? DeepCastToValue<U>[]
      : never;

export function valuize<T>(data: T): DeepCastToValue<T> {
  if (Array.isArray(data)) {
    return data.map(item => valuize(item)) as DeepCastToValue<T>;
  }
  
  return new Value(data as number) as DeepCastToValue<T>;
};

export const loss = (predictions: Value[], targets: Value[]) => {
  const combined = zip(targets, predictions);
  let result: Value[] = [];
  for (const combo of combined) {
    // calculate distance
    result.push((combo[0].subtract(combo[1]).pow(2)))
  }
  return result.reduce((acc: Value, cur: Value) => {
    return acc.add(cur)
  }, new Value(0));
}

export const oneHot = (vals: number[], num_classes: number): number[][] => {
  let result: number[][] = [];
  for (let i = 0; i < vals.length; i++) {
    const zerosRow: number[] = new Array(num_classes).fill(0.0);
    zerosRow[vals[i]] = 1.0;
    result.push(zerosRow);
  }

  return result;
}

// box muller transform to get random numbers over a normal gaussian distribution
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
export const randomNormal = (): number => {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// naive 2 dim O(n^3) is all i have in me right now
export const multiply = (a: Tensor, b: Tensor): Tensor => {
  if (a.dims.length !== 2 || b.dims.length !== 2) {
    throw new Error('I can only do 2d matrices right now');
  }

  if (a.dims[1] !== b.dims[0]) {
    throw new Error('Cant multiply these');
  }

  const result: Tensor = new Tensor([a.dims[0], b.dims[1]], 0);
  for (let i = 0; i < a.dims[0]; i++) {
    for (let j = 0; j < b.dims[1]; j++) {
      for (let k = 0; k < a.dims[1]; k++) {
        // TODO: might need to use the value multiply function
        // figure that out
        const currentProduct = result.at([i, j]).data;
        const additionalProduct = a.at([i, k]).data * b.at([k, j]).data;
        result.set([i, j], currentProduct + additionalProduct);
      }
    }
  }

  return result;
}

export const softmax = (t: Tensor): Tensor => {
  const exponentiated = t.map(item => item.exp());
  return exponentiated.normalize();
}