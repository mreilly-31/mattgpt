import { Value } from "./structures";

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

export const zip = (a: any[], b: any[]): any[][] => {
  let result = [];
  const maxCompatLength = Math.min(a.length, b.length);
  for (let i = 0; i < maxCompatLength; i++) {
    result.push([a[i], b[i]]);
  }

  return result;
};

export const sum = (arr: number[], start?: number): number => arr.reduce((acc, cur) => acc += cur, start ?? 0);

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
