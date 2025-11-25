import { sum } from "../matt-torch";

type Tensor<Dims extends readonly number[], T> =
  Dims extends readonly [number, ...infer Rest extends number[]]
    ? Tensor<Rest, T>[]
    : T;

type FillFunc<T> = () => T;

type TensorData<Dims extends readonly number[]> = Tensor<Dims, number>;


export class NDimTensor<Dims extends readonly number[]> {
  constructor(
    public readonly dims: Dims,
    public readonly data: TensorData<Dims>
  ) {}
}

export function tensor<Dims extends readonly number[]>(
  dimensions: Dims,
  fillFunc: FillFunc<number> | number
): Tensor<Dims, number> {
  const recursivelyGenerateArray = (
    sizeTuple: readonly number[]
  ): any => {
    if (sizeTuple.length === 0) {
      return typeof fillFunc === "function"
        ? (fillFunc as FillFunc<number>)()
        : fillFunc;
    }

    const [currentDimension, ...remainingDimensions] = sizeTuple;
    const result: any[] = [];

    for (let i = 0; i < currentDimension; i++) {
      result.push(recursivelyGenerateArray(remainingDimensions));
    }

    return result;
  };

  return recursivelyGenerateArray(dimensions);
}

export function normalize<Dims extends readonly number[]>(
  tensor: Tensor<Dims, number>
): Tensor<Dims, number> {
  const recursivelyNormalizeDimensions = (dim: any): any => {
    // dim will be either a base dimension of the tensor, or we will recurse
    if (!Array.isArray(dim)) {
      // empty dimensions, return
      return dim;
    }

    if (dim.length > 0 && !Array.isArray(dim[0])) {
      const row = dim as number[];
      const rowSum = sum(row);
      return row.map(p => p / rowSum);      
    }

    return dim.map(recursivelyNormalizeDimensions);
  }

  return recursivelyNormalizeDimensions(tensor) as Tensor<Dims, number>;
}
