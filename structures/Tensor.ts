import { off } from "node:process";
import { Value } from "./Value";

type IndexTuple = readonly number[];
type ValueInitializer =
  | number
  | Value
  | ((coords: number[]) => number | Value);

const isValue = (candidate: unknown): candidate is Value =>
  candidate instanceof Value;
export class Tensor {
  readonly dims: IndexTuple;
  private readonly strides: number[];
  private readonly data: Value[];
  readonly size: number

  constructor(dims: IndexTuple, fill: ValueInitializer = 0) {
    if (dims.length === 0) {
      throw new Error("A tensor must have at least one dimension");
    }

    this.dims = [...dims];
    this.size = this.dims.reduce((acc, cur) => acc *= cur, 1);
    this.strides = Tensor.buildStrides(dims);
    const elementCount = this.count();
    this.data = new Array(elementCount);

    for (let offset = 0; offset < elementCount; offset++) {
      const coords = this.unflatten(offset);
      this.data[offset] = this.resolveInitializer(fill, coords);
    }
  }

  static fromNestedArray(dims: IndexTuple, values: number[][] | number[]): Tensor {
    const tensor = new Tensor(dims);
    const fillRecursive = (coords: number[], depth: number, node: any) => {
      if (depth === dims.length) {
        tensor.set(coords, node as number);
        return;
      }

      node.forEach((child: any, idx: number) => {
        fillRecursive([...coords, idx], depth + 1, child);
      });
    };

    fillRecursive([], 0, values);
    return tensor;
  }

  /**
   * Returns the total count of elements in the tensor.
   * 
   * @returns number
   */
  count(): number {
    return this.dims.reduce((acc, cur) => acc * cur, 1);
  }

  /**
   * Returns an array containing the dimensions of the tensor.
   * 
   * @returns the dimensions of the tensor
   */
  shape(): number[] {
    return this.dims as number[];
  }

  /**
   * Returns the element in the tensor at the specified location.
   * 
   * @param indices - the index tuple of the tensor location being accessed
   * @returns the value at indices
   */
  at(indices: number[]): Value {
    const flatIndex = this.flatten(indices);
    return this.data[flatIndex];
  }

  /**
   * Return a whole row at the given location.
   * 
   * @param indices 
   * @returns the row in the tensor at the given indices
   */
  row(indices: number[]): Value[] {
    if (indices.length !== this.dims.length - 1) {
      throw new Error("Not a last axis row, can't get");
    }
    const flatIndex = this.flatten([...indices, 0]);
    return this.data.slice(flatIndex, flatIndex + this.dims[this.dims.length - 1]);
  }

  /**
   * Sets the Value at the provided indices.
   * 
   * @param indices - the index tuple of the tensor location being set
   * @param value  = the value to be set
   */
  set(indices: number[], value: number | Value) {
    const flatIndex = this.flatten(indices);
    this.data[flatIndex] = isValue(value) ? value : new Value(value);
  }

  /**
   * Iterate through a tensor and perform a transformation on each value.
   * 
   * @param transform - the transformation function to perform on the tensor value
   * @returns a newly transformed tensor
   */
  map(transform: (value: Value, coords: number[]) => Value | number): Tensor {
    const next = new Tensor(this.dims);
    this.data.map((val, index) => {
      // get my current position indices
      const coords = this.unflatten(index);
      // perform the transformation given
      const mapped = transform(val, coords);
      // assign the new value in the new tensor
      next.data[index] = isValue(mapped) ? mapped : new Value(mapped);
    });
    return next;
  }

  /**
   * Given a tensor, normalize each value within its dimension to a probability
   * 
   * @returns a new normalized tensor
   */
  normalize(): Tensor {
    if (this.dims.length === 0) {
      throw new Error(`This is a scalar tensor, cannot normalize`);
    }

    const prefixSums = new Map<string, number>();
    // get map key given arr of incices
    // if only 1 dim, return empty
    // else get concat of all indices except the last dim
    const getKeyForLocation = (loc: number[]) => loc.length > 1 ? loc.slice(0, -1).join(',') : '';

    // compute n-1 axis sums
    for (let i = 0; i < this.size; i++) {
      const loc = this.unflatten(i);
      const key = getKeyForLocation(loc);
      const previousSum = prefixSums.get(key) ?? 0;
      // add the current value to the already summed values along this last axis dimension
      prefixSums.set(key, previousSum + this.data[i].data);
    }

    // perform normalization
    return this.map((val, index) => {
      const key = getKeyForLocation(index);
      const rowSum = prefixSums.get(key) ?? 0;
      return val.data / rowSum;
    });
  }

  // better implementation of fillFunc in old tensor
  // fill having coords makes it much more useful
  // the generator was sweet but not effective for n dims
  private resolveInitializer(
    fill: ValueInitializer,
    coords: number[]
  ): Value {
    if (typeof fill === "number") {
      return new Value(fill);
    }

    if (isValue(fill)) {
      return new Value(fill.data);
    }

    if (typeof fill === "function") {
      const result = fill(coords);
      return isValue(result) ? result : new Value(result);
    }

    throw new Error("Unsupported initializer");
  }

  // given an arr of n dim indices, tell me where in the scalar tensor i am
  private flatten(indices: number[]): number {
    if (indices.length !== this.dims.length) {
      throw new Error(
        `Expected ${this.dims.length} indices, received ${indices.length}`
      );
    }

    return indices.reduce((acc, cur, idx) => {
      // what level are we on?
      const dim = this.dims[idx];
      if (cur < 0 || cur >= dim) {
        throw new Error(
          `Index ${cur} is out of bounds for dimension size ${dim}`
        );
      }
      // acc = how far into the 1D array we have gotten so far
      // cur = the index we're looking for on the current level
      // strides[idx] = how far we have to jump on this level
      // so we add the current offset to the index we seek * stride
      // then do for the next levels until we arrive at a single scalar index
      return acc + cur * this.strides[idx];
    }, 0);
  }

  // this is the flatten logic in reverse
  // thats why it's called unflatten i guess
  // given a 1d index, tell me where in the tensor I am
  private unflatten(offset: number): number[] {
    let remainder = offset;
    const coords = new Array(this.dims.length);

    for (let axis = 0; axis < this.dims.length; axis++) {
      const stride = this.strides[axis];
      coords[axis] = Math.floor(remainder / stride);
      remainder %= stride;
    }

    return coords;
  }

  // a stride is the jump necessary to go from one element to the next in the given dimension
  // https://docs.pytorch.org/docs/stable/generated/torch.Tensor.stride.html
  private static buildStrides(dims: IndexTuple): number[] {
    const strides = new Array(dims.length);
    let accumulator = 1;

    for (let axis = dims.length - 1; axis >= 0; axis--) {
      strides[axis] = accumulator;
      accumulator *= dims[axis];
    }

    return strides;
  }
}
