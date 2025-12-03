import { Value } from "./Value";

type IndexTuple = readonly number[];
type ValueInitializer = number | Value | ((coords: number[]) => number | Value);

const isValue = (candidate: unknown): candidate is Value =>
  candidate instanceof Value;

type Slice = {
  start?: number;
  end?: number;
  step?: number;
};


/**
 * My tensor implementation, following some pytorch naming etc but mostly just winging it
 * uses the Value class as the base data type
 * Will keep going with this as long as I can until I'm spending more time reimplementing pytorch than I am learning ai
 * But maybe that _is_ learning ai? we'll see
 */
export class Tensor {
  readonly dims: IndexTuple;
  private readonly strides: number[];
  private data: Value[];
  readonly size: number;

  constructor(dims: IndexTuple, fill: ValueInitializer = 0) {
    if (dims.length === 0) {
      throw new Error("A tensor must have at least one dimension");
    }

    this.dims = [...dims];
    this.size = this.dims.reduce((acc, cur) => (acc *= cur), 1);
    this.strides = Tensor.buildStrides(dims);
    const elementCount = this.count();
    this.data = new Array(elementCount);

    for (let offset = 0; offset < elementCount; offset++) {
      const coords = this.unflatten(offset);
      this.data[offset] = this.resolveInitializer(fill, coords);
    }
  }

  static fromNestedArray(
    dims: IndexTuple,
    values: number[][] | number[]
  ): Tensor {
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

  show(use_grad?: boolean): void {
    const is_grad = use_grad ?? false;
    const chunks: string[] = [];
    const visit = (prefix: number[], depth: number) => {
      if (depth === this.dims.length - 1) {
        const rowValues = this.vrow(prefix)
          .map((value) => is_grad ? value.grad : value.data)
          .join(", ");
        chunks.push(
          prefix.length ? " ".repeat(prefix.length) : "",
          `[${rowValues}]`
        );
        return;
      }
      chunks.push("[");

      for (let i = 0; i < this.dims[depth]; i++) {
        visit([...prefix, i], depth + 1);
      }

      chunks.push("]");
    };

    visit([], 0);
    console.log(`Tensor of shape ${this.shape()} (${is_grad ? 'Grad' : 'Data'}):\n${chunks.join("\n")}`);
  }

  show_grad(): void {
    this.show(true);
  }

  /**
   * Return the sum of all values in the tensor
   * 
   * @param start optional start value for the summation
   * @returns the sum of all values in the tensor.
   */
  sum(start?: number): Value {
    const sumStart = start ?? 0;
    return this.data.reduce((acc, cur) => acc.add(cur), new Value(sumStart));
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
   * Perform a backward pass on the tensor
   */
  backward() {
    this.data.forEach(item => item.backward());
  }

  /**
   * Return the row values as raw Value instances.
   * Deprecated: prefer row() which returns a Tensor view.
   */
  vrow(indices: number[]): Value[] {
    if (indices.length !== this.dims.length - 1) {
      throw new Error("Not a last axis row, can't get");
    }
    const flatIndex = this.flatten([...indices, 0]);
    return this.data.slice(
      flatIndex,
      flatIndex + this.dims[this.dims.length - 1]
    );
  }

  /**
   * Return a whole row at the given location as a Tensor.
   */
  row(indices: number[]): Tensor {
    const values = this.vrow(indices);
    const result = new Tensor([values.length], 0);
    values.forEach((value, idx) => {
      result.set([idx], value);
    });
    return result;
  }

  /**
   * Zero out the grads of each value in the tensor.
   * 
   * Mutates the tensor in-place.
   */
  zero_grad(): void {
    this.data.forEach((item) => {
      item.grad = 0.0;
    });
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
   * Slices the tensor along a single dimension, returning a 1-D tensor view.
   * Supports a Slice object for start/end/step semantics similar to Python.
   * pick which dimension you’re slicing, then describe the range along that dimension.
   * AI helped with this one.
   * 
   * EXAMPLE USAGE:
   * 
   * Extract a subset of rows from a batch: const middleRows = tensor.slice(0, new Slice({ start: 10, end: 20 })); keeps only rows 10–19 (axis 0) while preserving all columns. Handy when you want to inspect or process a mini-batch without copying the entire tensor manually.
   * 
   * Grab every other feature in a vector: const evenFeatures = tensor.slice(1, new Slice({ start: 0, step: 2 })); on a [batch, features] tensor returns a view with half the columns, useful for downsampling or debugging a particular feature subset.
   * 
   * Slice backward from the end: const lastThree = tensor.slice(0, new Slice({ start: -3 })); works like Python’s [-3:], so you can pull the trailing rows of a sequence (e.g., the final three time steps of each sample).
   * 
   * Strided sampling for visualization: const coarse = imageTensor.slice(1, new Slice({ start: 0, end: imageHeight, step: 4 })); lets you downsample an image-like tensor rapidly by walking every fourth row/column without writing custom loops.
   * 
   * @param dim - which dimension in the tensor to slice
   * @param slice.start - where to start the slice within axis
   * @param slice.end - where to end the slice within axis
   * @param slice.step - positive means go from start to end, negative goes from end to start (allow for inversion)
   */
  slice(dim: number, slice: Slice): Tensor {
    if (dim < 0 || dim >= this.dims.length) {
      throw new Error(`Axis ${dim} is out of bounds for tensor with ${this.dims.length} dims`);
    }

    const length = this.dims[dim];
    const start = slice.start ?? 0;
    const end = slice.end ?? length;
    const step = slice.step ?? 1;

    if (step === 0) {
      throw new Error("Slice step cannot be zero");
    }

    // normalize positive / negative values 
    // clamp to axis length to avoid any out of bounds
    const normalizedStart = start < 0 ? Math.max(length + start, 0) : Math.min(start, length);
    const normalizedEnd = end < 0 ? Math.max(length + end, 0) : Math.min(end, length);

    // build list of indices needed to access, in order of return desire (given step passed)
    const indices: number[] = [];
    if (step > 0) {
      for (let i = normalizedStart; i < normalizedEnd; i += step) {
        indices.push(i);
      }
    } else {
      for (let i = normalizedStart; i > normalizedEnd; i += step) {
        indices.push(i);
      }
    }

    const newDims = [...this.dims];
    newDims[dim] = indices.length;

    const result = new Tensor(newDims, 0);
    const srcCoords = new Array(this.dims.length).fill(0);
    const dstCoords = new Array(this.dims.length).fill(0);

    // recursively build the output
    const fillSlice = (depth: number) => {
      // leaf, fill tensor
      if (depth === this.dims.length) {
        result.set(dstCoords.slice(), this.at(srcCoords));
        return;
      }

      if (depth === dim) {
        for (let i = 0; i < indices.length; i++) {
          srcCoords[depth] = indices[i];
          dstCoords[depth] = i;
          fillSlice(depth + 1);
        }
      } else {
        for (let i = 0; i < this.dims[depth]; i++) {
          srcCoords[depth] = i;
          dstCoords[depth] = i;
          fillSlice(depth + 1);
        }
      }
    };

    fillSlice(0);
    return result;
  }

  /**
   * Iterate through a tensor and perform a transormation in place.
   * Modifies the existing data.
   * 
   * @param transform - the transformation function to perform on the tensor value
   * @returns void
   */
  forEach(transform: (value: Value, coords: number[]) => Value | number): void {
    this.data.forEach((val, index) => {
      const coords = this.unflatten(index);
      const result = transform(val, coords);
      if (isValue(result)) {
        val.data = result.data;
        val.grad = result.grad;
      } else {
        val.data = result;
      }
    });
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

    const prefixSums = new Map<string, Value>();
    // get map key given arr of incices
    // if only 1 dim, return empty
    // else get concat of all indices except the last dim
    const getKeyForLocation = (loc: number[]) =>
      loc.length > 1 ? loc.slice(0, -1).join(",") : "";

    // compute n-1 axis sums
    for (let i = 0; i < this.size; i++) {
      const loc = this.unflatten(i);
      const key = getKeyForLocation(loc);
      const previousSum = prefixSums.get(key) ?? new Value(0);
      // add the current value to the already summed values along this last axis dimension
      prefixSums.set(key, previousSum.add(this.data[i]));
    }

    // perform normalization
    return this.map((val, index) => {
      const key = getKeyForLocation(index);
      const rowSum = prefixSums.get(key);
      if (!rowSum) {
        throw new Error("missing a row sum for normalization");
      }
      return val.divide(rowSum);
    });
  }

  // better implementation of fillFunc in old tensor
  // fill having coords makes it much more useful
  // the generator was sweet but not effective for n dims
  private resolveInitializer(fill: ValueInitializer, coords: number[]): Value {
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
