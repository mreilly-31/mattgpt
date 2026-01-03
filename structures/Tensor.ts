import {
  getMatmulBackend,
  layerNormWasm,
  layerNormBackwardWasm,
  logsoftmaxWasm,
  logsoftmaxBackwardWasm,
  geluWasm,
  mlpFusedWasm,
  matmulBackwardWasm,
  matmulWasm,
  reduceMaxWasm,
  reduceMeanWasm,
  reduceSumWasm,
  softmaxWasm,
  softmaxBackwardWasm
} from "./wasm/wasm-math";

type Shape = number[];

type TapeNode = {
  backward: () => void;
};

export class Tape {
  private nodes: TapeNode[] = [];

  add(node: TapeNode): void {
    this.nodes.push(node);
  }

  backward(): void {
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      this.nodes[i].backward();
    }
  }

  clear(): void {
    this.nodes = [];
  }
}

const computeSize = (shape: Shape): number => {
  if (shape.length === 0) return 1;
  return shape.reduce((acc, cur) => acc * cur, 1);
};

const buildStrides = (shape: Shape): number[] => {
  const strides = new Array(shape.length);
  let accumulator = 1;
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    strides[axis] = accumulator;
    accumulator *= shape[axis];
  }
  return strides;
};

const flattenIndex = (coords: number[], strides: number[]): number => {
  let index = 0;
  for (let i = 0; i < coords.length; i++) {
    index += coords[i] * strides[i];
  }
  return index;
};

const broadcastShape = (a: Shape, b: Shape, op: string): Shape => {
  const rank = Math.max(a.length, b.length);
  const shape = new Array(rank);

  for (let i = 0; i < rank; i++) {
    const aDim = a[a.length - 1 - i] ?? 1;
    const bDim = b[b.length - 1 - i] ?? 1;
    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error(`${op} cannot broadcast shapes ${a} and ${b}`);
    }
    shape[rank - 1 - i] = Math.max(aDim, bDim);
  }

  return shape;
};

const unflattenIndex = (
  offset: number,
  shape: Shape,
  strides: number[]
): number[] => {
  let remainder = offset;
  const coords = new Array(shape.length);
  for (let axis = 0; axis < shape.length; axis++) {
    const stride = strides[axis];
    coords[axis] = Math.floor(remainder / stride);
    remainder %= stride;
  }
  return coords;
};

const broadcastIndex = (
  coords: number[],
  shape: Shape,
  strides: number[]
): number => {
  const offset = coords.length - shape.length;
  let index = 0;
  for (let axis = 0; axis < shape.length; axis++) {
    const dim = shape[axis];
    const coord = dim === 1 ? 0 : coords[axis + offset];
    index += coord * strides[axis];
  }
  return index;
};

const resolveTape = (
  a: Tensor,
  b?: Tensor
): Tape | undefined => {
  if (a.tape && b?.tape && a.tape !== b.tape) {
    throw new Error("Mixed tapes are not supported");
  }
  return a.tape ?? b?.tape;
};

const resolveTapeList = (tensors: Tensor[]): Tape | undefined => {
  let tape: Tape | undefined;
  for (const tensor of tensors) {
    if (!tensor.tape) continue;
    if (!tape) {
      tape = tensor.tape;
      continue;
    }
    if (tape !== tensor.tape) {
      throw new Error("Mixed tapes are not supported");
    }
  }
  return tape;
};

const normalizeAxis = (axis: number, rank: number): number => {
  const resolved = axis < 0 ? axis + rank : axis;
  if (resolved < 0 || resolved >= rank) {
    throw new Error(`Axis ${axis} is out of bounds for rank ${rank}`);
  }
  return resolved;
};

const normalizeAxes = (axis: number | number[], rank: number): number[] => {
  const axes = Array.isArray(axis) ? axis.slice() : [axis];
  const normalized = axes.map((ax) => normalizeAxis(ax, rank));
  const unique = Array.from(new Set(normalized));
  unique.sort((a, b) => a - b);
  return unique;
};

export class Tensor {
  public readonly shape: Shape;
  readonly strides: number[];
  readonly size: number;
  readonly data: Float32Array;
  readonly grad: Float32Array;
  readonly requiresGrad: boolean;
  readonly tape?: Tape;

  constructor(
    shape: Shape,
    options?: {
      data?: Float32Array | number[];
      grad?: Float32Array;
      requiresGrad?: boolean;
      tape?: Tape;
      reuseData?: boolean;
      reuseGrad?: boolean;
    }
  ) {
    this.shape = [...shape];
    this.size = computeSize(this.shape);
    this.strides = buildStrides(this.shape);
    this.requiresGrad = options?.requiresGrad ?? false;
    this.tape = options?.tape;

    if (options?.data) {
      const payload = Array.isArray(options.data)
        ? new Float32Array(options.data)
        : options.data;
      if (payload.length !== this.size) {
        throw new Error(
          `Expected data length ${this.size}, received ${payload.length}`
        );
      }
      this.data = options.reuseData ? payload : new Float32Array(payload);
    } else {
      this.data = new Float32Array(this.size);
    }

    if (options?.grad) {
      if (options.grad.length !== this.size) {
        throw new Error(
          `Expected grad length ${this.size}, received ${options.grad.length}`
        );
      }
      this.grad = options.reuseGrad ? options.grad : new Float32Array(options.grad);
    } else {
      this.grad = new Float32Array(this.size);
    }
  }

  static zeros(
    shape: Shape,
    requiresGrad = false,
    tape?: Tape
  ): Tensor {
    return new Tensor(shape, { requiresGrad, tape });
  }

  static fromArray(
    shape: Shape,
    values: number[] | Float32Array,
    requiresGrad = false,
    tape?: Tape
  ): Tensor {
    return new Tensor(shape, {
      data: values,
      requiresGrad,
      tape
    });
  }

  static scalar(value: number, requiresGrad = false, tape?: Tape): Tensor {
    return new Tensor([], {
      data: new Float32Array([value]),
      requiresGrad,
      tape
    });
  }

  zeroGrad(): void {
    this.grad.fill(0);
  }

  backward(): void {
    if (!this.tape) {
      throw new Error("No tape attached to tensor for backprop");
    }
    if (this.size !== 1) {
      throw new Error("backward() requires a scalar tensor");
    }
    this.grad[0] = 1;
    this.tape.backward();
  }

  reshape(shape: Shape): Tensor {
    const nextSize = computeSize(shape);
    if (nextSize !== this.size) {
      throw new Error(
        `Cannot reshape size ${this.size} into shape ${shape}`
      );
    }
    return new Tensor(shape, {
      data: this.data,
      grad: this.grad,
      requiresGrad: this.requiresGrad,
      tape: this.tape,
      reuseData: true,
      reuseGrad: true
    });
  }

  view(shape: Shape): Tensor {
    return this.reshape(shape);
  }

  add(other: Tensor): Tensor {
    const outShape = broadcastShape(this.shape, other.shape, "add");
    const tape = resolveTape(this, other);
    const requiresGrad = this.requiresGrad || other.requiresGrad;
    const out = new Tensor(outShape, { requiresGrad, tape });
    const outStrides = out.strides;
    for (let i = 0; i < out.size; i++) {
      const coords = unflattenIndex(i, outShape, outStrides);
      const aIndex = broadcastIndex(coords, this.shape, this.strides);
      const bIndex = broadcastIndex(coords, other.shape, other.strides);
      out.data[i] = this.data[aIndex] + other.data[bIndex];
    }
    if (requiresGrad && tape) {
      tape.add({
        backward: () => {
          for (let i = 0; i < out.size; i++) {
            const coords = unflattenIndex(i, outShape, outStrides);
            const aIndex = broadcastIndex(coords, this.shape, this.strides);
            const bIndex = broadcastIndex(coords, other.shape, other.strides);
            const grad = out.grad[i];
            if (this.requiresGrad) this.grad[aIndex] += grad;
            if (other.requiresGrad) other.grad[bIndex] += grad;
          }
        }
      });
    }
    return out;
  }

  addScalar(value: number): Tensor {
    const scalar = Tensor.scalar(value, false, this.tape);
    return this.add(scalar);
  }

  mul(other: Tensor): Tensor {
    const outShape = broadcastShape(this.shape, other.shape, "mul");
    const tape = resolveTape(this, other);
    const requiresGrad = this.requiresGrad || other.requiresGrad;
    const out = new Tensor(outShape, { requiresGrad, tape });
    const outStrides = out.strides;
    for (let i = 0; i < out.size; i++) {
      const coords = unflattenIndex(i, outShape, outStrides);
      const aIndex = broadcastIndex(coords, this.shape, this.strides);
      const bIndex = broadcastIndex(coords, other.shape, other.strides);
      out.data[i] = this.data[aIndex] * other.data[bIndex];
    }
    if (requiresGrad && tape) {
      tape.add({
        backward: () => {
          for (let i = 0; i < out.size; i++) {
            const coords = unflattenIndex(i, outShape, outStrides);
            const aIndex = broadcastIndex(coords, this.shape, this.strides);
            const bIndex = broadcastIndex(coords, other.shape, other.strides);
            const grad = out.grad[i];
            if (this.requiresGrad)
              this.grad[aIndex] += other.data[bIndex] * grad;
            if (other.requiresGrad)
              other.grad[bIndex] += this.data[aIndex] * grad;
          }
        }
      });
    }
    return out;
  }

  mulScalar(value: number): Tensor {
    const scalar = Tensor.scalar(value, false, this.tape);
    return this.mul(scalar);
  }

  divScalar(value: number): Tensor {
    const scalar = Tensor.scalar(value, false, this.tape);
    return this.div(scalar);
  }

  neg(): Tensor {
    return this.mulScalar(-1);
  }

  sub(other: Tensor): Tensor {
    return this.add(other.neg());
  }

  div(other: Tensor): Tensor {
    const tape = resolveTape(this, other);
    const requiresGrad = this.requiresGrad || other.requiresGrad;
    const outShape = broadcastShape(this.shape, other.shape, "div");
    const out = new Tensor(outShape, { requiresGrad, tape });
    const outStrides = out.strides;

    for (let i = 0; i < out.size; i++) {
      const coords = unflattenIndex(i, outShape, outStrides);
      const aIndex = broadcastIndex(coords, this.shape, this.strides);
      const bIndex = broadcastIndex(coords, other.shape, other.strides);
      out.data[i] = this.data[aIndex] / other.data[bIndex];
    }

    if (requiresGrad && tape) {
      tape.add({
        backward: () => {
          for (let i = 0; i < out.size; i++) {
            const coords = unflattenIndex(i, outShape, outStrides);
            const aIndex = broadcastIndex(coords, this.shape, this.strides);
            const bIndex = broadcastIndex(coords, other.shape, other.strides);
            const grad = out.grad[i];
            const aVal = this.data[aIndex];
            const bVal = other.data[bIndex];
            if (this.requiresGrad) this.grad[aIndex] += grad / bVal;
            if (other.requiresGrad) other.grad[bIndex] -= (aVal / (bVal * bVal)) * grad;
          }
        }
      });
    }

    return out;
  }

  matmulBiasAct(
    other: Tensor,
    bias: Tensor,
    activation: "none" | "relu" | "tanh" = "none"
  ): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error("matmulBiasAct requires 2D tensors");
    }
    if (bias.shape.length !== 1) {
      throw new Error("matmulBiasAct requires 1D bias tensor");
    }
    const [m, k] = this.shape;
    const [kOther, n] = other.shape;
    if (k !== kOther || bias.shape[0] !== n) {
      throw new Error("matmulBiasAct requires matching dimensions");
    }

    const tape = resolveTape(this, other) ?? resolveTape(this, bias);
    const requiresGrad =
      this.requiresGrad || other.requiresGrad || bias.requiresGrad;

    const wasmEnabled =
      process.env.TENSOR_WASM_DISABLE !== "1" &&
      process.env.TENSOR_WASM_FUSED_MLP !== "0";
    const backend = wasmEnabled ? getMatmulBackend() : null;
    const canWasm =
      backend &&
      backend.exports.mlp_fused &&
      !requiresGrad;

    if (canWasm) {
      const activationCode =
        activation === "relu" ? 1 : activation === "tanh" ? 2 : 0;
      const outData = mlpFusedWasm(
        backend,
        this.data,
        other.data,
        bias.data,
        m,
        k,
        n,
        activationCode
      );
      return new Tensor([m, n], {
        data: outData,
        requiresGrad: false,
        reuseData: true
      });
    }

    let out = this.matmul(other).add(bias);
    if (activation === "relu") {
      out = out.relu();
    } else if (activation === "tanh") {
      out = out.tanh();
    }
    return out;
  }

  relu(): Tensor {
    const tape = resolveTape(this);
    const out = new Tensor(this.shape, {
      requiresGrad: this.requiresGrad,
      tape
    });
    for (let i = 0; i < this.size; i++) {
      const val = this.data[i];
      out.data[i] = val > 0 ? val : 0;
    }
    if (this.requiresGrad && tape) {
      tape.add({
        backward: () => {
          for (let i = 0; i < out.size; i++) {
            this.grad[i] += (this.data[i] > 0 ? 1 : 0) * out.grad[i];
          }
        }
      });
    }
    return out;
  }

  sum(axis?: number | number[]): Tensor {
    const tape = resolveTape(this);
    if (axis === undefined) {
      const out = new Tensor([], {
        requiresGrad: this.requiresGrad,
        tape
      });
      let total = 0;
      for (let i = 0; i < this.size; i++) {
        total += this.data[i];
      }
      out.data[0] = total;
      if (this.requiresGrad && tape) {
        tape.add({
          backward: () => {
            const grad = out.grad[0];
            for (let i = 0; i < this.size; i++) {
              this.grad[i] += grad;
            }
          }
        });
      }
      return out;
    }

    if (this.shape.length === 0) {
      throw new Error("Cannot reduce scalar tensor by axis");
    }

    const axes = normalizeAxes(axis, this.shape.length);
    const axesSet = new Set(axes);
    const outShape = this.shape.filter((_, idx) => !axesSet.has(idx));
    const out = new Tensor(outShape, {
      requiresGrad: this.requiresGrad,
      tape
    });

    const wasmEnabled =
      process.env.TENSOR_WASM_DISABLE !== "1" &&
      process.env.TENSOR_WASM_REDUCE !== "0";
    const backend = wasmEnabled ? getMatmulBackend() : null;
    const axisIsLast = axes.length === 1 && axes[0] === this.shape.length - 1;
    const canWasm =
      backend &&
      backend.exports.reduce_sum &&
      axisIsLast &&
      (this.shape.length === 1 || this.shape.length === 2);

    if (canWasm) {
      const rows = this.shape.length === 1 ? 1 : this.shape[0];
      const cols = this.shape.length === 1 ? this.shape[0] : this.shape[1];
      const reduced = reduceSumWasm(backend, this.data, rows, cols);
      if (outShape.length === 0) {
        out.data[0] = reduced[0];
      } else {
        out.data.set(reduced);
      }
    } else {
      for (let i = 0; i < this.size; i++) {
        const coords = unflattenIndex(i, this.shape, this.strides);
        const outCoords = coords.filter((_, idx) => !axesSet.has(idx));
        const outIndex =
          outCoords.length === 0 ? 0 : flattenIndex(outCoords, out.strides);
        out.data[outIndex] += this.data[i];
      }
    }

    if (this.requiresGrad && tape) {
      tape.add({
        backward: () => {
          for (let i = 0; i < this.size; i++) {
            const coords = unflattenIndex(i, this.shape, this.strides);
            const outCoords = coords.filter((_, idx) => !axesSet.has(idx));
            const outIndex =
              outCoords.length === 0 ? 0 : flattenIndex(outCoords, out.strides);
            this.grad[i] += out.grad[outIndex];
          }
        }
      });
    }

    return out;
  }

  mean(axis?: number | number[]): Tensor {
    if (axis === undefined) {
      return this.sum().mulScalar(1 / this.size);
    }
    const axes = normalizeAxes(axis, this.shape.length);
    const count = axes.reduce((acc, dim) => acc * this.shape[dim], 1);
    const wasmEnabled =
      process.env.TENSOR_WASM_DISABLE !== "1" &&
      process.env.TENSOR_WASM_REDUCE !== "0";
    const backend = wasmEnabled ? getMatmulBackend() : null;
    const axisIsLast = axes.length === 1 && axes[0] === this.shape.length - 1;
    const canWasm =
      backend &&
      backend.exports.reduce_mean &&
      axisIsLast &&
      (this.shape.length === 1 || this.shape.length === 2);

    if (canWasm) {
      const rows = this.shape.length === 1 ? 1 : this.shape[0];
      const cols = this.shape.length === 1 ? this.shape[0] : this.shape[1];
      const reduced = reduceMeanWasm(backend, this.data, rows, cols);
      const outShape = this.shape.filter((_, idx) => idx !== axes[0]);
      const out = new Tensor(outShape, {
        requiresGrad: this.requiresGrad,
        tape: resolveTape(this)
      });
      if (outShape.length === 0) {
        out.data[0] = reduced[0];
      } else {
        out.data.set(reduced);
      }

      if (this.requiresGrad && out.tape) {
        out.tape.add({
          backward: () => {
            const scale = 1 / count;
            for (let i = 0; i < this.size; i++) {
              const coords = unflattenIndex(i, this.shape, this.strides);
              const outCoords = coords.filter((_, idx) => idx !== axes[0]);
              const outIndex =
                outCoords.length === 0 ? 0 : flattenIndex(outCoords, out.strides);
              this.grad[i] += out.grad[outIndex] * scale;
            }
          }
        });
      }
      return out;
    }

    return this.sum(axes).mulScalar(1 / count);
  }

  max(axis?: number | number[]): Tensor {
    const tape = resolveTape(this);
    if (axis === undefined) {
      const out = new Tensor([], {
        requiresGrad: this.requiresGrad,
        tape
      });
      let maxVal = this.data[0];
      for (let i = 1; i < this.size; i++) {
        if (this.data[i] > maxVal) maxVal = this.data[i];
      }
      out.data[0] = maxVal;
      if (this.requiresGrad && tape) {
        tape.add({
          backward: () => {
            let count = 0;
            for (let i = 0; i < this.size; i++) {
              if (this.data[i] === maxVal) count += 1;
            }
            const share = count === 0 ? 0 : out.grad[0] / count;
            for (let i = 0; i < this.size; i++) {
              if (this.data[i] === maxVal) this.grad[i] += share;
            }
          }
        });
      }
      return out;
    }

    if (this.shape.length === 0) {
      throw new Error("Cannot reduce scalar tensor by axis");
    }

    const axes = normalizeAxes(axis, this.shape.length);
    const axesSet = new Set(axes);
    const outShape = this.shape.filter((_, idx) => !axesSet.has(idx));
    const out = new Tensor(outShape, {
      requiresGrad: this.requiresGrad,
      tape
    });
    for (let i = 0; i < out.size; i++) {
      out.data[i] = -Infinity;
    }

    const wasmEnabled =
      process.env.TENSOR_WASM_DISABLE !== "1" &&
      process.env.TENSOR_WASM_REDUCE !== "0";
    const backend = wasmEnabled ? getMatmulBackend() : null;
    const axisIsLast = axes.length === 1 && axes[0] === this.shape.length - 1;
    const canWasm =
      backend &&
      backend.exports.reduce_max &&
      axisIsLast &&
      (this.shape.length === 1 || this.shape.length === 2);

    if (canWasm) {
      const rows = this.shape.length === 1 ? 1 : this.shape[0];
      const cols = this.shape.length === 1 ? this.shape[0] : this.shape[1];
      const reduced = reduceMaxWasm(backend, this.data, rows, cols);
      if (outShape.length === 0) {
        out.data[0] = reduced[0];
      } else {
        out.data.set(reduced);
      }
    } else {
      for (let i = 0; i < this.size; i++) {
        const coords = unflattenIndex(i, this.shape, this.strides);
        const outCoords = coords.filter((_, idx) => !axesSet.has(idx));
        const outIndex =
          outCoords.length === 0 ? 0 : flattenIndex(outCoords, out.strides);
        if (this.data[i] > out.data[outIndex]) {
          out.data[outIndex] = this.data[i];
        }
      }
    }

    if (this.requiresGrad && tape) {
      tape.add({
        backward: () => {
          const counts = new Array(out.size).fill(0);
          for (let i = 0; i < this.size; i++) {
            const coords = unflattenIndex(i, this.shape, this.strides);
            const outCoords = coords.filter((_, idx) => !axesSet.has(idx));
            const outIndex =
              outCoords.length === 0 ? 0 : flattenIndex(outCoords, out.strides);
            if (this.data[i] === out.data[outIndex]) {
              counts[outIndex] += 1;
            }
          }

          for (let i = 0; i < this.size; i++) {
            const coords = unflattenIndex(i, this.shape, this.strides);
            const outCoords = coords.filter((_, idx) => !axesSet.has(idx));
            const outIndex =
              outCoords.length === 0 ? 0 : flattenIndex(outCoords, out.strides);
            if (this.data[i] === out.data[outIndex]) {
              const share =
                counts[outIndex] === 0 ? 0 : out.grad[outIndex] / counts[outIndex];
              this.grad[i] += share;
            }
          }
        }
      });
    }

    return out;
  }

  layerNorm(gamma?: Tensor, beta?: Tensor, eps = 1e-5): Tensor {
    if (this.shape.length === 0) {
      throw new Error("layerNorm requires at least one dimension");
    }

    const featureDim = this.shape[this.shape.length - 1];
    const rows = this.size / featureDim;

    const needsGrad =
      this.requiresGrad || gamma?.requiresGrad || beta?.requiresGrad;
    const wasmEnabled =
      process.env.TENSOR_WASM_DISABLE !== "1" &&
      process.env.TENSOR_WASM_LAYERNORM !== "0";
    const backend = wasmEnabled ? getMatmulBackend() : null;
    const forwardWasm = backend && backend.exports.layernorm;
    const backwardWasm =
      process.env.TENSOR_WASM_DISABLE !== "1" &&
      process.env.TENSOR_WASM_BACKWARD !== "0" &&
      backend &&
      backend.exports.layernorm_backward;

    if (!needsGrad && forwardWasm) {
      const gammaData = gamma
        ? gamma.data
        : new Float32Array(featureDim).fill(1);
      const betaData = beta
        ? beta.data
        : new Float32Array(featureDim).fill(0);
      const result = layerNormWasm(
        backend!,
        this.data,
        gammaData,
        betaData,
        rows,
        featureDim,
        eps
      );
      return new Tensor(this.shape, {
        data: result,
        requiresGrad: false,
        reuseData: true
      });
    }

    if (needsGrad && forwardWasm && backwardWasm) {
      const tape = resolveTape(this, gamma) ?? resolveTape(this, beta);
      const out = new Tensor(this.shape, {
        requiresGrad: true,
        tape
      });
      const gammaData = gamma
        ? gamma.data
        : new Float32Array(featureDim).fill(1);
      const betaData = beta
        ? beta.data
        : new Float32Array(featureDim).fill(0);
      const result = layerNormWasm(
        backend!,
        this.data,
        gammaData,
        betaData,
        rows,
        featureDim,
        eps
      );
      out.data.set(result);

      if (tape) {
        tape.add({
          backward: () => {
            const grads = layerNormBackwardWasm(
              backend!,
              this.data,
              gammaData,
              out.grad,
              rows,
              featureDim,
              eps
            );
            if (this.requiresGrad) {
              for (let i = 0; i < this.grad.length; i++) {
                this.grad[i] += grads.gradInput[i];
              }
            }
            if (gamma?.requiresGrad) {
              for (let i = 0; i < gamma.grad.length; i++) {
                gamma.grad[i] += grads.gradGamma[i];
              }
            }
            if (beta?.requiresGrad) {
              for (let i = 0; i < beta.grad.length; i++) {
                beta.grad[i] += grads.gradBeta[i];
              }
            }
          }
        });
      }
      return out;
    }

    const mean = this.mean(this.shape.length - 1);
    const meanExpanded = mean.view([...mean.shape, 1]);
    const centered = this.sub(meanExpanded);
    const variance = centered.mul(centered).mean(this.shape.length - 1);
    const varianceExpanded = variance.view([...variance.shape, 1]);
    const norm = centered.div(varianceExpanded.addScalar(eps).sqrt());
    let out = norm;
    if (gamma) {
      out = out.mul(gamma);
    }
    if (beta) {
      out = out.add(beta);
    }
    return out;
  }

  exp(): Tensor {
    const tape = resolveTape(this);
    const out = new Tensor(this.shape, {
      requiresGrad: this.requiresGrad,
      tape
    });
    for (let i = 0; i < this.size; i++) {
      out.data[i] = Math.exp(this.data[i]);
    }
    if (this.requiresGrad && tape) {
      tape.add({
        backward: () => {
          for (let i = 0; i < out.size; i++) {
            this.grad[i] += out.data[i] * out.grad[i];
          }
        }
      });
    }
    return out;
  }

  log(): Tensor {
    const tape = resolveTape(this);
    const out = new Tensor(this.shape, {
      requiresGrad: this.requiresGrad,
      tape
    });
    for (let i = 0; i < this.size; i++) {
      out.data[i] = Math.log(this.data[i]);
    }
    if (this.requiresGrad && tape) {
      tape.add({
        backward: () => {
          for (let i = 0; i < out.size; i++) {
            this.grad[i] += (1 / this.data[i]) * out.grad[i];
          }
        }
      });
    }
    return out;
  }

  tanh(): Tensor {
    const tape = resolveTape(this);
    const out = new Tensor(this.shape, {
      requiresGrad: this.requiresGrad,
      tape
    });
    for (let i = 0; i < this.size; i++) {
      out.data[i] = Math.tanh(this.data[i]);
    }
    if (this.requiresGrad && tape) {
      tape.add({
        backward: () => {
          for (let i = 0; i < out.size; i++) {
            const t = out.data[i];
            this.grad[i] += (1 - t * t) * out.grad[i];
          }
        }
      });
    }
    return out;
  }

  // https://arxiv.org/pdf/1606.08415v3
  gelu(): Tensor {
    const tape = resolveTape(this);
    const wasmEnabled =
      process.env.TENSOR_WASM_DISABLE !== "1" &&
      process.env.TENSOR_WASM_GELU !== "0";
    const backend = wasmEnabled ? getMatmulBackend() : null;
    const canWasm = backend && backend.exports.gelu && !this.requiresGrad;
    const out = canWasm
      ? new Tensor(this.shape, {
          data: geluWasm(backend!, this.data),
          requiresGrad: false,
          reuseData: true
        })
      : new Tensor(this.shape, {
          requiresGrad: this.requiresGrad,
          tape
        });
    if (!canWasm) {
      const a = Math.sqrt(2 / Math.PI);
      for (let i = 0; i < this.size; i++) {
        const x = this.data[i];
        const x3 = x * x * x;
        const inner = a * (x + 0.044715 * x3);
        out.data[i] = 0.5 * x * (1 + Math.tanh(inner));
      }
    }
    if (this.requiresGrad && tape) {
      tape.add({
        backward: () => {
          const aLocal = Math.sqrt(2 / Math.PI);
          for (let i = 0; i < out.size; i++) {
            const x = this.data[i];
            const x2 = x * x;
            const x3 = x2 * x;
            const inner = aLocal * (x + 0.044715 * x3);
            const t = Math.tanh(inner);
            const sech2 = 1 - t * t;
            const innerGrad = aLocal * (1 + 3 * 0.044715 * x2);
            const grad = 0.5 * (1 + t) + 0.5 * x * sech2 * innerGrad;
            this.grad[i] += grad * out.grad[i];
          }
        }
      });
    }
    return out;
  }

  sigmoid(): Tensor {
    const tape = resolveTape(this);
    const out = new Tensor(this.shape, {
      requiresGrad: this.requiresGrad,
      tape
    });
    for (let i = 0; i < this.size; i++) {
      const val = this.data[i];
      out.data[i] = 1 / (1 + Math.exp(-val));
    }
    if (this.requiresGrad && tape) {
      tape.add({
        backward: () => {
          for (let i = 0; i < out.size; i++) {
            const s = out.data[i];
            this.grad[i] += s * (1 - s) * out.grad[i];
          }
        }
      });
    }
    return out;
  }

  sqrt(): Tensor {
    const tape = resolveTape(this);
    const out = new Tensor(this.shape, {
      requiresGrad: this.requiresGrad,
      tape
    });
    for (let i = 0; i < this.size; i++) {
      out.data[i] = Math.sqrt(this.data[i]);
    }
    if (this.requiresGrad && tape) {
      tape.add({
        backward: () => {
          for (let i = 0; i < out.size; i++) {
            const denom = out.data[i] === 0 ? 0 : 0.5 / out.data[i];
            this.grad[i] += denom * out.grad[i];
          }
        }
      });
    }
    return out;
  }

  softmax(dim?: number): Tensor {
    if (this.shape.length === 0) {
      throw new Error("softmax requires at least one dimension");
    }
    const axis = normalizeAxis(
      dim ?? this.shape.length - 1,
      this.shape.length
    );
    const tape = resolveTape(this);
    const wasmEnabled =
      process.env.TENSOR_WASM_DISABLE !== "1" &&
      process.env.TENSOR_WASM_SOFTMAX !== "0";
    const backend = wasmEnabled ? getMatmulBackend() : null;
    const canWasm =
      backend &&
      axis === this.shape.length - 1 &&
      (this.shape.length === 2 || this.shape.length === 1);
    const out = canWasm
      ? new Tensor(this.shape, {
          data: softmaxWasm(
            backend!,
            this.data,
            this.shape.length === 1 ? 1 : this.shape[0],
            this.shape.length === 1 ? this.shape[0] : this.shape[1]
          ),
          requiresGrad: this.requiresGrad,
          tape,
          reuseData: true
        })
      : new Tensor(this.shape, {
          requiresGrad: this.requiresGrad,
          tape
        });
    const axisSize = this.shape[axis];
    const baseShape = this.shape.filter((_, idx) => idx !== axis);
    const baseStrides = buildStrides(baseShape);
    const baseSize = computeSize(baseShape);

    if (canWasm) {
      if (this.requiresGrad && tape) {
        tape.add({
          backward: () => {
            const backwardWasmEnabled =
              process.env.TENSOR_WASM_DISABLE !== "1" &&
              process.env.TENSOR_WASM_BACKWARD !== "0";
            const backend = backwardWasmEnabled ? getMatmulBackend() : null;
            if (backend && backend.exports.softmax_backward) {
              const rows = this.shape.length === 1 ? 1 : this.shape[0];
              const cols = this.shape.length === 1 ? this.shape[0] : this.shape[1];
              const gradInput = softmaxBackwardWasm(
                backend,
                out.data,
                out.grad,
                rows,
                cols
              );
              for (let i = 0; i < this.grad.length; i++) {
                this.grad[i] += gradInput[i];
              }
              return;
            }

            for (let baseIndex = 0; baseIndex < baseSize; baseIndex++) {
              const baseCoords = unflattenIndex(baseIndex, baseShape, baseStrides);
              let dot = 0;
              for (let axisVal = 0; axisVal < axisSize; axisVal++) {
                const coords = new Array(this.shape.length);
                let cursor = 0;
                for (let idx = 0; idx < this.shape.length; idx++) {
                  if (idx === axis) {
                    coords[idx] = axisVal;
                  } else {
                    coords[idx] = baseCoords[cursor++];
                  }
                }
                const index = flattenIndex(coords, this.strides);
                dot += out.grad[index] * out.data[index];
              }
              for (let axisVal = 0; axisVal < axisSize; axisVal++) {
                const coords = new Array(this.shape.length);
                let cursor = 0;
                for (let idx = 0; idx < this.shape.length; idx++) {
                  if (idx === axis) {
                    coords[idx] = axisVal;
                  } else {
                    coords[idx] = baseCoords[cursor++];
                  }
                }
                const index = flattenIndex(coords, this.strides);
                this.grad[index] += out.data[index] * (out.grad[index] - dot);
              }
            }
          }
        });
      }
      return out;
    }

    for (let baseIndex = 0; baseIndex < baseSize; baseIndex++) {
      const baseCoords = unflattenIndex(baseIndex, baseShape, baseStrides);
      let maxVal = -Infinity;
      for (let axisVal = 0; axisVal < axisSize; axisVal++) {
        const coords = new Array(this.shape.length);
        let cursor = 0;
        for (let idx = 0; idx < this.shape.length; idx++) {
          if (idx === axis) {
            coords[idx] = axisVal;
          } else {
            coords[idx] = baseCoords[cursor++];
          }
        }
        const index = flattenIndex(coords, this.strides);
        const val = this.data[index];
        if (val > maxVal) maxVal = val;
      }

      let sumExp = 0;
      for (let axisVal = 0; axisVal < axisSize; axisVal++) {
        const coords = new Array(this.shape.length);
        let cursor = 0;
        for (let idx = 0; idx < this.shape.length; idx++) {
          if (idx === axis) {
            coords[idx] = axisVal;
          } else {
            coords[idx] = baseCoords[cursor++];
          }
        }
        const index = flattenIndex(coords, this.strides);
        const expVal = Math.exp(this.data[index] - maxVal);
        out.data[index] = expVal;
        sumExp += expVal;
      }

      for (let axisVal = 0; axisVal < axisSize; axisVal++) {
        const coords = new Array(this.shape.length);
        let cursor = 0;
        for (let idx = 0; idx < this.shape.length; idx++) {
          if (idx === axis) {
            coords[idx] = axisVal;
          } else {
            coords[idx] = baseCoords[cursor++];
          }
        }
        const index = flattenIndex(coords, this.strides);
        out.data[index] /= sumExp;
      }
    }

    if (this.requiresGrad && tape) {
      tape.add({
        backward: () => {
          for (let baseIndex = 0; baseIndex < baseSize; baseIndex++) {
            const baseCoords = unflattenIndex(baseIndex, baseShape, baseStrides);
            let dot = 0;
            for (let axisVal = 0; axisVal < axisSize; axisVal++) {
              const coords = new Array(this.shape.length);
              let cursor = 0;
              for (let idx = 0; idx < this.shape.length; idx++) {
                if (idx === axis) {
                  coords[idx] = axisVal;
                } else {
                  coords[idx] = baseCoords[cursor++];
                }
              }
              const index = flattenIndex(coords, this.strides);
              dot += out.grad[index] * out.data[index];
            }
            for (let axisVal = 0; axisVal < axisSize; axisVal++) {
              const coords = new Array(this.shape.length);
              let cursor = 0;
              for (let idx = 0; idx < this.shape.length; idx++) {
                if (idx === axis) {
                  coords[idx] = axisVal;
                } else {
                  coords[idx] = baseCoords[cursor++];
                }
              }
              const index = flattenIndex(coords, this.strides);
              this.grad[index] += out.data[index] * (out.grad[index] - dot);
            }
          }
        }
      });
    }

    return out;
  }

  logsoftmax(dim?: number): Tensor {
    if (this.shape.length === 0) {
      throw new Error("logsoftmax requires at least one dimension");
    }
    const axis = normalizeAxis(
      dim ?? this.shape.length - 1,
      this.shape.length
    );
    const tape = resolveTape(this);
    const wasmEnabled =
      process.env.TENSOR_WASM_DISABLE !== "1" &&
      process.env.TENSOR_WASM_LOGSOFTMAX !== "0";
    const backend = wasmEnabled ? getMatmulBackend() : null;
    const canWasm =
      backend &&
      axis === this.shape.length - 1 &&
      (this.shape.length === 2 || this.shape.length === 1);
    const out = canWasm
      ? new Tensor(this.shape, {
          data: logsoftmaxWasm(
            backend!,
            this.data,
            this.shape.length === 1 ? 1 : this.shape[0],
            this.shape.length === 1 ? this.shape[0] : this.shape[1]
          ),
          requiresGrad: this.requiresGrad,
          tape,
          reuseData: true
        })
      : new Tensor(this.shape, {
          requiresGrad: this.requiresGrad,
          tape
        });
    const axisSize = this.shape[axis];
    const baseShape = this.shape.filter((_, idx) => idx !== axis);
    const baseStrides = buildStrides(baseShape);
    const baseSize = computeSize(baseShape);

    if (canWasm) {
      if (this.requiresGrad && tape) {
        tape.add({
          backward: () => {
            const backwardWasmEnabled =
              process.env.TENSOR_WASM_DISABLE !== "1" &&
              process.env.TENSOR_WASM_BACKWARD !== "0";
            const backend = backwardWasmEnabled ? getMatmulBackend() : null;
            if (backend && backend.exports.logsoftmax_backward) {
              const rows = this.shape.length === 1 ? 1 : this.shape[0];
              const cols = this.shape.length === 1 ? this.shape[0] : this.shape[1];
              const gradInput = logsoftmaxBackwardWasm(
                backend,
                out.data,
                out.grad,
                rows,
                cols
              );
              for (let i = 0; i < this.grad.length; i++) {
                this.grad[i] += gradInput[i];
              }
              return;
            }

            for (let baseIndex = 0; baseIndex < baseSize; baseIndex++) {
              const baseCoords = unflattenIndex(baseIndex, baseShape, baseStrides);
              let sumGrad = 0;
              for (let axisVal = 0; axisVal < axisSize; axisVal++) {
                const coords = new Array(this.shape.length);
                let cursor = 0;
                for (let idx = 0; idx < this.shape.length; idx++) {
                  if (idx === axis) {
                    coords[idx] = axisVal;
                  } else {
                    coords[idx] = baseCoords[cursor++];
                  }
                }
                const index = flattenIndex(coords, this.strides);
                sumGrad += out.grad[index];
              }
              for (let axisVal = 0; axisVal < axisSize; axisVal++) {
                const coords = new Array(this.shape.length);
                let cursor = 0;
                for (let idx = 0; idx < this.shape.length; idx++) {
                  if (idx === axis) {
                    coords[idx] = axisVal;
                  } else {
                    coords[idx] = baseCoords[cursor++];
                  }
                }
                const index = flattenIndex(coords, this.strides);
                const softmaxVal = Math.exp(out.data[index]);
                this.grad[index] += out.grad[index] - softmaxVal * sumGrad;
              }
            }
          }
        });
      }
      return out;
    }

    for (let baseIndex = 0; baseIndex < baseSize; baseIndex++) {
      const baseCoords = unflattenIndex(baseIndex, baseShape, baseStrides);
      let maxVal = -Infinity;
      for (let axisVal = 0; axisVal < axisSize; axisVal++) {
        const coords = new Array(this.shape.length);
        let cursor = 0;
        for (let idx = 0; idx < this.shape.length; idx++) {
          if (idx === axis) {
            coords[idx] = axisVal;
          } else {
            coords[idx] = baseCoords[cursor++];
          }
        }
        const index = flattenIndex(coords, this.strides);
        const val = this.data[index];
        if (val > maxVal) maxVal = val;
      }

      let sumExp = 0;
      for (let axisVal = 0; axisVal < axisSize; axisVal++) {
        const coords = new Array(this.shape.length);
        let cursor = 0;
        for (let idx = 0; idx < this.shape.length; idx++) {
          if (idx === axis) {
            coords[idx] = axisVal;
          } else {
            coords[idx] = baseCoords[cursor++];
          }
        }
        const index = flattenIndex(coords, this.strides);
        sumExp += Math.exp(this.data[index] - maxVal);
      }
      const logSumExp = Math.log(sumExp) + maxVal;
      for (let axisVal = 0; axisVal < axisSize; axisVal++) {
        const coords = new Array(this.shape.length);
        let cursor = 0;
        for (let idx = 0; idx < this.shape.length; idx++) {
          if (idx === axis) {
            coords[idx] = axisVal;
          } else {
            coords[idx] = baseCoords[cursor++];
          }
        }
        const index = flattenIndex(coords, this.strides);
        out.data[index] = this.data[index] - logSumExp;
      }
    }

    if (this.requiresGrad && tape) {
      tape.add({
        backward: () => {
          for (let baseIndex = 0; baseIndex < baseSize; baseIndex++) {
            const baseCoords = unflattenIndex(baseIndex, baseShape, baseStrides);
            let sumGrad = 0;
            for (let axisVal = 0; axisVal < axisSize; axisVal++) {
              const coords = new Array(this.shape.length);
              let cursor = 0;
              for (let idx = 0; idx < this.shape.length; idx++) {
                if (idx === axis) {
                  coords[idx] = axisVal;
                } else {
                  coords[idx] = baseCoords[cursor++];
                }
              }
              const index = flattenIndex(coords, this.strides);
              sumGrad += out.grad[index];
            }
            for (let axisVal = 0; axisVal < axisSize; axisVal++) {
              const coords = new Array(this.shape.length);
              let cursor = 0;
              for (let idx = 0; idx < this.shape.length; idx++) {
                if (idx === axis) {
                  coords[idx] = axisVal;
                } else {
                  coords[idx] = baseCoords[cursor++];
                }
              }
              const index = flattenIndex(coords, this.strides);
              const softmaxVal = Math.exp(out.data[index]);
              this.grad[index] += out.grad[index] - softmaxVal * sumGrad;
            }
          }
        }
      });
    }

    return out;
  }

  matmul(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error("matmul requires 2D tensors");
    }
    const [m, k] = this.shape;
    const [kOther, n] = other.shape;
    if (k !== kOther) {
      throw new Error("matmul requires inner dimensions to match");
    }

    const tape = resolveTape(this, other);
    const requiresGrad = this.requiresGrad || other.requiresGrad;
    const backend = getMatmulBackend();
    const out = backend
      ? new Tensor([m, n], {
          data: matmulWasm(backend, this.data, other.data, m, k, n),
          requiresGrad,
          tape,
          reuseData: true
        })
      : new Tensor([m, n], { requiresGrad, tape });

    if (!backend) {
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          let acc = 0;
          for (let p = 0; p < k; p++) {
            const aIndex = i * this.strides[0] + p * this.strides[1];
            const bIndex = p * other.strides[0] + j * other.strides[1];
            acc += this.data[aIndex] * other.data[bIndex];
          }
          out.data[i * out.strides[0] + j * out.strides[1]] = acc;
        }
      }
    }

    if (requiresGrad && tape) {
      tape.add({
        backward: () => {
          const backwardWasmEnabled =
            process.env.TENSOR_WASM_DISABLE !== "1" &&
            process.env.TENSOR_WASM_BACKWARD !== "0";
          const backwardBackend = backwardWasmEnabled ? getMatmulBackend() : null;

          if (backwardBackend && backwardBackend.exports.matmul_backward) {
            const grads = matmulBackwardWasm(
              backwardBackend,
              this.data,
              other.data,
              out.grad,
              m,
              k,
              n
            );

            if (this.requiresGrad) {
              for (let i = 0; i < this.grad.length; i++) {
                this.grad[i] += grads.gradA[i];
              }
            }

            if (other.requiresGrad) {
              for (let i = 0; i < other.grad.length; i++) {
                other.grad[i] += grads.gradB[i];
              }
            }
            return;
          }

          if (this.requiresGrad) {
            for (let i = 0; i < m; i++) {
              for (let p = 0; p < k; p++) {
                let acc = 0;
                for (let j = 0; j < n; j++) {
                  const outIndex =
                    i * out.strides[0] + j * out.strides[1];
                  const bIndex =
                    p * other.strides[0] + j * other.strides[1];
                  acc += out.grad[outIndex] * other.data[bIndex];
                }
                const aIndex = i * this.strides[0] + p * this.strides[1];
                this.grad[aIndex] += acc;
              }
            }
          }

          if (other.requiresGrad) {
            for (let p = 0; p < k; p++) {
              for (let j = 0; j < n; j++) {
                let acc = 0;
                for (let i = 0; i < m; i++) {
                  const outIndex =
                    i * out.strides[0] + j * out.strides[1];
                  const aIndex = i * this.strides[0] + p * this.strides[1];
                  acc += this.data[aIndex] * out.grad[outIndex];
                }
                const bIndex =
                  p * other.strides[0] + j * other.strides[1];
                other.grad[bIndex] += acc;
              }
            }
          }
        }
      });
    }

    return out;
  }

  slice(dim: number, start?: number, end?: number, step = 1): Tensor {
    const axis = normalizeAxis(dim, this.shape.length);
    const length = this.shape[axis];

    const resolvedStep = step ?? 1;
    if (resolvedStep <= 0) {
      throw new Error("slice step must be positive");
    }

    const rawStart = start ?? 0;
    const rawEnd = end ?? length;

    const clamp = (value: number) => Math.min(Math.max(value, 0), length);
    const normalize = (value: number) =>
      value < 0 ? clamp(length + value) : clamp(value);

    const normalizedStart = normalize(rawStart);
    const normalizedEnd = normalize(rawEnd);

    const indices: number[] = [];
    for (let i = normalizedStart; i < normalizedEnd; i += resolvedStep) {
      indices.push(i);
    }

    const outShape = [...this.shape];
    outShape[axis] = indices.length;
    const tape = resolveTape(this);
    const out = new Tensor(outShape, {
      requiresGrad: this.requiresGrad,
      tape
    });

    for (let i = 0; i < out.size; i++) {
      const coords = unflattenIndex(i, outShape, out.strides);
      const srcCoords = coords.slice();
      srcCoords[axis] = indices[coords[axis]];
      const srcIndex = flattenIndex(srcCoords, this.strides);
      out.data[i] = this.data[srcIndex];
    }

    if (this.requiresGrad && tape) {
      tape.add({
        backward: () => {
          for (let i = 0; i < out.size; i++) {
            const coords = unflattenIndex(i, outShape, out.strides);
            const srcCoords = coords.slice();
            srcCoords[axis] = indices[coords[axis]];
            const srcIndex = flattenIndex(srcCoords, this.strides);
            this.grad[srcIndex] += out.grad[i];
          }
        }
      });
    }

    return out;
  }

  gather(dim: number, indices: Tensor): Tensor {
    const axis = normalizeAxis(dim, this.shape.length);
    const axisLength = this.shape[axis];
    const beforeDims = this.shape.slice(0, axis);
    const afterDims = this.shape.slice(axis + 1);
    const resultShape = [...beforeDims, ...indices.shape, ...afterDims];
    const tape = resolveTape(this);
    const out = new Tensor(resultShape, {
      requiresGrad: this.requiresGrad,
      tape
    });

    const srcCoords = new Array(this.shape.length).fill(0);
    const dstCoords = new Array(resultShape.length).fill(0);
    const indexCoords = new Array(indices.shape.length).fill(0);

    const normalizeIndex = (value: number): number => {
      if (!Number.isFinite(value)) {
        throw new Error("Index tensor contains non-finite values");
      }
      if (!Number.isInteger(value)) {
        throw new Error("Index tensor must contain integer values");
      }

      let normalized = value;
      if (normalized < 0) {
        normalized = axisLength + normalized;
      }
      if (normalized < 0 || normalized >= axisLength) {
        throw new Error(
          `Index ${value} is out of bounds for axis length ${axisLength}`
        );
      }
      return normalized;
    };

    const writeValue = (srcIndex: number, dstIndex: number) => {
      out.data[dstIndex] = this.data[srcIndex];
    };

    const fillAfter = (depth: number) => {
      if (depth === afterDims.length) {
        const srcIndex = flattenIndex(srcCoords, this.strides);
        const dstIndex = flattenIndex(dstCoords, out.strides);
        writeValue(srcIndex, dstIndex);
        return;
      }

      const srcAxis = axis + 1 + depth;
      const dstAxis = beforeDims.length + indices.shape.length + depth;
      for (let i = 0; i < afterDims[depth]; i++) {
        srcCoords[srcAxis] = i;
        dstCoords[dstAxis] = i;
        fillAfter(depth + 1);
      }
    };

    const fillIndices = (depth: number) => {
      if (depth === indices.shape.length) {
        const indexValue = normalizeIndex(
          indices.data[flattenIndex(indexCoords, indices.strides)]
        );
        srcCoords[axis] = indexValue;
        fillAfter(0);
        return;
      }

      const dstAxis = beforeDims.length + depth;
      for (let i = 0; i < indices.shape[depth]; i++) {
        indexCoords[depth] = i;
        dstCoords[dstAxis] = i;
        fillIndices(depth + 1);
      }
    };

    const fillBefore = (depth: number) => {
      if (depth === beforeDims.length) {
        fillIndices(0);
        return;
      }

      for (let i = 0; i < beforeDims[depth]; i++) {
        srcCoords[depth] = i;
        dstCoords[depth] = i;
        fillBefore(depth + 1);
      }
    };

    fillBefore(0);

    if (this.requiresGrad && tape) {
      tape.add({
        backward: () => {
          const addGrad = (srcIndex: number, dstIndex: number) => {
            this.grad[srcIndex] += out.grad[dstIndex];
          };

          const fillAfterGrad = (depth: number) => {
            if (depth === afterDims.length) {
              const srcIndex = flattenIndex(srcCoords, this.strides);
              const dstIndex = flattenIndex(dstCoords, out.strides);
              addGrad(srcIndex, dstIndex);
              return;
            }

            const srcAxis = axis + 1 + depth;
            const dstAxis = beforeDims.length + indices.shape.length + depth;
            for (let i = 0; i < afterDims[depth]; i++) {
              srcCoords[srcAxis] = i;
              dstCoords[dstAxis] = i;
              fillAfterGrad(depth + 1);
            }
          };

          const fillIndicesGrad = (depth: number) => {
            if (depth === indices.shape.length) {
              const indexValue = normalizeIndex(
                indices.data[flattenIndex(indexCoords, indices.strides)]
              );
              srcCoords[axis] = indexValue;
              fillAfterGrad(0);
              return;
            }

            const dstAxis = beforeDims.length + depth;
            for (let i = 0; i < indices.shape[depth]; i++) {
              indexCoords[depth] = i;
              dstCoords[dstAxis] = i;
              fillIndicesGrad(depth + 1);
            }
          };

          const fillBeforeGrad = (depth: number) => {
            if (depth === beforeDims.length) {
              fillIndicesGrad(0);
              return;
            }

            for (let i = 0; i < beforeDims[depth]; i++) {
              srcCoords[depth] = i;
              dstCoords[depth] = i;
              fillBeforeGrad(depth + 1);
            }
          };

          fillBeforeGrad(0);
        }
      });
    }

    return out;
  }

  permute(order: number[]): Tensor {
    if (order.length !== this.shape.length) {
      throw new Error("permute requires a full axis order");
    }
    const seen = new Set(order);
    if (seen.size !== order.length) {
      throw new Error("permute order must be a permutation");
    }
    for (const axis of order) {
      if (axis < 0 || axis >= this.shape.length) {
        throw new Error("permute axis out of bounds");
      }
    }

    const nextShape = order.map((axis) => this.shape[axis]);
    const tape = resolveTape(this);
    const out = new Tensor(nextShape, {
      requiresGrad: this.requiresGrad,
      tape
    });

    for (let i = 0; i < out.size; i++) {
      const outCoords = unflattenIndex(i, nextShape, out.strides);
      const srcCoords = new Array(this.shape.length);
      for (let axis = 0; axis < order.length; axis++) {
        srcCoords[order[axis]] = outCoords[axis];
      }
      const srcIndex = flattenIndex(srcCoords, this.strides);
      out.data[i] = this.data[srcIndex];
    }

    if (this.requiresGrad && tape) {
      tape.add({
        backward: () => {
          for (let i = 0; i < out.size; i++) {
            const outCoords = unflattenIndex(i, nextShape, out.strides);
            const srcCoords = new Array(this.shape.length);
            for (let axis = 0; axis < order.length; axis++) {
              srcCoords[order[axis]] = outCoords[axis];
            }
            const srcIndex = flattenIndex(srcCoords, this.strides);
            this.grad[srcIndex] += out.grad[i];
          }
        }
      });
    }

    return out;
  }

  transpose(dim0 = 0, dim1 = 1): Tensor {
    if (this.shape.length < 2) {
      throw new Error("transpose requires at least two dimensions");
    }
    const order = this.shape.map((_, idx) => idx);
    const tmp = order[dim0];
    order[dim0] = order[dim1];
    order[dim1] = tmp;
    return this.permute(order);
  }

  static concat(dim: number, tensors: Tensor[]): Tensor {
    if (tensors.length === 0) {
      throw new Error("concat requires at least one tensor");
    }
    const ref = tensors[0];
    const axis = normalizeAxis(dim, ref.shape.length);
    const tape = resolveTapeList(tensors);
    const requiresGrad = tensors.some((t) => t.requiresGrad);

    tensors.forEach((tensor) => {
      if (tensor.shape.length !== ref.shape.length) {
        throw new Error("concat requires matching ranks");
      }
      tensor.shape.forEach((size, idx) => {
        if (idx !== axis && size !== ref.shape[idx]) {
          throw new Error("concat requires matching shapes");
        }
      });
    });

    const outShape = [...ref.shape];
    outShape[axis] = tensors.reduce((acc, t) => acc + t.shape[axis], 0);
    const out = new Tensor(outShape, { requiresGrad, tape });

    let offset = 0;
    for (const tensor of tensors) {
      for (let i = 0; i < tensor.size; i++) {
        const coords = unflattenIndex(i, tensor.shape, tensor.strides);
        const dstCoords = coords.slice();
        dstCoords[axis] += offset;
        const dstIndex = flattenIndex(dstCoords, out.strides);
        out.data[dstIndex] = tensor.data[i];
      }
      offset += tensor.shape[axis];
    }

    if (requiresGrad && tape) {
      tape.add({
        backward: () => {
          let gradOffset = 0;
          for (const tensor of tensors) {
            if (!tensor.requiresGrad) {
              gradOffset += tensor.shape[axis];
              continue;
            }
            for (let i = 0; i < tensor.size; i++) {
              const coords = unflattenIndex(i, tensor.shape, tensor.strides);
              const dstCoords = coords.slice();
              dstCoords[axis] += gradOffset;
              const dstIndex = flattenIndex(dstCoords, out.strides);
              tensor.grad[i] += out.grad[dstIndex];
            }
            gradOffset += tensor.shape[axis];
          }
        }
      });
    }

    return out;
  }

  static stack(dim: number, tensors: Tensor[]): Tensor {
    if (tensors.length === 0) {
      throw new Error("stack requires at least one tensor");
    }
    const ref = tensors[0];
    const rank = ref.shape.length;
    const axis = dim < 0 ? dim + rank + 1 : dim;
    if (axis < 0 || axis > rank) {
      throw new Error(`Axis ${dim} is out of bounds for rank ${rank + 1}`);
    }

    const tape = resolveTapeList(tensors);
    const requiresGrad = tensors.some((t) => t.requiresGrad);

    tensors.forEach((tensor) => {
      if (tensor.shape.length !== rank) {
        throw new Error("stack requires matching ranks");
      }
      tensor.shape.forEach((size, idx) => {
        if (size !== ref.shape[idx]) {
          throw new Error("stack requires matching shapes");
        }
      });
    });

    const outShape = [...ref.shape];
    outShape.splice(axis, 0, tensors.length);
    const out = new Tensor(outShape, { requiresGrad, tape });

    tensors.forEach((tensor, tensorIndex) => {
      for (let i = 0; i < tensor.size; i++) {
        const coords = unflattenIndex(i, tensor.shape, tensor.strides);
        const dstCoords = new Array(rank + 1);
        let cursor = 0;
        for (let axisIdx = 0; axisIdx < rank + 1; axisIdx++) {
          if (axisIdx === axis) {
            dstCoords[axisIdx] = tensorIndex;
          } else {
            dstCoords[axisIdx] = coords[cursor++];
          }
        }
        const dstIndex = flattenIndex(dstCoords, out.strides);
        out.data[dstIndex] = tensor.data[i];
      }
    });

    if (requiresGrad && tape) {
      tape.add({
        backward: () => {
          tensors.forEach((tensor, tensorIndex) => {
            if (!tensor.requiresGrad) return;
            for (let i = 0; i < tensor.size; i++) {
              const coords = unflattenIndex(i, tensor.shape, tensor.strides);
              const dstCoords = new Array(rank + 1);
              let cursor = 0;
              for (let axisIdx = 0; axisIdx < rank + 1; axisIdx++) {
                if (axisIdx === axis) {
                  dstCoords[axisIdx] = tensorIndex;
                } else {
                  dstCoords[axisIdx] = coords[cursor++];
                }
              }
              const dstIndex = flattenIndex(dstCoords, out.strides);
              tensor.grad[i] += out.grad[dstIndex];
            }
          });
        }
      });
    }

    return out;
  }
}
