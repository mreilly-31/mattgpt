import fs from "node:fs";
import path from "node:path";

type MatmulExports = {
  memory: WebAssembly.Memory;
  matmul: (
    aPtr: number,
    bPtr: number,
    cPtr: number,
    m: number,
    k: number,
    n: number
  ) => void;
  matmul_backward?: (
    aPtr: number,
    bPtr: number,
    gradPtr: number,
    gradAPtr: number,
    gradBPtr: number,
    m: number,
    k: number,
    n: number
  ) => void;
  reduce_sum?: (
    inputPtr: number,
    outputPtr: number,
    rows: number,
    cols: number
  ) => void;
  reduce_mean?: (
    inputPtr: number,
    outputPtr: number,
    rows: number,
    cols: number
  ) => void;
  reduce_max?: (
    inputPtr: number,
    outputPtr: number,
    rows: number,
    cols: number
  ) => void;
  layernorm?: (
    inputPtr: number,
    gammaPtr: number,
    betaPtr: number,
    outputPtr: number,
    rows: number,
    cols: number,
    eps: number
  ) => void;
  layernorm_backward?: (
    inputPtr: number,
    gammaPtr: number,
    gradOutPtr: number,
    gradInputPtr: number,
    gradGammaPtr: number,
    gradBetaPtr: number,
    rows: number,
    cols: number,
    eps: number
  ) => void;
  mlp_fused?: (
    inputPtr: number,
    weightPtr: number,
    biasPtr: number,
    outputPtr: number,
    m: number,
    k: number,
    n: number,
    activation: number
  ) => void;
  gelu?: (
    inputPtr: number,
    outputPtr: number,
    len: number
  ) => void;
  softmax?: (
    inputPtr: number,
    outputPtr: number,
    rows: number,
    cols: number
  ) => void;
  softmax_backward?: (
    softmaxPtr: number,
    gradPtr: number,
    gradInputPtr: number,
    rows: number,
    cols: number
  ) => void;
  logsoftmax?: (
    inputPtr: number,
    outputPtr: number,
    rows: number,
    cols: number
  ) => void;
  logsoftmax_backward?: (
    logsoftmaxPtr: number,
    gradPtr: number,
    gradInputPtr: number,
    rows: number,
    cols: number
  ) => void;
};

type MatmulBackend = {
  exports: MatmulExports;
  buffer: Float32Array;
};

let backend: MatmulBackend | null = null;
let attemptedLoad = false;

const ensureCapacity = (exports: MatmulExports, floatsNeeded: number): void => {
  const bytesNeeded = floatsNeeded * 4; // float32 (4 bytes), so floatsNeeded * 4
  const currentBytes = exports.memory.buffer.byteLength;
  if (currentBytes >= bytesNeeded) return;
  const pagesNeeded = Math.ceil((bytesNeeded - currentBytes) / 65536);
  exports.memory.grow(pagesNeeded);
};

const ensureBuffer = (
  backend: MatmulBackend,
  floatsNeeded: number
): Float32Array => {
  ensureCapacity(backend.exports, floatsNeeded);
  backend.buffer = new Float32Array(backend.exports.memory.buffer);
  return backend.buffer;
};

const writeRange = (
  buffer: Float32Array,
  data: Float32Array,
  offset: number
): void => {
  buffer.set(data, offset);
};

const readRange = (
  buffer: Float32Array,
  offset: number,
  length: number
): Float32Array => buffer.slice(offset, offset + length);

export const tryLoadMatmulWasm = (): void => {
  if (attemptedLoad) return;
  attemptedLoad = true;

  if (process.env.TENSOR_WASM_DISABLE === "1") return;
  if (process.env.TENSOR_WASM_MATMUL === "0") return;

  const wasmPath =
    process.env.TENSOR_WASM_MATMUL_PATH ??
    path.join(__dirname, "wasm-math.wasm");
  if (!fs.existsSync(wasmPath)) return;

  const bytes = fs.readFileSync(wasmPath);
  const module = new WebAssembly.Module(bytes);
  const instance = new WebAssembly.Instance(module, {});
  const exports = instance.exports as unknown as MatmulExports;

  if (typeof exports.matmul !== "function" || !exports.memory) {
    throw new Error("wasm-math.wasm exports must include memory and matmul()");
  }

  backend = {
    exports,
    buffer: new Float32Array(exports.memory.buffer)
  };
};

export const getMatmulBackend = (): MatmulBackend | null => {
  tryLoadMatmulWasm();
  return backend;
};

export const matmulWasm = (
  backend: MatmulBackend,
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number
): Float32Array => {
  const aLen = m * k;
  const bLen = k * n;
  const cLen = m * n;
  const total = aLen + bLen + cLen;

  const buffer = ensureBuffer(backend, total);

  const aOffset = 0;
  const bOffset = aLen;
  const cOffset = aLen + bLen;

  writeRange(buffer, a, aOffset);
  writeRange(buffer, b, bOffset);

  backend.exports.matmul(
    aOffset * 4,
    bOffset * 4,
    cOffset * 4,
    m,
    k,
    n
  );

  return readRange(buffer, cOffset, cLen);
};

export const matmulBackwardWasm = (
  backend: MatmulBackend,
  a: Float32Array,
  b: Float32Array,
  grad: Float32Array,
  m: number,
  k: number,
  n: number
): { gradA: Float32Array; gradB: Float32Array } => {
  if (!backend.exports.matmul_backward) {
    throw new Error("matmul_backward is not exported by wasm-math.wasm");
  }

  const aLen = m * k;
  const bLen = k * n;
  const gradLen = m * n;
  const gradALen = m * k;
  const gradBLen = k * n;
  const total = aLen + bLen + gradLen + gradALen + gradBLen;

  const buffer = ensureBuffer(backend, total);

  const aOffset = 0;
  const bOffset = aLen;
  const gradOffset = aLen + bLen;
  const gradAOffset = gradOffset + gradLen;
  const gradBOffset = gradAOffset + gradALen;

  writeRange(buffer, a, aOffset);
  writeRange(buffer, b, bOffset);
  writeRange(buffer, grad, gradOffset);

  backend.exports.matmul_backward(
    aOffset * 4,
    bOffset * 4,
    gradOffset * 4,
    gradAOffset * 4,
    gradBOffset * 4,
    m,
    k,
    n
  );

  return {
    gradA: readRange(buffer, gradAOffset, gradALen),
    gradB: readRange(buffer, gradBOffset, gradBLen)
  };
};

export const reduceSumWasm = (
  backend: MatmulBackend,
  input: Float32Array,
  rows: number,
  cols: number
): Float32Array => {
  if (!backend.exports.reduce_sum) {
    throw new Error("reduce_sum is not exported by wasm-math.wasm");
  }

  const total = rows * cols;
  const buffer = ensureBuffer(backend, total + rows);
  const inOffset = 0;
  const outOffset = total;

  writeRange(buffer, input, inOffset);
  backend.exports.reduce_sum(inOffset * 4, outOffset * 4, rows, cols);

  return readRange(buffer, outOffset, rows);
};

export const reduceMeanWasm = (
  backend: MatmulBackend,
  input: Float32Array,
  rows: number,
  cols: number
): Float32Array => {
  if (!backend.exports.reduce_mean) {
    throw new Error("reduce_mean is not exported by wasm-math.wasm");
  }

  const total = rows * cols;
  const buffer = ensureBuffer(backend, total + rows);
  const inOffset = 0;
  const outOffset = total;

  writeRange(buffer, input, inOffset);
  backend.exports.reduce_mean(inOffset * 4, outOffset * 4, rows, cols);

  return readRange(buffer, outOffset, rows);
};

export const reduceMaxWasm = (
  backend: MatmulBackend,
  input: Float32Array,
  rows: number,
  cols: number
): Float32Array => {
  if (!backend.exports.reduce_max) {
    throw new Error("reduce_max is not exported by wasm-math.wasm");
  }

  const total = rows * cols;
  const buffer = ensureBuffer(backend, total + rows);
  const inOffset = 0;
  const outOffset = total;

  writeRange(buffer, input, inOffset);
  backend.exports.reduce_max(inOffset * 4, outOffset * 4, rows, cols);

  return readRange(buffer, outOffset, rows);
};

export const layerNormWasm = (
  backend: MatmulBackend,
  input: Float32Array,
  gamma: Float32Array,
  beta: Float32Array,
  rows: number,
  cols: number,
  eps: number
): Float32Array => {
  if (!backend.exports.layernorm) {
    throw new Error("layernorm is not exported by wasm-math.wasm");
  }

  const total = rows * cols;
  const buffer = ensureBuffer(backend, total * 2 + cols * 2);
  const inputOffset = 0;
  const gammaOffset = total;
  const betaOffset = gammaOffset + cols;
  const outputOffset = betaOffset + cols;

  writeRange(buffer, input, inputOffset);
  writeRange(buffer, gamma, gammaOffset);
  writeRange(buffer, beta, betaOffset);

  backend.exports.layernorm(
    inputOffset * 4,
    gammaOffset * 4,
    betaOffset * 4,
    outputOffset * 4,
    rows,
    cols,
    eps
  );

  return readRange(buffer, outputOffset, total);
};

export const layerNormBackwardWasm = (
  backend: MatmulBackend,
  input: Float32Array,
  gamma: Float32Array,
  gradOut: Float32Array,
  rows: number,
  cols: number,
  eps: number
): { gradInput: Float32Array; gradGamma: Float32Array; gradBeta: Float32Array } => {
  if (!backend.exports.layernorm_backward) {
    throw new Error("layernorm_backward is not exported by wasm-math.wasm");
  }

  const total = rows * cols;
  const buffer = ensureBuffer(backend, total * 3 + cols * 2);
  const inputOffset = 0;
  const gammaOffset = total;
  const gradOutOffset = gammaOffset + cols;
  const gradInputOffset = gradOutOffset + total;
  const gradGammaOffset = gradInputOffset + total;
  const gradBetaOffset = gradGammaOffset + cols;

  writeRange(buffer, input, inputOffset);
  writeRange(buffer, gamma, gammaOffset);
  writeRange(buffer, gradOut, gradOutOffset);

  backend.exports.layernorm_backward(
    inputOffset * 4,
    gammaOffset * 4,
    gradOutOffset * 4,
    gradInputOffset * 4,
    gradGammaOffset * 4,
    gradBetaOffset * 4,
    rows,
    cols,
    eps
  );

  return {
    gradInput: readRange(buffer, gradInputOffset, total),
    gradGamma: readRange(buffer, gradGammaOffset, cols),
    gradBeta: readRange(buffer, gradBetaOffset, cols)
  };
};

export const mlpFusedWasm = (
  backend: MatmulBackend,
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array,
  m: number,
  k: number,
  n: number,
  activation: number
): Float32Array => {
  if (!backend.exports.mlp_fused) {
    throw new Error("mlp_fused is not exported by wasm-math.wasm");
  }

  const inLen = m * k;
  const wLen = k * n;
  const bLen = n;
  const outLen = m * n;
  const total = inLen + wLen + bLen + outLen;
  const buffer = ensureBuffer(backend, total);

  const inputOffset = 0;
  const weightOffset = inLen;
  const biasOffset = weightOffset + wLen;
  const outputOffset = biasOffset + bLen;

  writeRange(buffer, input, inputOffset);
  writeRange(buffer, weight, weightOffset);
  writeRange(buffer, bias, biasOffset);

  backend.exports.mlp_fused(
    inputOffset * 4,
    weightOffset * 4,
    biasOffset * 4,
    outputOffset * 4,
    m,
    k,
    n,
    activation
  );

  return readRange(buffer, outputOffset, outLen);
};

export const geluWasm = (
  backend: MatmulBackend,
  input: Float32Array
): Float32Array => {
  if (!backend.exports.gelu) {
    throw new Error("gelu is not exported by wasm-math.wasm");
  }

  const len = input.length;
  const buffer = ensureBuffer(backend, len * 2);
  const inputOffset = 0;
  const outputOffset = len;

  writeRange(buffer, input, inputOffset);
  backend.exports.gelu(inputOffset * 4, outputOffset * 4, len);
  return readRange(buffer, outputOffset, len);
};

export const softmaxWasm = (
  backend: MatmulBackend,
  input: Float32Array,
  rows: number,
  cols: number
): Float32Array => {
  if (!backend.exports.softmax) {
    throw new Error("softmax is not exported by wasm-math.wasm");
  }

  const total = rows * cols;
  const buffer = ensureBuffer(backend, total * 2);

  const inOffset = 0;
  const outOffset = total;
  writeRange(buffer, input, inOffset);

  backend.exports.softmax(inOffset * 4, outOffset * 4, rows, cols);

  return readRange(buffer, outOffset, total);
};

export const logsoftmaxWasm = (
  backend: MatmulBackend,
  input: Float32Array,
  rows: number,
  cols: number
): Float32Array => {
  if (!backend.exports.logsoftmax) {
    throw new Error("logsoftmax is not exported by wasm-math.wasm");
  }

  const total = rows * cols;
  const buffer = ensureBuffer(backend, total * 2);

  const inOffset = 0;
  const outOffset = total;
  writeRange(buffer, input, inOffset);

  backend.exports.logsoftmax(inOffset * 4, outOffset * 4, rows, cols);

  return readRange(buffer, outOffset, total);
};

export const softmaxBackwardWasm = (
  backend: MatmulBackend,
  softmax: Float32Array,
  grad: Float32Array,
  rows: number,
  cols: number
): Float32Array => {
  if (!backend.exports.softmax_backward) {
    throw new Error("softmax_backward is not exported by wasm-math.wasm");
  }

  const total = rows * cols;
  const buffer = ensureBuffer(backend, total * 3);
  const softmaxOffset = 0;
  const gradOffset = total;
  const gradInputOffset = total * 2;

  writeRange(buffer, softmax, softmaxOffset);
  writeRange(buffer, grad, gradOffset);

  backend.exports.softmax_backward(
    softmaxOffset * 4,
    gradOffset * 4,
    gradInputOffset * 4,
    rows,
    cols
  );

  return readRange(buffer, gradInputOffset, total);
};

export const logsoftmaxBackwardWasm = (
  backend: MatmulBackend,
  logsoftmax: Float32Array,
  grad: Float32Array,
  rows: number,
  cols: number
): Float32Array => {
  if (!backend.exports.logsoftmax_backward) {
    throw new Error("logsoftmax_backward is not exported by wasm-math.wasm");
  }

  const total = rows * cols;
  const buffer = ensureBuffer(backend, total * 3);
  const logOffset = 0;
  const gradOffset = total;
  const gradInputOffset = total * 2;

  writeRange(buffer, logsoftmax, logOffset);
  writeRange(buffer, grad, gradOffset);

  backend.exports.logsoftmax_backward(
    logOffset * 4,
    gradOffset * 4,
    gradInputOffset * 4,
    rows,
    cols
  );

  return readRange(buffer, gradInputOffset, total);
};
