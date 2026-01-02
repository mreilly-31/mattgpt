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
};

type MatmulBackend = {
  exports: MatmulExports;
  buffer: Float32Array;
};

let backend: MatmulBackend | null = null;
let attemptedLoad = false;

const ensureCapacity = (exports: MatmulExports, floatsNeeded: number): void => {
  const bytesNeeded = floatsNeeded * 4;
  const currentBytes = exports.memory.buffer.byteLength;
  if (currentBytes >= bytesNeeded) return;
  const pagesNeeded = Math.ceil((bytesNeeded - currentBytes) / 65536);
  exports.memory.grow(pagesNeeded);
};

export const tryLoadMatmulWasm = (): void => {
  if (attemptedLoad) return;
  attemptedLoad = true;

  if (process.env.TENSOR_WASM_MATMUL !== "1") return;

  const wasmPath =
    process.env.TENSOR_WASM_MATMUL_PATH ??
    path.join(__dirname, "matmul.wasm");
  if (!fs.existsSync(wasmPath)) return;

  const bytes = fs.readFileSync(wasmPath);
  const module = new WebAssembly.Module(bytes);
  const instance = new WebAssembly.Instance(module, {});
  const exports = instance.exports as unknown as MatmulExports;

  if (typeof exports.matmul !== "function" || !exports.memory) {
    throw new Error("matmul.wasm exports must include memory and matmul()");
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

  ensureCapacity(backend.exports, total);
  backend.buffer = new Float32Array(backend.exports.memory.buffer);

  const aOffset = 0;
  const bOffset = aLen;
  const cOffset = aLen + bLen;

  backend.buffer.set(a, aOffset);
  backend.buffer.set(b, bOffset);

  backend.exports.matmul(
    aOffset * 4,
    bOffset * 4,
    cOffset * 4,
    m,
    k,
    n
  );

  return backend.buffer.slice(cOffset, cOffset + cLen);
};
