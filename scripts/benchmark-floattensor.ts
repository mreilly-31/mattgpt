import { FloatTensor, Tape, Tensor } from "../structures";

type Shape = number[];

const parseShape = (arg?: string): Shape => {
  if (!arg) return [128, 128];
  return arg.split(",").map((part) => {
    const value = Number(part.trim());
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`Invalid shape value: ${part}`);
    }
    return value;
  });
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

const createData = (size: number, seed = 1337): number[] => {
  let state = seed >>> 0;
  const next = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
  const data = new Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = next() * 2 - 1;
  }
  return data;
};

const nowMs = (): number => Number(process.hrtime.bigint()) / 1e6;

const time = (label: string, iterations: number, fn: () => void): void => {
  const start = nowMs();
  for (let i = 0; i < iterations; i++) {
    fn();
  }
  const duration = nowMs() - start;
  const perIter = duration / iterations;
  console.log(`${label}: ${duration.toFixed(2)}ms total (${perIter.toFixed(2)}ms/iter)`);
};

const main = () => {
  const shape = parseShape(process.argv[2]);
  const iterations = process.argv[3] ? Number(process.argv[3]) : 10;
  if (!Number.isInteger(iterations) || iterations <= 0) {
    throw new Error("iterations must be a positive integer");
  }

  const size = shape.reduce((acc, cur) => acc * cur, 1);
  const data = createData(size);
  const strides = buildStrides(shape);

  const makeTensor = () =>
    new Tensor(shape, (coords) => data[flattenIndex(coords, strides)]);

  const makeFloatTensor = (tape: Tape) =>
    FloatTensor.fromArray(shape, data, true, tape);

  console.log(`shape=${shape.join("x")} size=${size} iterations=${iterations}`);

  time("FloatTensor forward", iterations, () => {
    const tape = new Tape();
    const t = makeFloatTensor(tape);
    t.relu().sum();
  });

  time("FloatTensor forward+backward", iterations, () => {
    const tape = new Tape();
    const t = makeFloatTensor(tape);
    const out = t.relu().sum();
    out.backward();
  });

  time("Tensor forward", iterations, () => {
    const t = makeTensor();
    t.relu().sum();
  });

  time("Tensor forward+backward", iterations, () => {
    const t = makeTensor();
    const out = t.relu().sum();
    out.backward();
  });
};

main();
