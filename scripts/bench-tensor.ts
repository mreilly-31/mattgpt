import { Tensor, Tape } from "../structures/Tensor";

const now = (): number => Number(process.hrtime.bigint()) / 1e6;

const shapeA = [256, 256];
const shapeB = [256, 256];
const sizeA = shapeA[0] * shapeA[1];
const sizeB = shapeB[0] * shapeB[1];

const dataA = Array.from({ length: sizeA }, () => Math.random());
const dataB = Array.from({ length: sizeB }, () => Math.random());

const forward = (): void => {
  const tape = new Tape();
  const a = Tensor.fromArray(shapeA, dataA, true, tape);
  const b = Tensor.fromArray(shapeB, dataB, true, tape);
  a.matmul(b).relu().sum();
};

const forwardNoGrad = (): void => {
  const a = Tensor.fromArray(shapeA, dataA, false);
  const b = Tensor.fromArray(shapeB, dataB, false);
  a.matmul(b).relu().sum();
};

const forwardBackward = (): void => {
  const tape = new Tape();
  const a = Tensor.fromArray(shapeA, dataA, true, tape);
  const b = Tensor.fromArray(shapeB, dataB, true, tape);
  const out = a.matmul(b).relu().sum();
  out.backward();
};

const softmaxForward = (): void => {
  const a = Tensor.fromArray(shapeA, dataA, false);
  a.softmax(-1);
};

const timeOnce = (fn: () => void): number => {
  const start = now();
  fn();
  return now() - start;
};

const forwardMs = timeOnce(forward);
const forwardNoGradMs = timeOnce(forwardNoGrad);
const backwardMs = timeOnce(forwardBackward);
const softmaxMs = timeOnce(softmaxForward);

console.log(
  `RESULT forward_ms=${forwardMs.toFixed(4)} forward_nograd_ms=${forwardNoGradMs.toFixed(4)} backward_ms=${backwardMs.toFixed(4)} softmax_ms=${softmaxMs.toFixed(4)}`
);
