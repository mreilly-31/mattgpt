// https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
import { Value, Graph, Neuron, Layer, MultiLayerPerceptron } from "../structures";
import { zip, valuize, loss } from "../matt-torch";
import fs from "node:fs";
// const x_1 = new Value(2.0, [], "", "x_1");
// const x_2 = new Value(0.0, [], "", "x_2");

// const w_1 = new Value(-3.0, [], "", "w_1");
// const w_2 = new Value(1.0, [], "", "w_2");

// const bias = new Value(6.8813735870195432, [], "", "bias");
// const x_1w_1 = x_1.multiply(w_1);
// x_1w_1.label = "x1w1";
// const x_2w_2 = x_2.multiply(w_2);
// x_2w_2.label = "x2w2";

// const all = x_1w_1.add(x_2w_2);
// all.label = "all";

// const n = all.add(bias);
// n.label = 'n';

// const out = n.tanh();
// out.label = 'out';

// out.backward();

const g = new Graph();
// g.draw(out);


const x: Value[] = [new Value(2.0), new Value(3.0), new Value(-1.0)];

// const neuron = new Neuron(2);

// console.log(neuron.invoke(x));

// const layer = new Layer(2, 3);

// console.log(layer.invoke(x))

// this implements the mlp png included
const mlp = new MultiLayerPerceptron(3, [4, 4, 1]);
// const [result] = mlp.invoke(x);
// g.draw(result)



const xs = valuize([
  [2.0, 3.0, -1.0], //target 1.0
  [3.0, -1.0, 0.5], // target -1.0
  [0.5, 1.0, 1.0], // target -1.0
  [1.0, 1.0, -1.0] // target 1.0
]);
const targets = [new Value(1.0), new Value(-1.0), new Value(-1.0), new Value(1.0)];

// const dotGraph = g.draw(lossVal);
// fs.writeFileSync('dot.txt', dotGraph);
const step_size = 0.05; // learning rate
const iters = 100;

for (let i = 0; i < iters; i++) {
  let pred: Value[] = [];
  // complete a forward pass through the neural network
  for (const x of xs) {
    // extract from return which is [Value]
    pred.push(mlp.invoke(x)[0])
  }
  // calculate the loss
  const lossVal = loss(pred, targets);

  // we need to reset the gradients before performing the backward pass
  // otherwise, the gradients will be additive for every iteration
  for (const p of mlp.parameters()) {
    p.grad = 0.0;
  }

  lossVal.backward();

  for (const p of mlp.parameters()) {
    // time for some gradient descent
    // move p.data by a small amount along the gradient
    // step size is negated because the gradient is a vector pointing in the direction of increased loss
    // so we move down that vector (gradient descent) to minimize loss 
    p.data += -1.0 * step_size * p.grad;
  }

  console.log(i, lossVal.data);
  if (i === iters - 1) {
    console.log('FINAL ANSWER: ', pred);
  }
}