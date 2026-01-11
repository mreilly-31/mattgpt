/**
 * Linear class applies an affine transformation
 * y = x @ W + b
 * 
 * the goal here is to transform input features into something more useful
 * given x = [x_1, x_2, x_3, ..., x_n] we want to compute m new features
 * the transformation we use to create m features from n input must:
 *   1. be differentiable
 *   2. mix all inputs together
 * 
 * so one neuron's computation is:
 *   y = w_1*x_1 + w_2*x_2 + ... + w_n*x_n + b
 * where w_i is how important the input x_i is and b shifts the output
 * so its really just a weighted sum plus a biaas
 * 
* for all neurons, we get x @ W + b
 */
import { ModelComponent } from "./ModelComponent";
import { Tape, Tensor } from "./Tensor";
import { initBias, initWeights } from "../matt-torch";

export class Linear extends ModelComponent {
  in_features: number;
  out_features: number;
  bias: Tensor | null;
  weights: Tensor;

  constructor(
    in_features: number,
    out_features: number,
    bias = true,
    tape?: Tape
  ) {
    super();
    if (in_features <= 0 || out_features <= 0) {
      throw new Error("in and out features must be greater than 0");
    }
    this.in_features = in_features;
    this.out_features = out_features;
    this.weights = initWeights(in_features, out_features, tape);
    this.bias = bias ? initBias(in_features, out_features, tape) : null;
  }

  forward(
    x: Tensor,
    activation: "none" | "relu" | "tanh" | "gelu" = "none"
  ): Tensor {
    if (x.shape.length < 2) {
      throw new Error("Linear.forward expects at least a 2D input tensor");
    }
    const inDim = x.shape[x.shape.length - 1];
    if (inDim !== this.in_features) {
      throw new Error(
        `Linear.forward expects in_features=${this.in_features}, got ${inDim}`
      );
    }

    const batchSize = x.size / inDim;
    const flat = x.view([batchSize, inDim]);
    let out = this.bias
      ? flat.matmulBiasAct(this.weights, this.bias, activation)
      : flat.matmul(this.weights);

    if (!this.bias) {
      if (activation === "relu") out = out.relu();
      if (activation === "tanh") out = out.tanh();
      if (activation === "gelu") out = out.gelu();
    }

    const outShape = [...x.shape.slice(0, -1), this.out_features];
    return out.view(outShape);
  }

  parameters(): Tensor[] {
    return this.bias ? [this.weights, this.bias] : [this.weights];
  }
}
