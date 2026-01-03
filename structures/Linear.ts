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
import { Tensor } from "./Tensor";
import { initBias, initWeights } from "../matt-torch";

export class Linear {
  in_features: number;
  out_features: number;
  bias: Tensor | null;
  weights: Tensor;

  constructor(in_features: number, out_features: number, bias = true) {
    if (in_features <= 0 || out_features <= 0) {
      throw new Error("in and out features must be greater than 0");
    }
    this.in_features = in_features;
    this.out_features = out_features;
    this.weights = initWeights(in_features, out_features);
    this.bias = bias ? initBias(in_features, out_features) : null;
  }

  forward(
    x: Tensor,
    activation: "none" | "relu" | "tanh" | "gelu" = "none"
  ): Tensor {
    if (x.shape.length !== 2) {
      throw new Error("Linear.forward expects a 2D input tensor");
    }
    if (x.shape[1] !== this.in_features) {
      throw new Error(
        `Linear.forward expects in_features=${this.in_features}, got ${x.shape[1]}`
      );
    }

    if (this.bias) {
      if (activation === "gelu") {
        return x.matmul(this.weights).add(this.bias).gelu();
      }
      return x.matmulBiasAct(this.weights, this.bias, activation);
    }
    const out = x.matmul(this.weights);
    if (activation === "relu") return out.relu();
    if (activation === "tanh") return out.tanh();
    if (activation === "gelu") return out.gelu();
    return out;
  }

  parameters(): Tensor[] {
    return this.bias ? [this.weights, this.bias] : [this.weights];
  }
}
