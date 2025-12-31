/**
 * Dropout is used by models to prevent overfitting during training.
 * Specific neurons can be relied upon too heavily when minimizing loss. this leads to memorization
 * 
 * Dropout randomly disables neurons during training to enfore redundancy.
 * 
 * during training, we randomly set some activations to 0 and rescale outputs to keep expectations constant
 * 
 * during inference, do nothing
 * 
 * let x = input activation, p = dropout probability
 * the mask becomes
 *  m_i = 0 p% of the time, 1 (1-p)% of the time
 * then scale the matrix by 1-p to maintain magnitude
 */

import { Tensor } from "./Tensor";

export class Dropout {
  p: number;
  training: boolean;

  constructor(p = 0.5) {
    if (p < 0 || p >= 1) {
      throw new Error("Dropout probability must be in [0, 1)");
    }
    this.p = p;
    this.training = true;
  }

  train(): void {
    this.training = true;
  }

  eval(): void {
    this.training = false;
  }

  forward(x: Tensor): Tensor {
    if (!this.training || this.p === 0) {
      return x;
    }

    const keepProb = 1 - this.p;
    const scale = 1 / keepProb;
    const maskData = new Array(x.size);
    for (let i = 0; i < x.size; i++) {
      maskData[i] = Math.random() < keepProb ? scale : 0;
    }
    const mask = Tensor.fromArray(x.shape, maskData);
    return x.mul(mask);
  }
}
