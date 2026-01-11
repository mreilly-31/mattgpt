import { Tensor } from "./Tensor";

export abstract class ModelComponent {
  protected training = true;

  abstract forward(input: Tensor, ...args: unknown[]): Tensor;

  parameters(): Tensor[] {
    return [];
  }

  train(): this {
    this.training = true;
    return this;
  }

  eval(): this {
    this.training = false;
    return this;
  }
}
