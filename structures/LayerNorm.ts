import { ModelComponent } from "./ModelComponent";
import { Tape, Tensor } from "./Tensor";

export class LayerNorm extends ModelComponent {
  private features: number;
  private eps: number;
  private gamma: Tensor;
  private beta: Tensor;

  constructor(features: number, eps = 1e-5, tape?: Tape) {
    super();
    if (features <= 0) {
      throw new Error("LayerNorm features must be greater than 0");
    }
    this.features = features;
    this.eps = eps;
    this.gamma = Tensor.fromArray(
      [features],
      new Array(features).fill(1),
      true,
      tape
    );
    this.beta = Tensor.fromArray(
      [features],
      new Array(features).fill(0),
      true,
      tape
    );
  }

  forward(input: Tensor): Tensor {
    if (input.shape.length === 0) {
      throw new Error("LayerNorm.forward expects a non-scalar tensor");
    }
    const lastDim = input.shape[input.shape.length - 1];
    if (lastDim !== this.features) {
      throw new Error(
        `LayerNorm.forward expects last dimension ${this.features}, got ${lastDim}`
      );
    }
    return input.layerNorm(this.gamma, this.beta, this.eps);
  }

  parameters(): Tensor[] {
    return [this.gamma, this.beta];
  }
}
