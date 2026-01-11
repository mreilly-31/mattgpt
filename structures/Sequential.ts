import { ModelComponent } from "./ModelComponent";
import { Tensor } from "./Tensor";

export class Sequential extends ModelComponent {
  private layers: ModelComponent[];

  constructor(layers: ModelComponent[]) {
    super();
    this.layers = layers;
  }

  forward(input: Tensor): Tensor {
    return this.layers.reduce((acc, layer) => layer.forward(acc), input);
  }

  parameters(): Tensor[] {
    return this.layers.flatMap((layer) => layer.parameters());
  }

  train(): this {
    super.train();
    this.layers.forEach((layer) => layer.train());
    return this;
  }

  eval(): this {
    super.eval();
    this.layers.forEach((layer) => layer.eval());
    return this;
  }
}
