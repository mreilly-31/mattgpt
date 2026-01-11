import { Dropout } from "./Dropout";
import { Linear } from "./Linear";
import { ModelComponent } from "./ModelComponent";
import { Tape, Tensor } from "./Tensor";

export class FeedForward extends ModelComponent {
  private fc1: Linear;
  private fc2: Linear;
  private dropout?: Dropout;
  private activation: "relu" | "tanh" | "gelu";

  constructor(
    inFeatures: number,
    hiddenFeatures = inFeatures * 4,
    outFeatures = inFeatures,
    dropout = 0,
    activation: "relu" | "tanh" | "gelu" = "gelu",
    tape?: Tape
  ) {
    super();
    this.fc1 = new Linear(inFeatures, hiddenFeatures, true, tape);
    this.fc2 = new Linear(hiddenFeatures, outFeatures, true, tape);
    this.activation = activation;
    if (dropout > 0) {
      this.dropout = new Dropout(dropout);
    }
  }

  forward(input: Tensor): Tensor {
    let out = this.fc1.forward(input, this.activation);
    if (this.dropout) {
      out = this.dropout.forward(out);
    }
    return this.fc2.forward(out);
  }

  parameters(): Tensor[] {
    const params = [...this.fc1.parameters(), ...this.fc2.parameters()];
    if (this.dropout) {
      params.push(...this.dropout.parameters());
    }
    return params;
  }

  train(): this {
    super.train();
    this.fc1.train();
    this.fc2.train();
    if (this.dropout) this.dropout.train();
    return this;
  }

  eval(): this {
    super.eval();
    this.fc1.eval();
    this.fc2.eval();
    if (this.dropout) this.dropout.eval();
    return this;
  }
}
