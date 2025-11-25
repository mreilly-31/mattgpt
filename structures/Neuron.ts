import { Value } from "./Value";
import { zip } from "../matt-torch";
export class Neuron {
  weights: Value[];
  bias: Value

  private randValue = (min: number, max: number): Value => new Value(Math.random() * (max - min) + min);

  constructor(numWeights: number) {
    this.weights = new Array(numWeights).fill(0).map(_ => this.randValue(-1, 1));
    this.bias = this.randValue(-1, 1);
  }

  invoke(vals: Value[]): Value {
    const zipped = zip(this.weights, vals);
    // This is a neuron activation!
    const sum = zipped.reduce((acc, cur) => acc.add(cur[0].multiply(cur[1])), this.bias);
    const result = sum.tanh(); // non linearity 
    return result;
  }

  parameters(): Value[] {
    return [...this.weights, this.bias];
  }
}