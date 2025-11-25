import { Value } from "./Value";
import { Neuron } from "./Neuron";

export class Layer {
  neurons: Neuron[];

  constructor(neuronDimensionality: number, numberOfNeurons: number) {
    this.neurons = new Array(numberOfNeurons).fill(0).map(_ => new Neuron(neuronDimensionality));
  }

  invoke(vals: Value[]): Value[] {
    const result = this.neurons.map(neuron => neuron.invoke(vals));
    return result;
  }

  parameters(): Value[] {
    let result: Value[] = [];
    for (const neuron of this.neurons) {
      for (const param of neuron.parameters()) {
        result.push(param);
      }
    }

    return result;
  }
}