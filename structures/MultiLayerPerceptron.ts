// Multi Layer Perceptron hell yea
import { Value } from "./Value";
import { Layer } from "./Layer";


export class MultiLayerPerceptron {
  layers: Layer[];

  constructor(numberOfInputs: number, layerSizes: number[]) {
    const allInputs = [numberOfInputs, ...layerSizes];
    let layers = [];
    for (let i = 0; i < layerSizes.length; i++) {
      layers.push(new Layer(allInputs[i], allInputs[i + 1]));
    }
    this.layers = layers;
  }

  invoke(x: Value[]): Value[] {
    let result = [...x];
    for (const layer of this.layers) {
      result = layer.invoke(x);
    }
    return result;
  }

  parameters() {
    let result: Value[] = [];
    for (const layer of this.layers) {
      result.push(...layer.parameters());
    }

    return result;
  }
}