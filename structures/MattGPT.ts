import { LayerNorm } from "./LayerNorm";
import { ModelComponent } from "./ModelComponent";
import { Tape, Tensor } from "./Tensor";
import { TransformerBlock, TransformerBlockConfig } from "./TransformerBlock";

export type MattGPTConfig = {
  vocabSize: number;
  contextLength: number;
  embedDim: number;
  numHeads: number;
  numLayers: number;
  dropout: number;
  tape?: Tape;
};

type TensorState = {
  shape: number[];
  data: number[];
};

export type MattGPTState = {
  config: Omit<MattGPTConfig, "tape">;
  tensors: TensorState[];
};

export class MattGPT extends ModelComponent {
  private vocabSize: number;
  private contextLength: number;
  private embedDim: number;
  private numHeads: number;
  private numLayers: number;
  private dropout: number;
  private tokenEmbedding: Tensor;
  private positionEmbedding: Tensor;
  private blocks: TransformerBlock[];
  private lnFinal: LayerNorm;
  private head: Tensor;

  constructor(config: MattGPTConfig) {
    super();
    this.vocabSize = config.vocabSize;
    this.contextLength = config.contextLength;
    this.embedDim = config.embedDim;
    this.numHeads = config.numHeads;
    this.numLayers = config.numLayers;
    this.dropout = config.dropout;

    const tokenData = new Array(this.vocabSize * this.embedDim)
      .fill(0)
      .map(() => Math.random() * 0.02 - 0.01);
    const posData = new Array(this.contextLength * this.embedDim)
      .fill(0)
      .map(() => Math.random() * 0.02 - 0.01);
    const headData = new Array(this.embedDim * this.vocabSize)
      .fill(0)
      .map(() => Math.random() * 0.02 - 0.01);

    const tape = config.tape;
    this.tokenEmbedding = Tensor.fromArray(
      [this.vocabSize, this.embedDim],
      tokenData,
      true,
      tape
    );
    this.positionEmbedding = Tensor.fromArray(
      [this.contextLength, this.embedDim],
      posData,
      true,
      tape
    );

    const blockConfig: TransformerBlockConfig = {
      embedDim: this.embedDim,
      numHeads: this.numHeads,
      contextLength: this.contextLength,
      dropout: this.dropout,
      tape
    };
    this.blocks = new Array(this.numLayers)
      .fill(0)
      .map(() => new TransformerBlock(blockConfig));
    this.lnFinal = new LayerNorm(this.embedDim, 1e-5, tape);
    this.head = Tensor.fromArray(
      [this.embedDim, this.vocabSize],
      headData,
      true,
      tape
    );
  }

  forward(idx: Tensor): Tensor {
    if (idx.shape.length !== 2) {
      throw new Error("MattGPT.forward expects [batch, time] token indices");
    }
    const [batch, time] = idx.shape;
    if (time > this.contextLength) {
      throw new Error("Sequence length exceeds context length");
    }

    const tokenEmb = this.tokenEmbedding.gather(0, idx);
    const posData = new Array(time);
    for (let i = 0; i < time; i++) {
      posData[i] = i;
    }
    const posIndices = Tensor.fromArray([time], posData);
    const posEmb = this.positionEmbedding
      .gather(0, posIndices)
      .view([1, time, this.embedDim]);

    let x = tokenEmb.add(posEmb);
    for (const block of this.blocks) {
      x = block.forward(x);
    }
    x = this.lnFinal.forward(x);

    const flat = x.view([batch * time, this.embedDim]);
    const logits = flat.matmul(this.head).view([batch, time, this.vocabSize]);
    return logits;
  }

  getState(): MattGPTState {
    const config: Omit<MattGPTConfig, "tape"> = {
      vocabSize: this.vocabSize,
      contextLength: this.contextLength,
      embedDim: this.embedDim,
      numHeads: this.numHeads,
      numLayers: this.numLayers,
      dropout: this.dropout
    };
    const tensors = this.parameters().map((tensor) => ({
      shape: [...tensor.shape],
      data: Array.from(tensor.data)
    }));
    return { config, tensors };
  }

  loadState(state: MattGPTState): void {
    const params = this.parameters();
    if (params.length !== state.tensors.length) {
      throw new Error(
        `State tensor count ${state.tensors.length} does not match model parameter count ${params.length}`
      );
    }
    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const saved = state.tensors[i];
      const shapeMatch =
        param.shape.length === saved.shape.length &&
        param.shape.every((dim, idx) => dim === saved.shape[idx]);
      if (!shapeMatch) {
        throw new Error(
          `State tensor ${i} has shape ${saved.shape}, expected ${param.shape}`
        );
      }
      if (saved.data.length !== param.size) {
        throw new Error(
          `State tensor ${i} has data length ${saved.data.length}, expected ${param.size}`
        );
      }
      param.data.set(saved.data);
    }
  }

  static fromState(state: MattGPTState, tape?: Tape): MattGPT {
    const model = new MattGPT({ ...state.config, tape });
    model.loadState(state);
    return model;
  }

  parameters(): Tensor[] {
    return [
      this.tokenEmbedding,
      this.positionEmbedding,
      this.head,
      ...this.blocks.flatMap((block) => block.parameters()),
      ...this.lnFinal.parameters()
    ];
  }

  train(): this {
    super.train();
    this.blocks.forEach((block) => block.train());
    this.lnFinal.train();
    return this;
  }

  eval(): this {
    super.eval();
    this.blocks.forEach((block) => block.eval());
    this.lnFinal.eval();
    return this;
  }
}
