import { LayerNorm } from "./LayerNorm";
import { FeedForward } from "./FeedForward";
import { ModelComponent } from "./ModelComponent";
import { MultiHeadAttention } from "./MultiHeadAttention";
import { Tape, Tensor } from "./Tensor";

export type TransformerBlockConfig = {
  embedDim: number;
  numHeads: number;
  contextLength: number;
  dropout: number;
  tape?: Tape;
};

export class TransformerBlock extends ModelComponent {
  private ln1: LayerNorm;
  private ln2: LayerNorm;
  private attn: MultiHeadAttention;
  private mlp: FeedForward;

  constructor(config: TransformerBlockConfig) {
    super();
    this.ln1 = new LayerNorm(config.embedDim, 1e-5, config.tape);
    this.ln2 = new LayerNorm(config.embedDim, 1e-5, config.tape);
    this.attn = new MultiHeadAttention({
      dims_in: config.embedDim,
      dims_out: config.embedDim,
      context_length: config.contextLength,
      dropout: config.dropout,
      num_heads: config.numHeads,
      training: true,
      tape: config.tape
    });
    this.mlp = new FeedForward(
      config.embedDim,
      config.embedDim * 4,
      config.embedDim,
      config.dropout,
      "gelu",
      config.tape
    );
  }

  forward(x: Tensor): Tensor {
    const attnOut = this.attn.forward(this.ln1.forward(x));
    const resid1 = x.add(attnOut);
    const mlpOut = this.mlp.forward(this.ln2.forward(resid1));
    return resid1.add(mlpOut);
  }

  parameters(): Tensor[] {
    return [
      ...this.ln1.parameters(),
      ...this.ln2.parameters(),
      ...this.attn.parameters(),
      ...this.mlp.parameters()
    ];
  }

  train(): this {
    super.train();
    this.ln1.train();
    this.ln2.train();
    this.attn.train();
    this.mlp.train();
    return this;
  }

  eval(): this {
    super.eval();
    this.ln1.eval();
    this.ln2.eval();
    this.attn.eval();
    this.mlp.eval();
    return this;
  }
}
