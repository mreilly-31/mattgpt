import { ninf, ones, triu } from "../matt-torch";
import { Dropout } from "./Dropout";
import { Linear } from "./Linear";
import { ModelComponent } from "./ModelComponent";
import { Tape, Tensor } from "./Tensor";

interface MultiHeadAttentionConfig {
  dims_in: number;
  dims_out: number;
  context_length: number;
  dropout: number;
  num_heads: number;
  training: boolean;
  tape?: Tape;
}

export class MultiHeadAttention extends ModelComponent{
  dims_in: number;
  dims_out: number;
  context_length: number;
  dropout: Dropout;
  num_heads: number;
  head_dim: number;
  W_query: Linear;
  W_key: Linear;
  W_value: Linear;
  out_proj: Linear;
  mask: Tensor;
  training: boolean;

  constructor(params: MultiHeadAttentionConfig) {
    super();
    this.dims_in = params.dims_in;
    this.dims_out = params.dims_out;
    this.context_length = params.context_length;
    this.training = params.training;
    this.dropout = new Dropout(params.dropout);
    if (this.training) {
      this.dropout.train();
    }
    this.num_heads = params.num_heads;
    if (this.dims_out % this.num_heads !== 0) {
      throw new Error("dims_out must be divisible by num_heads");
    }
    this.head_dim = Math.trunc(params.dims_out / params.num_heads);
    const tape = params.tape;
    this.W_query = new Linear(this.dims_in, this.dims_out, true, tape);
    this.W_key = new Linear(this.dims_in, this.dims_out, true, tape);
    this.W_value = new Linear(this.dims_in, this.dims_out, true, tape);
    this.out_proj = new Linear(this.dims_out, this.dims_out, true, tape);
    this.mask = triu(ones([this.context_length, this.context_length], false), 1, false);
  }

  train(): this {
    this.training = true;
    this.dropout.train();
    this.W_query.train();
    this.W_key.train();
    this.W_value.train();
    this.out_proj.train();
    return this;
  }

  eval(): this {
    this.training = false;
    this.dropout.eval();
    this.W_query.eval();
    this.W_key.eval();
    this.W_value.eval();
    this.out_proj.eval();
    return this;
  }

  forward(x: Tensor) {
    const [b, num_tokens, dims_in] = x.shape;

    const flatX = x.view([b * num_tokens, dims_in]);
    let keys = this.W_key.forward(flatX).view([
      b,
      num_tokens,
      this.dims_out
    ]);
    let queries = this.W_query.forward(flatX).view([
      b,
      num_tokens,
      this.dims_out
    ]);
    let values = this.W_value.forward(flatX).view([
      b,
      num_tokens,
      this.dims_out
    ]);

    keys = keys.view([b, num_tokens, this.num_heads, this.head_dim]);
    queries = queries.view([b, num_tokens, this.num_heads, this.head_dim]);
    values = values.view([b, num_tokens, this.num_heads, this.head_dim]);

    keys = keys.transpose(1, 2);
    queries = queries.transpose(1, 2);
    values = values.transpose(1, 2);

    let attn_scores = queries.matmul(keys.transpose(2, 3));
    let mask_fill = this.mask.slice(0, 0, num_tokens).slice(1, 0, num_tokens);
    attn_scores = attn_scores.add(mask_fill.mulScalar(ninf));

    let attn_weights = attn_scores
      .divScalar(this.head_dim ** 0.5)
      .softmax(-1);
    attn_weights = this.dropout.forward(attn_weights);

    let context_vector = attn_weights.matmul(values).transpose(1, 2);
    context_vector = context_vector.view([b, num_tokens, this.dims_out]);
    context_vector = this.out_proj.forward(context_vector);

    return context_vector;
  }

  parameters(): Tensor[] {
    return [
      ...this.W_query.parameters(),
      ...this.W_key.parameters(),
      ...this.W_value.parameters(),
      ...this.out_proj.parameters()
    ];
  }
}
