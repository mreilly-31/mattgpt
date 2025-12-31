export class Value {
  data: number;
  grad: number;
  _backward: () => void;
  _prev: Set<Value>;
  op: string;
  label: string;
  static gradEnabled = true;

  constructor(
    data: number,
    _children: Value[] = [],
    op: string = "",
    label: string = ""
  ) {
    this.data = data;
    this._prev = new Set(_children);
    this.grad = 0.0;
    this.op = op;
    this.label = label;
    this._backward = () => {};
  }

  static v(val: number): Value {
    return new Value(val);
  }

  static dot(row: Value[], col: Value[]): Value {
    if (row.length !== col.length) {
      throw new Error("Dot product requires matching lengths");
    }
    let sum = 0;
    for (let i = 0; i < row.length; i++) {
      sum += row[i].data * col[i].data;
    }
    if (!Value.gradEnabled) {
      return new Value(sum);
    }
    const result = new Value(sum, [...row, ...col], "dot");
    result._backward = () => {
      for (let i = 0; i < row.length; i++) {
        row[i].grad += col[i].data * result.grad;
        col[i].grad += row[i].data * result.grad;
      }
    };
    return result;
  }

  static withNoGrad<T>(fn: () => T): T {
    const previous = Value.gradEnabled;
    Value.gradEnabled = false;
    try {
      return fn();
    } finally {
      Value.gradEnabled = previous;
    }
  }

  private static make(
    data: number,
    children: Value[],
    op: string,
    backward: (result: Value) => void
  ): Value {
    if (!Value.gradEnabled) {
      return new Value(data);
    }
    const result = new Value(data, children, op);
    result._backward = () => backward(result);
    return result;
  }

  print() {
    return `${this.label}: Data: ${this.data}, Grad: ${this.grad}`;
  }

  add(other: Value) {
    return Value.make(this.data + other.data, [this, other], "+", (result) => {
      this.grad += 1.0 * result.grad;
      other.grad += 1.0 * result.grad;
    });
  }

  addScalar(other: number) {
    return Value.make(this.data + other, [this], "+c", (result) => {
      this.grad += 1.0 * result.grad;
    });
  }

  multiply(other: Value) {
    return Value.make(this.data * other.data, [this, other], "*", (result) => {
      this.grad += other.data * result.grad;
      other.grad += this.data * result.grad;
    });
  }

  mulScalar(other: number) {
    return Value.make(this.data * other, [this], "*c", (result) => {
      this.grad += other * result.grad;
    });
  }

  pow(raiseTo: number) {
    return Value.make(this.data ** raiseTo, [this], "^", (result) => {
      this.grad += raiseTo * (this.data ** (raiseTo - 1)) * result.grad;
    });
  }

  subScalar(other: number) {
    return this.addScalar(-other);
  }

  divScalar(other: number) {
    return this.mulScalar(1 / other);
  }

  divide(other: Value) {
    return this.multiply(other.pow(-1));
  }

  negative() {
    return this.mulScalar(-1);
  }

  subtract(other: Value) {
    return this.add(other.negative());
  }

  log() {
    return Value.make(Math.log(this.data), [this], "log", (result) => {
      this.grad += (1 / this.data) * result.grad;
    });
  }

  tanh() {
    const x = this.data;
    // https://mathworld.wolfram.com/HyperbolicTangent.html
    const t = (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1);
    return Value.make(t, [this], "tanh()", (result) => {
      this.grad += (1 - t ** 2) * result.grad;
    });
  }

  exp() {
    return Value.make(Math.exp(this.data), [this], "e^x", (result) => {
      this.grad += result.data * result.grad;
    });
  }

  relu() {
    const out = Math.max(0, this.data);
    return Value.make(out, [this], "ReLU", (result) => {
      this.grad += (this.data > 0 ? 1 : 0) * result.grad;
    });
  }

  backward() {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const build_topo = (v: Value) => {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._prev) {
          build_topo(child);
        }
        topo.push(v);
      }
    };
    build_topo(this);

    this.grad = 1.0;
    for (const node of topo.reverse()) {
      node._backward();
    }
  }
}

export function v(val: number): Value {
  return Value.v(val);
}
