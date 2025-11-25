export class Value {
  data: number;
  grad: number;
  _backward: () => void;
  _prev: Set<Value>;
  op: string;
  label: string;

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

  print() {
    return `${this.label}: Data: ${this.data}, Grad: ${this.grad}`;
  }

  add(other: Value) {
    const result = new Value(this.data + other.data, [this, other], "+");

    result._backward = () => {
      this.grad += 1.0 * result.grad;
      other.grad += 1.0 * result.grad;
    };

    return result;
  }

  multiply(other: Value) {
    const result = new Value(this.data * other.data, [this, other], "*");

    result._backward = () => {
      this.grad += other.data * result.grad;
      other.grad += this.data * result.grad;
    };

    return result;
  }

  pow(raiseTo: number) {
    const result = new Value(this.data ** raiseTo, [this], '^');

    result._backward = () => {
      this.grad += raiseTo * (this.data**(raiseTo - 1)) * result.grad;
    }

    return result;
  }

  divide(other: Value) {
    return this.multiply(other.pow(-1));
  }

  negative() {
    return this.multiply(new Value(-1));
  }

  subtract(other: Value) {
    return this.add(other.negative());
  }

  tanh() {
    const x = this.data;
    // https://mathworld.wolfram.com/HyperbolicTangent.html
    const t = (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1);
    const result = new Value(t, [this], 'tanh()');

    result._backward = () => {
      this.grad += (1 - t**2) * result.grad;
    }

    return result;
  }

  exp() {
    const x = this.data;
    const result = new Value(Math.exp(x), [this], 'e^x');

    this._backward = () => {
      result.grad += result.data * result.grad;
    }

    return result;
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