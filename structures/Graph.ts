import {
  attribute as _,
  Digraph,
  Node,
  Edge,
  toDot,
} from "ts-graphviz";
import { Value } from "./Value";

export class Graph {
  graph: Digraph;
  constructor() {
    this.graph = new Digraph();
  }

  trace(root: Value) {
    const nodes = new Set<Value>();
    const edges = new Set<Value[]>();
    const build = (v: Value) => {
      if (!nodes.has(v)) {
        nodes.add(v);
        for (const child of v._prev) {
          edges.add([child, v]);
          build(child);
        }
      }
    };

    build(root);
    return { nodes, edges };
  }

  draw(root: Value) {
    const { nodes, edges } = this.trace(root);
    const nodeMap = new Map<string, Node>();
    for (const node of nodes) {
      const graphNode = new Node(node.print(), { shape: 'record' });
      this.graph.addNode(graphNode);
      nodeMap.set(node.print(), graphNode);
      if (node.op) {
        const opId = `${node.print()} ${node.op}`;
        const opNode = new Node(opId, { label: node.op });
        this.graph.addNode(opNode);
        nodeMap.set(opId, opNode);
        const opEdge = new Edge([opNode, graphNode]);
        this.graph.addEdge(opEdge);
      }
    }

    for (const edge of edges) {
      const edge_1 = nodeMap.get(edge[0].print());
      const edge_2 = nodeMap.get(`${edge[1].print()} ${edge[1].op}`);

      if (!edge_1 || !edge_2) {
        continue;
      }

      const grapEdge = new Edge([edge_1, edge_2]);
      this.graph.addEdge(grapEdge);
    }
    const dot = toDot(this.graph);
    // console.log(dot);
    // paste into https://viz-js.com/
    return dot;
  }
}
