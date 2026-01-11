import fs from "node:fs";

type MergePair = [string, string];

type TokenizerState = {
  vocab: Record<string, number>;
  merges: Record<string, number>;
};

const encoder = new TextEncoder();
const decoder = new TextDecoder();

const tokenKey = (bytes: Uint8Array): string => Array.from(bytes).join(",");

const pairKey = (left: string, right: string): string => `${left} ${right}`;

const pairKeyIds = (left: number, right: number): string => `${left},${right}`;

const concatBytes = (a: Uint8Array, b: Uint8Array): Uint8Array => {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
};

type HeapEntry = {
  key: string;
  count: number;
};

class MaxHeap {
  private data: HeapEntry[];

  constructor(entries: HeapEntry[] = []) {
    this.data = entries.slice();
    if (this.data.length > 0) {
      this.heapify();
    }
  }

  private heapify(): void {
    for (let i = Math.floor(this.data.length / 2) - 1; i >= 0; i--) {
      this.siftDown(i);
    }
  }

  private siftUp(index: number): void {
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (this.data[parent].count >= this.data[index].count) {
        break;
      }
      [this.data[parent], this.data[index]] = [this.data[index], this.data[parent]];
      index = parent;
    }
  }

  private siftDown(index: number): void {
    const length = this.data.length;
    while (true) {
      const left = index * 2 + 1;
      const right = left + 1;
      let largest = index;
      if (left < length && this.data[left].count > this.data[largest].count) {
        largest = left;
      }
      if (right < length && this.data[right].count > this.data[largest].count) {
        largest = right;
      }
      if (largest === index) {
        break;
      }
      [this.data[largest], this.data[index]] = [this.data[index], this.data[largest]];
      index = largest;
    }
  }

  push(entry: HeapEntry): void {
    this.data.push(entry);
    this.siftUp(this.data.length - 1);
  }

  pop(): HeapEntry | undefined {
    if (this.data.length === 0) return undefined;
    const top = this.data[0];
    const last = this.data.pop()!;
    if (this.data.length > 0) {
      this.data[0] = last;
      this.siftDown(0);
    }
    return top;
  }
}

class MinHeap {
  private data: HeapEntry[];

  constructor(entries: HeapEntry[] = []) {
    this.data = entries.slice();
    if (this.data.length > 0) {
      this.heapify();
    }
  }

  private heapify(): void {
    for (let i = Math.floor(this.data.length / 2) - 1; i >= 0; i--) {
      this.siftDown(i);
    }
  }

  private siftUp(index: number): void {
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (this.data[parent].count <= this.data[index].count) {
        break;
      }
      [this.data[parent], this.data[index]] = [this.data[index], this.data[parent]];
      index = parent;
    }
  }

  private siftDown(index: number): void {
    const length = this.data.length;
    while (true) {
      const left = index * 2 + 1;
      const right = left + 1;
      let smallest = index;
      if (left < length && this.data[left].count < this.data[smallest].count) {
        smallest = left;
      }
      if (right < length && this.data[right].count < this.data[smallest].count) {
        smallest = right;
      }
      if (smallest === index) {
        break;
      }
      [this.data[smallest], this.data[index]] = [this.data[index], this.data[smallest]];
      index = smallest;
    }
  }

  push(entry: HeapEntry): void {
    this.data.push(entry);
    this.siftUp(this.data.length - 1);
  }

  pop(): HeapEntry | undefined {
    if (this.data.length === 0) return undefined;
    const top = this.data[0];
    const last = this.data.pop()!;
    if (this.data.length > 0) {
      this.data[0] = last;
      this.siftDown(0);
    }
    return top;
  }
}

export class BPETokenizer {
  private vocab = new Map<string, number>();
  private idToToken = new Map<number, Uint8Array>();
  private merges = new Map<string, number>();
  private mergeRanksById = new Map<string, number>();
  private mergePairToId = new Map<string, number>();
  private byteToId: number[] | null = null;

  constructor(state?: TokenizerState) {
    if (state) {
      Object.entries(state.vocab).forEach(([key, id]) => {
        this.vocab.set(key, id);
        const bytes = key.length === 0 ? new Uint8Array() : Uint8Array.from(key.split(",").map(Number));
        this.idToToken.set(id, bytes);
      });
      Object.entries(state.merges).forEach(([key, rank]) => {
        this.merges.set(key, rank);
      });
      this.rebuildCaches();
    }
  }

  static fromState(state: TokenizerState): BPETokenizer {
    return new BPETokenizer(state);
  }

  private rebuildCaches(): void {
    this.mergeRanksById.clear();
    this.mergePairToId.clear();
    this.byteToId = null;

    for (const [key, rank] of this.merges.entries()) {
      const [leftKey, rightKey] = key.split(" ");
      const leftId = this.vocab.get(leftKey);
      const rightId = this.vocab.get(rightKey);
      if (leftId === undefined || rightId === undefined) {
        continue;
      }
      const leftBytes = this.idToToken.get(leftId);
      const rightBytes = this.idToToken.get(rightId);
      if (!leftBytes || !rightBytes) {
        continue;
      }
      const mergedKey = tokenKey(concatBytes(leftBytes, rightBytes));
      const mergedId = this.vocab.get(mergedKey);
      if (mergedId === undefined) {
        continue;
      }
      const idKey = pairKeyIds(leftId, rightId);
      this.mergeRanksById.set(idKey, rank);
      this.mergePairToId.set(idKey, mergedId);
    }
  }

  private getByteToId(): number[] {
    if (this.byteToId) return this.byteToId;
    const table = new Array(256).fill(-1);
    for (let i = 0; i < 256; i++) {
      const id = this.vocab.get(String(i));
      if (id !== undefined) {
        table[i] = id;
      }
    }
    this.byteToId = table;
    return table;
  }

  trainFromFile(filepath: string, vocabSize: number): void {
    const contents = fs.readFileSync(filepath, 'utf8');
    this.train(contents, vocabSize);
  }

  train(text: string, vocabSize: number): void {
    if (vocabSize < 256) {
      throw new Error("vocabSize must be at least 256 for byte-level BPE");
    }

    this.vocab.clear();
    this.idToToken.clear();
    this.merges.clear();
    this.mergeRanksById.clear();
    this.mergePairToId.clear();
    this.byteToId = null;

    const bytes = encoder.encode(text);
    const length = bytes.length;
    const tokens: number[] = new Array(length);
    for (let i = 0; i < length; i++) {
      const key = String(bytes[i]);
      if (!this.vocab.has(key)) {
        const id = this.vocab.size;
        this.vocab.set(key, id);
        this.idToToken.set(id, Uint8Array.from([bytes[i]]));
      }
      tokens[i] = this.vocab.get(key)!;
    }

    const prev: number[] = new Array(length);
    const next: number[] = new Array(length);
    const alive: boolean[] = new Array(length).fill(true);
    for (let i = 0; i < length; i++) {
      prev[i] = i - 1;
      next[i] = i + 1 < length ? i + 1 : -1;
    }

    const pairPositions = new Map<string, Set<number>>();
    const pairCounts = new Map<string, number>();

    for (let i = 0; i < length - 1; i++) {
      const key = pairKeyIds(tokens[i], tokens[i + 1]);
      let positions = pairPositions.get(key);
      if (!positions) {
        positions = new Set();
        pairPositions.set(key, positions);
      }
      positions.add(i);
      pairCounts.set(key, (pairCounts.get(key) ?? 0) + 1);
    }

    let heap = new MaxHeap(
      Array.from(pairCounts.entries()).map(([key, count]) => ({ key, count }))
    );

    const addPair = (pos: number, leftId: number, rightId: number): void => {
      const key = pairKeyIds(leftId, rightId);
      let positions = pairPositions.get(key);
      if (!positions) {
        positions = new Set();
        pairPositions.set(key, positions);
      }
      if (!positions.has(pos)) {
        positions.add(pos);
        const count = (pairCounts.get(key) ?? 0) + 1;
        pairCounts.set(key, count);
        heap.push({ key, count });
      }
    };

    const removePair = (pos: number, leftId: number, rightId: number): void => {
      const key = pairKeyIds(leftId, rightId);
      const positions = pairPositions.get(key);
      if (!positions || !positions.has(pos)) {
        return;
      }
      positions.delete(pos);
      const count = (pairCounts.get(key) ?? 1) - 1;
      if (count <= 0) {
        pairCounts.delete(key);
        pairPositions.delete(key);
      } else {
        pairCounts.set(key, count);
        heap.push({ key, count });
      }
    };

    const logEvery = Math.max(25, Math.floor((vocabSize - 256) / 20));
    while (this.vocab.size < vocabSize) {
      let best: HeapEntry | undefined;
      while (true) {
        const candidate = heap.pop();
        if (!candidate) break;
        const current = pairCounts.get(candidate.key);
        if (current === candidate.count && current >= 2) {
          best = candidate;
          break;
        }
      }
      if (!best) break;

      const [leftIdStr, rightIdStr] = best.key.split(",");
      const leftId = Number(leftIdStr);
      const rightId = Number(rightIdStr);
      const positions = pairPositions.get(best.key);
      if (!positions) continue;

      const validPositions: number[] = [];
      const sortedPositions = Array.from(positions).sort((a, b) => a - b);
      for (const pos of sortedPositions) {
        const right = next[pos];
        if (!alive[pos] || right === -1 || !alive[right]) {
          removePair(pos, leftId, rightId);
          continue;
        }
        if (tokens[pos] !== leftId || tokens[right] !== rightId) {
          removePair(pos, leftId, rightId);
          continue;
        }
        validPositions.push(pos);
      }

      if (validPositions.length === 0) {
        continue;
      }

      const leftBytes = this.idToToken.get(leftId);
      const rightBytes = this.idToToken.get(rightId);
      if (!leftBytes || !rightBytes) {
        break;
      }

      const mergedBytes = concatBytes(leftBytes, rightBytes);
      const mergedKey = tokenKey(mergedBytes);
      if (this.vocab.has(mergedKey)) {
        break;
      }

      const newId = this.vocab.size;
      this.vocab.set(mergedKey, newId);
      this.idToToken.set(newId, mergedBytes);
      const leftKey = tokenKey(leftBytes);
      const rightKey = tokenKey(rightBytes);
      const rank = this.merges.size;
      this.merges.set(pairKey(leftKey, rightKey), rank);
      this.mergeRanksById.set(pairKeyIds(leftId, rightId), rank);
      this.mergePairToId.set(pairKeyIds(leftId, rightId), newId);

      let mergedAny = false;
      for (const pos of validPositions) {
        const right = next[pos];
        if (right === -1 || !alive[pos] || !alive[right]) {
          continue;
        }
        if (tokens[pos] !== leftId || tokens[right] !== rightId) {
          continue;
        }

        const prevIdx = prev[pos];
        const nextIdx = next[right];

        if (prevIdx !== -1) {
          removePair(prevIdx, tokens[prevIdx], tokens[pos]);
        }
        removePair(pos, leftId, rightId);
        if (nextIdx !== -1) {
          removePair(right, tokens[right], tokens[nextIdx]);
        }

        tokens[pos] = newId;
        alive[right] = false;
        next[pos] = nextIdx;
        if (nextIdx !== -1) {
          prev[nextIdx] = pos;
        }

        if (prevIdx !== -1) {
          addPair(prevIdx, tokens[prevIdx], tokens[pos]);
        }
        if (nextIdx !== -1) {
          addPair(pos, tokens[pos], tokens[nextIdx]);
        }
        mergedAny = true;
      }

      if (!mergedAny) {
        break;
      }

      if (this.vocab.size >= 256) {
        const mergesDone = Math.max(0, this.vocab.size - 256);
        if (mergesDone % logEvery === 0) {
          console.log(
            `[tokenizer] merges=${mergesDone} vocab=${this.vocab.size} topPairCount=${best.count}`
          );
        }
      }
    }
  }

  encode(text: string): number[] {
    if (this.mergeRanksById.size === 0 && this.merges.size > 0) {
      this.rebuildCaches();
    }
    const bytes = encoder.encode(text);
    const byteToId = this.getByteToId();
    const tokens: number[] = new Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) {
      const id = byteToId[bytes[i]];
      if (id === -1) {
        throw new Error(`Unknown byte token: ${bytes[i]}`);
      }
      tokens[i] = id;
    }

    if (tokens.length < 2 || this.mergeRanksById.size === 0) {
      return tokens;
    }

    const length = tokens.length;
    const prev: number[] = new Array(length);
    const next: number[] = new Array(length);
    const alive: boolean[] = new Array(length).fill(true);
    for (let i = 0; i < length; i++) {
      prev[i] = i - 1;
      next[i] = i + 1 < length ? i + 1 : -1;
    }

    const pairPositions = new Map<string, Set<number>>();
    const heap = new MinHeap();
    const ranks = this.mergeRanksById;

    const addPair = (pos: number, leftId: number, rightId: number): void => {
      const key = pairKeyIds(leftId, rightId);
      const rank = ranks.get(key);
      if (rank === undefined) return;
      let positions = pairPositions.get(key);
      if (!positions) {
        positions = new Set();
        pairPositions.set(key, positions);
      }
      if (!positions.has(pos)) {
        positions.add(pos);
        heap.push({ key, count: rank });
      }
    };

    const removePair = (pos: number, leftId: number, rightId: number): void => {
      const key = pairKeyIds(leftId, rightId);
      const positions = pairPositions.get(key);
      if (!positions || !positions.has(pos)) {
        return;
      }
      positions.delete(pos);
      if (positions.size === 0) {
        pairPositions.delete(key);
      }
    };

    for (let i = 0; i < length - 1; i++) {
      addPair(i, tokens[i], tokens[i + 1]);
    }

    let iter = 0;
    const logEvery = Math.max(1000, Math.floor(tokens.length / 20));
    while (true) {
      let best: HeapEntry | undefined;
      while (true) {
        const candidate = heap.pop();
        if (!candidate) break;
        const currentRank = ranks.get(candidate.key);
        if (currentRank !== candidate.count) {
          continue;
        }
        const positions = pairPositions.get(candidate.key);
        if (positions && positions.size > 0) {
          best = candidate;
          break;
        }
      }
      if (!best) break;

      const [leftIdStr, rightIdStr] = best.key.split(",");
      const leftId = Number(leftIdStr);
      const rightId = Number(rightIdStr);
      const positions = pairPositions.get(best.key);
      if (!positions) continue;

      const validPositions: number[] = [];
      const sortedPositions = Array.from(positions).sort((a, b) => a - b);
      for (const pos of sortedPositions) {
        const right = next[pos];
        if (!alive[pos] || right === -1 || !alive[right]) {
          removePair(pos, leftId, rightId);
          continue;
        }
        if (tokens[pos] !== leftId || tokens[right] !== rightId) {
          removePair(pos, leftId, rightId);
          continue;
        }
        validPositions.push(pos);
      }

      if (validPositions.length === 0) {
        continue;
      }

      const mergedId = this.mergePairToId.get(best.key);
      if (mergedId === undefined) {
        removePair(validPositions[0], leftId, rightId);
        continue;
      }

      for (const pos of validPositions) {
        const right = next[pos];
        if (right === -1 || !alive[pos] || !alive[right]) {
          continue;
        }
        if (tokens[pos] !== leftId || tokens[right] !== rightId) {
          continue;
        }

        const prevIdx = prev[pos];
        const nextIdx = next[right];

        if (prevIdx !== -1) {
          removePair(prevIdx, tokens[prevIdx], tokens[pos]);
        }
        removePair(pos, leftId, rightId);
        if (nextIdx !== -1) {
          removePair(right, tokens[right], tokens[nextIdx]);
        }

        tokens[pos] = mergedId;
        alive[right] = false;
        next[pos] = nextIdx;
        if (nextIdx !== -1) {
          prev[nextIdx] = pos;
        }

        if (prevIdx !== -1) {
          addPair(prevIdx, tokens[prevIdx], tokens[pos]);
        }
        if (nextIdx !== -1) {
          addPair(pos, tokens[pos], tokens[nextIdx]);
        }
      }

      iter += 1;
      if (iter % logEvery === 0) {
        let aliveCount = 0;
        for (let i = 0; i < alive.length; i++) {
          if (alive[i]) aliveCount++;
        }
        console.log(`[tokenizer] encode merges=${iter} remaining=${aliveCount}`);
      }
    }

    const output: number[] = [];
    let cursor = 0;
    while (cursor !== -1 && !alive[cursor]) {
      cursor = next[cursor];
    }
    while (cursor !== -1) {
      output.push(tokens[cursor]);
      cursor = next[cursor];
    }
    return output;
  }

  decode(ids: number[]): string {
    const bytes: number[] = [];
    for (const id of ids) {
      const token = this.idToToken.get(id);
      if (!token) {
        throw new Error(`Unknown token id: ${id}`);
      }
      for (const byte of token) {
        bytes.push(byte);
      }
    }
    return decoder.decode(new Uint8Array(bytes));
  }

  getState(): TokenizerState {
    const vocab: Record<string, number> = {};
    for (const [key, id] of this.vocab.entries()) {
      vocab[key] = id;
    }
    const merges: Record<string, number> = {};
    for (const [key, rank] of this.merges.entries()) {
      merges[key] = rank;
    }
    return { vocab, merges };
  }
}
