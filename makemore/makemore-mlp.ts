import fs from "node:fs";
import { Tensor } from "../structures";
import { multiply, oneHot, randomNormal } from "../matt-torch";

const names = fs.readFileSync("./names.txt", "utf8");
const words = names.split("\n");
console.log(words.length + ' words');
const SPECIAL = ".";
const uniqChars = [SPECIAL, ...Array.from(new Set(words.join(""))).sort()];
console.log(uniqChars.length + ' unique characters');
const stoi = (s: string) => uniqChars.indexOf(s);
const itos = (i: number) => uniqChars[i];

const BLOCK_SIZE = 3;

let X: number[][] = [];
let Y: number[] = [];

for (const word of words.slice(0,5)) {
  console.log(word);
  let context: number[] = new Array(BLOCK_SIZE).fill(0);
  const chars = word.split('').concat([SPECIAL]);
  for (const char of chars) {
    const idx = stoi(char);
    X.push(context);
    Y.push(idx);
    console.log(`${context.map(item => itos(item)).join('')} ---> ${itos(idx)}`);
    context = context.slice(1).concat([idx]);
  }
}


const xT = Tensor.fromNestedArray([X.length, BLOCK_SIZE], X);
xT.show()
const yT = Tensor.fromNestedArray([Y.length], Y);
yT.show()

const lookupTable = new Tensor([uniqChars.length, 2], () => randomNormal());
lookupTable.show()

const oneHotT = Tensor.fromNestedArray([1, uniqChars.length], oneHot([5], uniqChars.length));
oneHotT.show()

const product = multiply(oneHotT, lookupTable);
product.show()
const r = lookupTable.row([5]);
console.log(r)

const sub = lookupTable.slice(0, { start: 5, end: 8 });
sub.show();