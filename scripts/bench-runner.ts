import { spawnSync } from "node:child_process";

type Result = {
  forward: number;
  forwardNoGrad: number;
  backward: number;
  softmax: number;
};

const RUNS = 10;
const DISCARD = 3;

const parseResult = (output: string, label: string): Result => {
  const match = output.match(
    /RESULT\s+forward_ms=([0-9.]+)\s+forward_nograd_ms=([0-9.]+)\s+backward_ms=([0-9.]+)\s+softmax_ms=([0-9.]+)/
  );
  if (!match) {
    throw new Error(`${label} output missing RESULT line`);
  }
  return {
    forward: Number(match[1]),
    forwardNoGrad: Number(match[2]),
    backward: Number(match[3]),
    softmax: Number(match[4])
  };
};

const runOnce = (
  label: string,
  command: string,
  args: string[],
  env?: NodeJS.ProcessEnv
): Result => {
  const result = spawnSync(command, args, {
    env: env ?? process.env,
    encoding: "utf8"
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`${label} failed with exit code ${result.status}`);
  }
  return parseResult(result.stdout ?? "", label);
};

const summarize = (values: number[]) => {
  const total = values.reduce((acc, cur) => acc + cur, 0);
  return total / values.length;
};

const printSummary = (
  label: string,
  tensor: Result[],
  pytorch: Result[]
): void => {
  const tensorForward = summarize(tensor.map((r) => r.forward));
  const tensorForwardNoGrad = summarize(tensor.map((r) => r.forwardNoGrad));
  const tensorBackward = summarize(tensor.map((r) => r.backward));
  const tensorSoftmax = summarize(tensor.map((r) => r.softmax));
  const torchForward = summarize(pytorch.map((r) => r.forward));
  const torchForwardNoGrad = summarize(pytorch.map((r) => r.forwardNoGrad));
  const torchBackward = summarize(pytorch.map((r) => r.backward));
  const torchSoftmax = summarize(pytorch.map((r) => r.softmax));

  const deltaForward = torchForward - tensorForward;
  const deltaForwardNoGrad = torchForwardNoGrad - tensorForwardNoGrad;
  const deltaBackward = torchBackward - tensorBackward;
  const deltaSoftmax = torchSoftmax - tensorSoftmax;
  const pctForward = (deltaForward / torchForward) * 100;
  const pctForwardNoGrad = (deltaForwardNoGrad / torchForwardNoGrad) * 100;
  const pctBackward = (deltaBackward / torchBackward) * 100;
  const pctSoftmax = (deltaSoftmax / torchSoftmax) * 100;

  console.log(`\n== ${label} Summary ==`);
  console.log(`Tensor forward avg:   ${tensorForward.toFixed(2)} ms`);
  console.log(`PyTorch forward avg:  ${torchForward.toFixed(2)} ms`);
  console.log(
    `Delta forward:        ${deltaForward.toFixed(2)} ms (${pctForward.toFixed(1)}%)`
  );
  console.log(`Tensor no-grad avg:   ${tensorForwardNoGrad.toFixed(2)} ms`);
  console.log(`PyTorch no-grad avg:  ${torchForwardNoGrad.toFixed(2)} ms`);
  console.log(
    `Delta no-grad:        ${deltaForwardNoGrad.toFixed(2)} ms (${pctForwardNoGrad.toFixed(1)}%)`
  );
  console.log(`Tensor backward avg:  ${tensorBackward.toFixed(2)} ms`);
  console.log(`PyTorch backward avg: ${torchBackward.toFixed(2)} ms`);
  console.log(
    `Delta backward:       ${deltaBackward.toFixed(2)} ms (${pctBackward.toFixed(1)}%)`
  );
  console.log(`Tensor softmax avg:   ${tensorSoftmax.toFixed(2)} ms`);
  console.log(`PyTorch softmax avg:  ${torchSoftmax.toFixed(2)} ms`);
  console.log(
    `Delta softmax:        ${deltaSoftmax.toFixed(2)} ms (${pctSoftmax.toFixed(1)}%)`
  );
};

const buildMarkdownSummary = (
  label: string,
  tensor: Result[],
  pytorch: Result[]
): string => {
  const tensorForward = summarize(tensor.map((r) => r.forward));
  const tensorForwardNoGrad = summarize(tensor.map((r) => r.forwardNoGrad));
  const tensorBackward = summarize(tensor.map((r) => r.backward));
  const tensorSoftmax = summarize(tensor.map((r) => r.softmax));
  const torchForward = summarize(pytorch.map((r) => r.forward));
  const torchForwardNoGrad = summarize(pytorch.map((r) => r.forwardNoGrad));
  const torchBackward = summarize(pytorch.map((r) => r.backward));
  const torchSoftmax = summarize(pytorch.map((r) => r.softmax));

  const rows = [
    ["forward", tensorForward, torchForward],
    ["forward_nograd", tensorForwardNoGrad, torchForwardNoGrad],
    ["backward", tensorBackward, torchBackward],
    ["softmax", tensorSoftmax, torchSoftmax]
  ];

  const lines = [
    `### ${label}`,
    "",
    "| metric | Tensor avg (ms) | PyTorch avg (ms) | Delta (ms) | Delta (%) |",
    "| --- | ---: | ---: | ---: | ---: |"
  ];

  for (const [metric, tensorVal, torchVal] of rows) {
    const delta = torchVal - tensorVal;
    const pct = (delta / torchVal) * 100;
    lines.push(
      `| ${metric} | ${tensorVal.toFixed(2)} | ${torchVal.toFixed(
        2
      )} | ${delta.toFixed(2)} | ${pct.toFixed(1)}% |`
    );
  }

  return lines.join("\n");
};

const runSeries = (
  label: string,
  command: string,
  args: string[],
  env?: NodeJS.ProcessEnv
): Result[] => {
  const results: Result[] = [];
  for (let i = 0; i < RUNS; i++) {
    const runLabel = `${label} run ${i + 1}`;
    const result = runOnce(runLabel, command, args, env);
    if (i >= DISCARD) {
      results.push(result);
    }
  }
  return results;
};

const main = () => {
  console.log("RUNNING BENCHMARK SCRIPTS");
  const tensorResults = runSeries("Tensor (TS)", "npx", [
    "tsx",
    "scripts/bench-tensor.ts"
  ]);
  const torchResults = runSeries("PyTorch", "python3", [
    "scripts/bench-pytorch.py"
  ]);
  printSummary("Benchmark (TS)", tensorResults, torchResults);

  console.log("\nMARKDOWN SUMMARY\n");
  console.log(buildMarkdownSummary("Benchmark (TS)", tensorResults, torchResults));
};

main();
