following along with andrej karpathy's youtube ai course

in typescript

for fun

it's not illegal

**update**
There is still a lot I could do to optimize, but matmul is just not gonna happen at any kind of scale with this setup
gonna keep going, just will never actually build anything big



### Benchmark (TS)

| metric | Tensor avg (ms) | PyTorch avg (ms) | Delta (ms) | Delta (%) |
| --- | ---: | ---: | ---: | ---: |
| forward | 50.13 | 2.44 | -47.69 | -1956.0% |
| forward_nograd | 45.55 | 1.01 | -44.54 | -4428.2% |
| backward | 154.02 | 11.70 | -142.33 | -1216.9% |
| softmax | 5.90 | 0.52 | -5.38 | -1038.0% |
