following along with andrej karpathy's youtube ai course

in typescript

for fun

it's not illegal

**update**
There is still a lot I could do to optimize, but matmul is just not gonna happen at any kind of scale with this setup
gonna keep going, just will never actually build anything big



== Benchmark Summary ==
Tensor forward avg:   49.99 ms
PyTorch forward avg:  2.14 ms
Delta forward:        -47.85 ms (-2233.3%)
Tensor no-grad avg:   45.73 ms
PyTorch no-grad avg:  0.99 ms
Delta no-grad:        -44.74 ms (-4535.0%)
Tensor backward avg:  152.23 ms
PyTorch backward avg: 11.51 ms
Delta backward:       -140.72 ms (-1223.1%)