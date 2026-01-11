import time
import torch

device = "mps"  # set to "mps" or "cuda" if desired
shape_a = (256, 256)
shape_b = (256, 256)


def forward():
    a = torch.randn(shape_a, device=device, requires_grad=True)
    b = torch.randn(shape_b, device=device, requires_grad=True)
    return (a @ b).relu().sum()

def forward_no_grad():
    a = torch.randn(shape_a, device=device, requires_grad=False)
    b = torch.randn(shape_b, device=device, requires_grad=False)
    return (a @ b).relu().sum()

def softmax_forward():
    a = torch.randn(shape_a, device=device, requires_grad=False)
    return torch.softmax(a, dim=-1)


def time_once(fn):
    t0 = time.time()
    fn()
    return (time.time() - t0) * 1000


forward_ms = time_once(forward)
forward_nograd_ms = time_once(forward_no_grad)
backward_ms = time_once(lambda: forward().backward())
softmax_ms = time_once(softmax_forward)

print(
    f"RESULT forward_ms={forward_ms:.4f} forward_nograd_ms={forward_nograd_ms:.4f} backward_ms={backward_ms:.4f} softmax_ms={softmax_ms:.4f}"
)
