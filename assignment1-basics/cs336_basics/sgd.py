from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


def lr_cosine_schedule(t, a_max, a_min, t_w, t_c):
    if t < t_w:
        return a_max * t / t_w
    elif t_w <= t <= t_c:
        return a_min + 0.5 * (a_max - a_min) * (
            1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))
        )
    else:
        return a_min


def gradient_clipping(params, l_max):
    eps = 1e-6
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            total_norm += (p.grad.data**2).sum().item()
    total_norm = math.sqrt(total_norm)
    if total_norm < l_max:
        return
    else:
        for p in params:
            if p.grad is not None:
                p.grad.data = (l_max / (total_norm + eps)) * p.grad.data


# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1e3)
# for t in range(100):
#     opt.zero_grad()  # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean()  # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward()  # Run backward pass, which computes gradients.
#     opt.step()  # Run optimizer step.
