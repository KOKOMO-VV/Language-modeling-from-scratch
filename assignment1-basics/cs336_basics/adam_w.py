from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 0)  # Get iteration number from the state, or 0.\
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                lr_t = lr * (
                    math.sqrt(1 - betas[1] ** (t + 1)) / (1 - betas[0] ** (t + 1))
                )  # Update learning rate.
                p.data -= lr * weight_decay * p.data  # Update weight decay.
                m = betas[0] * m + (1 - betas[0]) * grad  # Update first moment vector.
                v = betas[1] * v + (1 - betas[1]) * grad**2
                p.data -= (lr_t * m) / (torch.sqrt(v) + eps)  # Update weight decay.
                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m  # Update first moment vector in the state.
                state["v"] = v  # Update second moment vector in the state.
        return loss
