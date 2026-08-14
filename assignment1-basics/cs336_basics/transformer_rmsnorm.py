import torch
import torch.nn as nn


class RMSnorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        tensor = torch.ones(d_model, device=device, dtype=dtype)
        self.gain = nn.Parameter(tensor)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS_a = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x / RMS_a) * self.gain
        return result.to(in_dtype)
