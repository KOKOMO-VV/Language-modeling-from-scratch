import torch
import torch.nn as nn
import math
from einops import einsum as einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)
        mean = 0.0
        variance = 2 / (in_features + out_features)
        std = math.sqrt(variance)

        self.weight = nn.Parameter(
            torch.nn.init.trunc_normal_(tensor, mean, std, -3 * std, 3 * std)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "o i, ... i -> ... o")
