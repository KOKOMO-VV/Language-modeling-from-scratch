import torch
import torch.nn as nn
from .transformer_linear import Linear


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = round((8 * d_model) / 3 / 64) * 64
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp_x = self.w1.forward(x)
        result1 = temp_x * torch.sigmoid(temp_x)
        result2 = self.w3.forward(x)
        result = result1 * result2
        return self.w2.forward(result)
