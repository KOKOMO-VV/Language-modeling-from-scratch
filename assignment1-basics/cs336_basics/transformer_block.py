import torch
import torch.nn as nn
from .transformer_rmsnorm import RMSnorm
from .transformer_swiglu import SwiGLU
from .casual_multihead_self_attention import CasualMultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: int | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        self.norm1 = RMSnorm(d_model, device=None, dtype=None)
        self.norm2 = RMSnorm(d_model, device=None, dtype=None)
        self.attention = CasualMultiHeadSelfAttention(
            d_model, num_heads, theta, max_seq_len
        )
        self.ffn = SwiGLU(d_model, d_ff, device=None, dtype=None)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        temp_result = x + self.attention.forward(self.norm1.forward(x))
        result = temp_result + self.ffn.forward(self.norm2.forward(temp_result))
        return result
