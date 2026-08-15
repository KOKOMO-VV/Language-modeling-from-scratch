import torch
import torch.nn as nn
from .transformer_linear import Linear
from .transformer_rope import RoPE
from einops import rearrange
from .transformer_attention import (
    softmax as softmax,
    scaled_dot_product_attention as scaled_dot_product_attention,
)


class CasualMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: int | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = Linear(d_model, d_model, device=None, dtype=None)
        self.w_k = Linear(d_model, d_model, device=None, dtype=None)
        self.w_v = Linear(d_model, d_model, device=None, dtype=None)
        self.w_o = Linear(d_model, d_model, device=None, dtype=None)
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device=None)
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. use w_q, w_k, w_v to project x, and get whole Q, K, V
        Q = self.w_q.forward(x)
        K = self.w_k.forward(x)
        V = self.w_v.forward(x)

        # 2. split d_model of Q K V to (num_heads, d_k) and move num_heads to the front of seq_len
        new_Q = rearrange(
            Q,
            "... seq (heads d_k) -> ... heads seq d_k",
            heads=self.num_heads,
        )
        new_K = rearrange(
            K,
            "... seq (heads d_k) -> ... heads seq d_k",
            heads=self.num_heads,
        )
        new_V = rearrange(
            V,
            "... seq (heads d_k) -> ... heads seq d_k",
            heads=self.num_heads,
        )

        # 3. rope Q K

        # 3.1 generate token_positions
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(0, seq_len)

        if self.rope is not None:
            new_Q = self.rope.forward(new_Q, token_positions)
            new_K = self.rope.forward(new_K, token_positions)

        # 4. create mask(seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).to(torch.bool)

        # 5. use dot_production to calculate attention of rope_Q rope_K and new_V
        attention_result = scaled_dot_product_attention(new_Q, new_K, new_V, mask)

        # 6. turn the split results to the whole (... , seq_len, d_model)
        attention_result_whole = rearrange(
            attention_result, "... heads seq d_k -> ... seq (heads d_k) "
        )

        # 7. use w_o to do the final projection
        result = self.w_o.forward(attention_result_whole)

        return result
