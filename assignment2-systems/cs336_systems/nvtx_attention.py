"""
NVTX-annotated version of scaled_dot_product_attention, for use with `nsys profile`.

Import `apply_nvtx_patch()` and call it BEFORE importing/constructing the model
(TransformerLM builds its attention modules at __init__ time, and those modules
already bound the *original* `scaled_dot_product_attention` name into their own
module namespace via `from ... import scaled_dot_product_attention`, so we patch
both the source module and the module that imported it).
"""

import math

import torch
from torch.cuda import nvtx
from einops import einsum

import cs336_basics.transformer_attention as ta


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = Q.shape[-1]

    with nvtx.range("computing attention scores"):
        temp_qk = einsum(Q, K, "... n d , ... m d -> ... n m")
        new_temp = temp_qk / math.sqrt(d_k)
        if mask is not None:
            mask_ = torch.where(mask, 0, float("-inf"))
            new_temp = new_temp + mask_

    with nvtx.range("computing softmax"):
        softmax_r = ta.softmax(new_temp, dim=-1)

    with nvtx.range("final matmul"):
        r = einsum(softmax_r, V, "... n m , ... m d -> ... n d")

    return r


def apply_nvtx_patch() -> None:
    # patch the definition site
    ta.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    # patch the call site(s): any module that did
    # `from cs336_basics.transformer_attention import scaled_dot_product_attention`
    # holds its own reference and won't see the line above.
    import cs336_basics.casual_multihead_self_attention as attn_mod

    attn_mod.scaled_dot_product_attention = annotated_scaled_dot_product_attention
