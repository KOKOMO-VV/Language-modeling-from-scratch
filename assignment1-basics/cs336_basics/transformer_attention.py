import torch
from einops import einsum
import math


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    new_x = x - x.max(dim=dim, keepdim=True).values
    softmax_x = torch.exp(new_x) / torch.sum(torch.exp(new_x), dim=dim, keepdim=True)
    return softmax_x


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = Q.shape[-1]
    temp_qk = einsum(Q, K, "... n d , ... m d -> ... n m")
    new_temp = temp_qk / math.sqrt(d_k)
    if mask is not None:
        mask = torch.where(mask, 0, float("-inf"))
        new_temp = new_temp + mask
    softmax_r = softmax(new_temp, dim=-1)
    r = einsum(softmax_r, V, "... n m , ... m d -> ... n d")
    return r
