import torch
import torch.nn as nn
import math


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        # use torch to create a sequence tensor
        seq_tensor = torch.arange(max_seq_len, device=device)

        # use torch to create a k_tensor
        k_tensor = torch.arange(1, d_k // 2 + 1, device=device)

        temp_theta = theta ** ((2 * k_tensor - 2) / d_k)

        # reshape the sequence tensor so that it could divide temp_theta
        seq_tensor_reshape = seq_tensor.reshape(max_seq_len, 1)

        angle_ik = seq_tensor_reshape / temp_theta
        cos_table = torch.cos(angle_ik)
        sin_table = torch.sin(angle_ik)

        # save the two tables to register_buffer without saving to static_dict()
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]
        x_split = [x[..., 0::2], x[..., 1::2]]
        x_split_new = x_split.copy()
        x_split_new[0] = x_split[0] * cos - x_split[1] * sin
        x_split_new[1] = x_split[0] * sin + x_split[1] * cos
        # or use .flatten(start_dim=-2)
        x_new = torch.stack([x_split_new[0], x_split_new[1]], dim=-1).reshape(x.shape)
        return x_new
