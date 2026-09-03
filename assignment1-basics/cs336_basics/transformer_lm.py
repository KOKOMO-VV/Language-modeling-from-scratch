import torch
import torch.nn as nn
from .transformer_block import TransformerBlock
from .transformer_embedding import Embedding
from .transformer_rmsnorm import RMSnorm
from .transformer_linear import Linear


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: int | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=None, dtype=None)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, theta, context_length)
                for i in range(num_layers)
            ]
        )
        self.ln_final = RMSnorm(d_model, device=None, dtype=None)
        self.lm_head = Linear(d_model, vocab_size, device=None, dtype=None)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        temp_result = self.token_embeddings(token_ids)
        for layer in self.layers:
            temp_result = layer.forward(temp_result)
        rms_ln_final = self.ln_final.forward(temp_result)
        result = self.lm_head.forward(rms_ln_final)
        return result
