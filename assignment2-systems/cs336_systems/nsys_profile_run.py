"""
2.1.4 nsys_profile: run one (model_size, context_length, mode) combo under
`nsys profile`, with NVTX ranges around forward / backward / optimizer, and
around the attention internals (via nvtx_attention.apply_nvtx_patch()).

Run this SCRIPT wrapped by `nsys profile`, e.g.:

  uv run nsys profile -o results/nsys_small_ctx256 --force-overwrite=true \
      python cs336_systems/nsys_profile_run.py \
      --model_size small --context_length 256 --mode full
"""

import argparse

import torch
from torch.cuda import nvtx

# IMPORTANT: patch before importing TransformerLM, so the model's attention
# submodules pick up the annotated function when they're constructed.
from cs336_systems.nvtx_attention import apply_nvtx_patch

apply_nvtx_patch()

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adam_w import AdamW
from cs336_basics.cross_entropy import cross_entropy

MODEL_SIZES = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", choices=MODEL_SIZES.keys(), default="small")
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--measure_steps", type=int, default=5)
    p.add_argument(
        "--mode", choices=["forward", "forward_backward", "full"], default="full"
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = get_args()
    cfg = MODEL_SIZES[args.model_size]

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=cfg["num_layers"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
    ).to(args.device)
    optimizer = AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )

    x = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length), device=args.device
    )
    y = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length), device=args.device
    )

    def step():
        if args.mode == "full":
            optimizer.zero_grad(set_to_none=True)
        with nvtx.range("forward"):
            logits = model(x)
        if args.mode != "forward":
            with nvtx.range("backward"):
                loss = cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
                loss.backward()
            if args.mode == "full":
                with nvtx.range("optimizer"):
                    optimizer.step()
        torch.cuda.synchronize()

    for _ in range(args.warmup_steps):
        with nvtx.range("warmup_step"):
            step()

    for i in range(args.measure_steps):
        with nvtx.range(f"measure_step_{i}"):
            step()


if __name__ == "__main__":
    main()
