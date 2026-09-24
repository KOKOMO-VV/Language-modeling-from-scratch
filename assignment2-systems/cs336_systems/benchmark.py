# benchmark.py
import argparse
import timeit

import numpy as np
import torch
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adam_w import AdamW
from cs336_basics.cross_entropy import cross_entropy

MODEL_SIZES = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", choices=MODEL_SIZES.keys(), default="small")
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--measure_steps", type=int, default=10)
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

    # use random inputs for benchmarking
    x = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length), device=args.device
    )
    y = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length), device=args.device
    )

    def step():
        if args.mode == "full":
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if args.mode != "forward":
            loss = cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
            loss.backward()
            if args.mode == "full":
                optimizer.step()
        torch.cuda.synchronize()

    # 热身：不计时
    for _ in range(args.warmup_steps):
        step()

    # 正式计时
    # use timeit for more accurate timing (ns precision)
    timings = []
    for _ in range(args.measure_steps):
        start = timeit.default_timer()
        step()
        timings.append(timeit.default_timer() - start)

    timings = np.array(timings)
    print(
        f"model={args.model_size} mode={args.mode} "
        f"warmup={args.warmup_steps} n={args.measure_steps} "
        f"mean={timings.mean()*1000:.2f}ms std={timings.std()*1000:.2f}ms"
    )


if __name__ == "__main__":
    main()
