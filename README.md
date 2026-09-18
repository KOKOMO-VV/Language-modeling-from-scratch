# CS336 Assignment 1 — Learning Notes

Personal notes written while working through Stanford CS336's Assignment 1 (Basics) on my own. These reflect my own learning process and understanding — not official course content, and not a solution set.

## Contents

### [transformer-build-notes.md](./transformer-build-notes.md)

The main set of notes, organized chapter by chapter around the components implemented in this assignment. Each chapter follows a **Why → How** structure — first the motivation for why a component is needed at all, then how it's actually built.

- **1. Tokenizer** — why tokenization is needed, then building a byte-level BPE tokenizer: pre-tokenization, vocabulary training, and the encode/decode pipeline.
- **2. Transformer** — why the Transformer architecture looks the way it does (compared to RNNs), then building it piece by piece: embeddings, RoPE, causal multi-head self-attention, RMSNorm, SwiGLU, and how it all assembles into a full model.
- **3. Optimizer** — why plain SGD isn't enough, then building AdamW: momentum, adaptive per-parameter learning rates, decoupled weight decay, and how the update loop fits into PyTorch's `Optimizer` interface.

### [Writups.md](./Writups.md)

Worked-out FLOPs, parameter count, and memory accounting for the models built in these notes — how many parameters a given configuration has, where the FLOPs are spent during a forward pass and an AdamW step, and which parts of memory usage are fixed (parameters/gradients/optimizer state) versus scale with batch size (activations).

Part 1 (FLOPs & parameter accounting) is done — Part 2 to come.

## Disclaimer

This repository documents my own learning process for CS336. It is not affiliated with or endorsed by the course staff, and isn't intended to be used as a solution reference.
