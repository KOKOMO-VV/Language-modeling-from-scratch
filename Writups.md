### FLOPs and Parameter Accouning

#### Configuration 1: GPT-2-XL scale

```python
vocab_size=50257, context_length=1024, num_layers=48, dmodel=1600, num_heads=25, dff=4288
```

**Forward-pass FLOPs**

The part that scales with `num_layers` breaks into four pieces per layer:

- QKV projection: $1600\times1600\times1024\times2\times3$
- QK⊤QK^\top QK⊤ and attention@V: $1024\times1024\times1600\times2\times2$
- Output projection WOW_O WO: $1600\times1600\times1024\times2$
- FFN: $1600\times4288\times1024\times2\times2 + 4288\times1600\times1024\times2$ 

The part that's computed only once for the whole model, not multiplied by layer count:

- LM head: $1600\times50257\times1024\times2$

**AdamW's per-step FLOPs**

AdamW's FLOPs for one step depend only on the total number of scalar parameters in the model — not on how those parameters are shaped into matrices. The reason is that AdamW's update rule contains **no matrix multiplication at all**. Looking back at the `step()` implementation:

python

```python
p.data -= lr * weight_decay * p.data
m = betas[0] * m + (1 - betas[0]) * grad
v = betas[1] * v + (1 - betas[1]) * grad**2
p.data -= (lr_t * m) / (torch.sqrt(v) + eps)
```

all four lines are purely elementwise operations — add, subtract, multiply, divide, square root. Counting how many operations each parameter goes through, line by line:

Line 1, `p.data -= lr * weight_decay * p.data`: 2 multiplies (`weight_decay * p.data`, `lr * (...)`) and 1 subtract — 3 operations.

Line 2, `m = betas[0]*m + (1-betas[0])*grad`: 2 multiplies (`1-betas[0]` is a scalar computed once, not counted per parameter) and 1 add — 3 operations.

Line 3, `v = betas[1]*v + (1-betas[1])*grad**2`: 1 square, 2 multiplies, 1 add — 4 operations.

Line 4, `p.data -= (lr_t*m) / (torch.sqrt(v)+eps)`: 1 square root, 1 add, 1 multiply, 1 divide, 1 subtract — 5 operations.

Summing across all four lines gives $3+3+4+5=15$; multiplying this by the model's total parameter count gives the total FLOPs needed to run one full AdamW step.

**Parameter count (verification)**

- Token embedding: vocab_size$×$$d_{model}$$=50257×1600 = 80,411,200$
- Per layer: $4×(1600×1600)+2×1600+3×(4288×1600)=10,240,000+3,200+20,582,400=30,825,600$ 
- Across all 48 layers: $30,825,600×48=1,479,628,800$
- Final RMSNorm: $1,600$
- LM head: $50257×1600=80,411,200$

Sum: $80,411,200+1,479,628,800+1,600+80,411,200=1,640,452,800$— roughly 1.64 billion parameters, matching the expected result.

**Memory footprint (parameters alone)**

$1,640,452,800×4 bytes (float32)=6,561,811,200 bytes≈6.56 GB$

------

#### Configuration 2: custom small model

```python
vocab_size=10000, context_length=256, dmodel=512, dff=1344, Θ=10000, num_layers=4, num_heads=16
```

**Fixed part: parameter-related memory (independent of batch_size)**

- Token embedding =vocab_size$×d_{model}$$=10,000×512=5,120,000$
- Per-layer $W_Q,W_K,W_V,W_O$$=4×d_{model}^2=4×(512×512)=1,048,576$
- Per-layer's two RMSNorms $=2×d_{model}=2×512=1,024$
- Per-layer $W_1, W_2, W_3 (SwiGLU) =3×d_{ff}×d_{model}=3×1344×512=2,064,384$
- Final RMSNorm $=d_{model}=512$
- LM head =vocab_size$×d_{model}$$=10000×512=5,120,000$

Total parameter count: $22,696,448$

This parameter count is exactly what determines three of the fixed items in memory usage (none of which scale with batch_size): the parameters themselves ($×4 bytes$), the gradients (one per parameter, likewise $×4 bytes$), and the AdamW optimizer state (`m` and `v`, one each per parameter, so $×4 bytes ×2$). Together these amount to four full copies of the parameter count:

$22,696,448×4=90,785,792 ("parameter-equivalents"),90,785,792×4 bytes=363,143,168 bytes$

**Variable part: activation-related memory (scales linearly with batch_size)**

The key here is to pin down each intermediate tensor's **shape** first, then count elements — these shapes naturally carry a `batch_size` dimension, so their element counts scale proportionally with batch_size, which is an entirely separate calculation from the "fixed part" parameter count above.

QKV projection, the weighted sum (after `softmax(QK^T/√d_k)@V`), and the output projection: since $h×d_k=d_{model}$ (splitting into multiple heads just breaks up the last dimension without changing the total element count), all three have a total element count equivalent to **(batch_size,context_length,d_model)**.

$QK⊤$ and softmax are the exception: these must be counted as batch_size $×$ num_heads $×$ context_length$^2$ not as **(batch_size,context_length,d_model)**— this matters because `context_length` appears squared here, and its magnitude is usually much larger than the other terms.

The FFN needs its shape traced step by step through the SwiGLU formula $FFN(x)=W2(SiLU(W1x)⊙W3x)$\: $W_1x$ maps from $d_{model}$ to $d_{ff}$, giving shape **(batch_size,context_length,d_ff)**; **SiLU** acts elementwise, leaving the shape unchanged; $W_3x$ is likewise a dmodel→$d_{\text{model}}\to d_{ff}$  mapping, also shape $d_{ff}$ ; the elementwise product of the two keeps shape $d_{ff}$; only after $W_2(\cdot)$ maps $d_{ff}$ back down to $d_{\text{model}}$ does the output shape return to **(batch_size,context_length,d_model)** — so the FFN contributes 4 intermediate tensors of size $d_{ff}$ plus 1 output tensor of size $d_{\text{model}}$.

Listed out (each figure is the element count per unit of batch_size):

- Per-layer's 2 RMSNorms: context_length $×$$ d_{model}$$×2=262,144$
- $QKV$ projection: context_length $× d_{model} ×3=393,216$
- $QK⊤$: num_heads $×$ context_length$^2$ $= 1,048,576$
- softmax: num_heads $×$ context_length$^2$ $= 1,048,576$
- Weighted sum: context_length $× d_{model} = 131,072$
- Output projection: context_length $× d_{model} = 131,072$
- FFN (4 intermediates of size $d_{ff}$  + 1 output of size $d_{\text{model}}$): $1,507,3281$
- Final RMSNorm: $131,072$
- LM head: context_length $×$ vocab_size $=2,560,000$
- Cross-entropy: context_length $×$ vocab_size $= 2,560,000$

Total: $23,339,008 ×$ batch_size activation elements.

