### FLOPs and Parameter Accounting

#### Configuration 1: GPT-2-XL scale

```python
vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=4288
```

**Forward-pass FLOPs**

The part that scales with `num_layers` breaks into four pieces per layer:

- QKV projection: $1600\times1600\times1024\times2\times3$
- $QK^\top$ and attention@V: $1024\times1024\times1600\times2\times2$
- Output projection $W_O$: $1600\times1600\times1024\times2$
- FFN: $1600\times4288\times1024\times2\times2 + 4288\times1600\times1024\times2$

The part that's computed only once for the whole model, not multiplied by layer count:

- LM head: $1600\times50257\times1024\times2$

**AdamW's per-step FLOPs**

AdamW's FLOPs for one step depend only on the total number of scalar parameters in the model — not on how those parameters are shaped into matrices. The reason is that AdamW's update rule contains **no matrix multiplication at all**. Looking back at the `step()` implementation:

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

- Token embedding: $\text{vocab\\_size}\times d_{\text{model}} = 50257\times1600 = 80{,}411{,}200$
- Per layer: $4\times(1600\times1600)+2\times1600+3\times(4288\times1600)=10{,}240{,}000+3{,}200+20{,}582{,}400=30{,}825{,}600$
- Across all 48 layers: $30{,}825{,}600\times48=1{,}479{,}628{,}800$
- Final RMSNorm: $1{,}600$
- LM head: $50257\times1600=80{,}411{,}200$

Sum: $80{,}411{,}200+1{,}479{,}628{,}800+1{,}600+80{,}411{,}200=1{,}640{,}452{,}800$ — roughly 1.64 billion parameters, matching the expected result.

**Memory footprint (parameters alone)**

$$
1{,}640{,}452{,}800 \times 4\ \text{bytes (float32)} = 6{,}561{,}811{,}200\ \text{bytes} \approx 6.56\ \text{GB}
$$

---

#### Configuration 2: custom small model

```python
vocab_size=10000, context_length=256, d_model=512, d_ff=1344, theta=10000, num_layers=4, num_heads=16
```

**Fixed part: parameter-related memory (independent of batch_size)**

- Token embedding $= \text{vocab\\_size}\times d_{\text{model}} = 10000\times512 = 5{,}120{,}000$
- Per-layer $W_Q, W_K, W_V, W_O = 4\times d_{\text{model}}^2 = 4\times(512\times512) = 1{,}048{,}576$
- Per-layer's two RMSNorms $= 2\times d_{\text{model}} = 2\times512 = 1{,}024$
- Per-layer $W_1, W_2, W_3$ (SwiGLU) $= 3\times d_{ff}\times d_{\text{model}} = 3\times1344\times512 = 2{,}064{,}384$
- Final RMSNorm $= d_{\text{model}} = 512$
- LM head $= \text{vocab\\_size}\times d_{\text{model}} = 10000\times512 = 5{,}120{,}000$

Total parameter count: $22{,}696{,}448$.

This parameter count is exactly what determines three of the fixed items in memory usage (none of which scale with batch_size): the parameters themselves ($\times4$ bytes), the gradients (one per parameter, likewise $\times4$ bytes), and the AdamW optimizer state (`m` and `v`, one each per parameter, so $\times4$ bytes $\times2$). Together these amount to four full copies of the parameter count:

$$
22{,}696{,}448 \times 4 = 90{,}785{,}792\ \text{("parameter-equivalents")},\quad 90{,}785{,}792\times4\ \text{bytes} = 363{,}143{,}168\ \text{bytes}
$$

**Variable part: activation-related memory (scales linearly with batch_size)**

The key here is to pin down each intermediate tensor's **shape** first, then count elements — these shapes naturally carry a `batch_size` dimension, so their element counts scale proportionally with batch_size, which is an entirely separate calculation from the "fixed part" parameter count above.

QKV projection, the weighted sum (after `softmax(QK^T/√d_k)@V`), and the output projection: since $h\times d_k = d_{\text{model}}$ (splitting into multiple heads just breaks up the last dimension without changing the total element count), all three have a total element count equivalent to $(\text{batch\\_size}, \text{context\\_length}, d_{\text{model}})$.

$QK^\top$ and softmax are the exception: these must be counted as $\text{batch\\_size}\times\text{num\\_heads}\times\text{context\\_length}^2$, not as $(\text{batch\\_size}, \text{context\\_length}, d_{\text{model}})$ — this matters because `context_length` appears squared here, and its magnitude is usually much larger than the other terms.

The FFN needs its shape traced step by step through the SwiGLU formula $\text{FFN}(x)=W_2(\text{SiLU}(W_1x)\odot W_3x)$: $W_1x$ maps from $d_{\text{model}}$ to $d_{ff}$, giving shape $(\text{batch\\_size},\text{context\\_length},d_{ff})$; SiLU acts elementwise, leaving the shape unchanged; $W_3x$ is likewise a $d_{\text{model}}\to d_{ff}$ mapping, also shape $d_{ff}$; the elementwise product of the two keeps shape $d_{ff}$; only after $W_2(\cdot)$ maps $d_{ff}$ back down to $d_{\text{model}}$ does the output shape return to $(\text{batch\\_size},\text{context\\_length},d_{\text{model}})$ — so the FFN contributes 4 intermediate tensors of size $d_{ff}$ plus 1 output tensor of size $d_{\text{model}}$.

The final RMSNorm is counted as $(\text{batch\\_size},\text{context\\_length},d_{\text{model}})$; the LM head and cross-entropy are counted as $(\text{batch\\_size},\text{context\\_length},\text{vocab\\_size})$.

Listed out (each figure is the element count per unit of batch_size):

- Per-layer's 2 RMSNorms: $\text{context\\_length}\times d_{\text{model}}\times2 = 262{,}144$
- QKV projection: $\text{context\\_length}\times d_{\text{model}}\times3 = 393{,}216$
- $QK^\top$: $\text{num\\_heads}\times\text{context\\_length}^2 = 1{,}048{,}576$
- softmax: $\text{num\\_heads}\times\text{context\\_length}^2 = 1{,}048{,}576$
- Weighted sum: $\text{context\\_length}\times d_{\text{model}} = 131{,}072$
- Output projection: $\text{context\\_length}\times d_{\text{model}} = 131{,}072$
- FFN (4 intermediates of size $d_{ff}$ + 1 output of size $d_{\text{model}}$): $1{,}507{,}328$
- Final RMSNorm: $131{,}072$
- LM head: $\text{context\\_length}\times\text{vocab\\_size} = 2{,}560{,}000$
- Cross-entropy: $\text{context\\_length}\times\text{vocab\\_size} = 2{,}560{,}000$

Total: $23{,}339{,}008\times\text{batch\\_size}$ activation elements.

**Part 2 — to be continued.**
