```python
vocab_size=50257`、`context_length=1024`、`num_layers=48`、`d_model=1600`、`num_heads=25`,`d_ff=4288
```

FLOPS

**每一层(要乘以 `num_layers`)的部分:**

- QKV 投影: $1600×1600×1024×2×3$ (Embedding)
- QK⊤ + attention@V: $1024×1024×1600×2×2$
- 输出投影 WO:$1600×1600×1024×2$
- FFN:$1600×4288×1024×2×2+4288×1600×1024×2$

**不乘 `num_layers` 的部分(整个模型只算一次):**

- LM head: $1600×50257×1024×2$

**AdamW 这一步的 FLOPs,只取决于"模型总共有多少个参数值(标量)",不取决于每个参数矩阵的具体形状** 关键原因是:AdamW 的更新公式里,**没有任何矩阵乘法**——你回头看一下自己写的 `step()`:

python

```python
p.data -= lr * weight_decay * p.data
m = betas[0] * m + (1 - betas[0]) * grad
v = betas[1] * v + (1 - betas[1]) * grad**2
p.data -= (lr_t * m) / (torch.sqrt(v) + eps)
```

这几行全部都是**逐元素(elementwise)**运算——加、减、乘、除、开方

**第一行:`p.data -= lr \* weight_decay \* p.data`**

- `weight_decay * p.data`(1 次乘)
- `lr * (...)`(1 次乘)
- `p.data - (...)`(1 次减)
   → 3 次运算

**第二行:`m = betas[0] \* m + (1 - betas[0]) \* grad`**

- `betas[0] * m`(1 次乘)
- `(1-betas[0]) * grad`(1 次乘,`1-betas[0]` 是标量提前算好的,不算在每个参数头上)
- 两者相加(1 次加)
   → 3 次运算

**第三行:`v = betas[1] \* v + (1 - betas[1]) \* grad\**2`**

- `grad**2`(1 次乘,平方)
- `betas[1] * v`(1 次乘)
- `(1-betas[1]) * (...)`(1 次乘)
- 两者相加(1 次加)
   → 4 次运算

**第四行:`p.data -= (lr_t \* m) / (torch.sqrt(v) + eps)`**

- `torch.sqrt(v)`(1 次开方)
- `(...) + eps`(1 次加)
- `lr_t * m`(1 次乘)
- 除法(1 次除)
- `p.data - (...)`(1 次减)
   → 5 次运算

加起来:3+3+4+5=15 乘以模型的总参数量,就是运行一步 AdamW 所需的总 FLOP

- token embedding: **vocab_size x d_model** 50257×1600=80,411,200
- 每层: 4×(1600×1600)+2×1600+3×(4288×1600)=10,240,000+3,200+20,582,400=30,825,600
- 48 层总计: 30,825,600×48=1,479,628,800
- 最后的 RMSNorm: 1,600
- LM head: 50257×1600=80,411,200

加起来: 80,411,200+1,479,628,800+1,600+80,411,200=1,640,452,800

正好对上你算出来的 **1,640,452,800**(约 16.4 亿个参数)。

总参数量 1,640,452,800 × 每个参数 4 字节(float32)= 6,561,811,200 字节

按 1GB=10^9 1GB 字节换算:6,561,811,200÷10^9≈6.56GB

==Vocab size 10000==  

==Context length 256== 

==d_model 512== 

==d_ff 1344== 

==RoPE theta parameter Θ 10000== 

==Number of layers and heads 4 layers, 16 heads.==

38.056.448 个参数

**而这个参数量,恰好同时决定了内存占用里的三项(都是"固定部分",不随 batch_size 变化):**

- 参数本身占用的内存(参数量 × 4 字节)
- 梯度占用的内存(每个参数都有一个对应的梯度,同样是参数量 × 4 字节)
- AdamW 优化器状态占用的内存(`m` 和 `v` 各自也是每个参数一份,所以是参数量 × 4 字节 × 2)



**唯一跟 batch_size 相关的第四项——激活值(activations)——完全是另外单独算的**,跟这个参数量数字没有任何关系。它的大小取决于前向传播过程中,各个中间张量(比如 QKV、attention 分数、FFN 中间结果)的**形状**——而这些形状里,天然就带着一个 `batch_size` 维度(比如形状是 `(batch_size, context_length, d_model)` 这种),所以张量里元素的总个数,自然就跟 batch_size 成正比。

**QKV 投影、加权求和(softmax(QK^T)@V 之后)、输出投影这三个——是的,总元素数上等价于 `(batch_size, context_length, d_model)`。**

- QKV 投影:Q=WQxQ=W_Qx Q=WQx、K=WKxK=W_Kx K=WKx、V=WVxV=W_Vx V=WVx,因为 h×dk=dmodelh \times d_k = d_{model} h×dk=dmodel(多头切分只是把最后一维拆开,总元素数不变),所以整体形状(还没拆分成多头之前)就是 `(batch_size, context_length, d_model)`。
- 加权求和(`softmax(QK^T/√d_k) @ V`):每个 head 算出来是 `(batch_size, num_heads, context_length, d_v)`,把所有 head 拼回去(`d_v × num_heads = d_model`)之后,总元素数等价于 `(batch_size, context_length, d_model)`。
- 输出投影 WOW_O WO 的结果:形状就是 `(batch_size, context_length, d_model)`。

这两项(QK^T、softmax)对应的元素总数,应该用 `batch_size × num_heads × context_length × context_length` 去算,而不是 `batch_size × context_length × d_model`。这一点很关键,因为 `context_length` 是平方关系,量级上通常比其他几项大不少。

回忆一下 SwiGLU 的公式:$FFN(x)=W2(SiLU(W1x)⊙W3x)\text{FFN}(x) = W_2(\text{SiLU}(W_1x) \odot W_3x) FFN(x)=W2(SiLU(W1x)⊙W3x)$,以及权重矩阵的维度:$W_1, W_3 \in \mathbb{R}^{d_{ff} \times d_{model}} W_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$。

- **W_1x**: W1 把输入从 dmodeld_{model} dmodel 维映射到 dffd_{ff} dff 维,所以结果形状是 `(batch_size, context_length, d_ff)`,**不是** `d_model`。
- **SiLU 作用在 W_1x 上**:SiLU 是逐元素的激活函数,不改变形状,所以还是 `(batch_size, context_length, d_ff)`。
- **W_3x**:同样是 dmodel→dffd_{model} \to d_{ff} dmodel→dff 的映射,形状也是 `(batch_size, context_length, d_ff)`。
- **两者逐元素相乘**(`SiLU(W1x) ⊙ W3x`):两个形状相同的张量逐元素相乘,形状不变,还是 `(batch_size, context_length, d_ff)`。
- **W_2(...) :这一步 W_2 把 dff 维**映射回d_model 维,所以**只有这一步**的输出,形状才是 `(batch_size, context_length, d_model)`。

 最后RMSNorm 是 `(batch_size, context_length, d_model)`,后两项(LM head、cross-entropy)都应该按 `(batch_size, context_length, vocab_size)` 去算元素个数



**固定部分(参数量相关)**:

- token embedding = vocab_size * d_model = 10000 * 512 = 5120000
- 每层的 W_Q,W_K,W_V,W_O WQ,WK,WV,WO = 4 * d_model * d_model =  4 * (512 * 512) = 1048576
- 每层的两个 RMSNorm = 2 * d_model = 2 * 512 = 1024
- 每层的 W_1,W_2,W_3 W1,W2,W3 = 3 * d_ff * d_model = 3 * 1344 * 512 = 2064384
- 最后的 RMSNorm = d_model = 512
- LM head = vocab_size * d_model = 10000 * 512 = 5120000

​	22.696.448 parameters

​	* 4 * 4  = 90.785.792 * 4 = 363.143.168 bytes

**可变部分(激活值相关,每单位 batch_size)**:

- 每层的 RMSNorm(2个) = batch_size * context_length *  d_model * 2  = 262144 * batch_size

- QKV 投影 = batch_size * context_length *  d_model * 3 = 393216 * batch_size

- QK⊤ = batch_size * num_heads * context_length * context_length = 1048576 * batch_size

- softmax = batch_size * num_heads * context_length * context_length = 1048576 * batch_size

- 加权求和 =  batch_size * context_length *  d_model = 131072 * batch_size

- 输出投影 =  batch_size * context_length *  d_model = 131072 * batch_size

- FFN 的 4 个 dff 大小的中间量 + 1 个 dmodel 大小的输出 = batch_size * context_length *  d_ff * 4 + batch_size * context_length *  d_model = 1507328 * batch_size

- 最后的 RMSNorm = batch_size * context_length *  d_model = 131072 * batch_size

- LM head(输出 embedding) = batch_size * context_length * vocab_size = 2560000 * batch_size

- cross-entropy = batch_size * context_length * vocab_size = 2560000 * batch_size

  23.339.008 * batch_size parameters

  