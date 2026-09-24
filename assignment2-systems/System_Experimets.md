# System section tasks

1. ### End-to-End Benchmarking 

    **(NVIDIA RTX PRO 6000, 1GPU 24vCPU 218GiB)**

   ##### b) forward / forward+backward / full 的时间对比

   ```
           forward      forward+backward   full(+optimizer)
   small     19.59ms         56.36ms           69.05ms
   medium    52.52ms        151.31ms          192.51ms
   large    117.22ms        340.63ms          433.28ms
   xl       331.24ms        910.40ms          OOM(显存不够)
   ```

   把 backward 单独算出来看(用 forward_backward 减去 forward):

   ```
           纯backward耗时    backward/forward 的倍数
   small      36.77ms              1.88x
   medium     98.79ms              1.88x
   large     223.41ms              1.91x
   xl        579.16ms              1.75x
   ```

   这里有个很规律的现象:**backward 大约是 forward 的 1.8~1.9 倍**,而且这个倍数在四个不同大小的模型上几乎一致。这不是巧合,是有道理的:forward 只需要算一遍"输入 → 输出",而 backward 需要对每一层都算一次梯度(链式法则),数学上的计算量大致是 forward 的 2 倍左右(业界常说的经验值就是 backward ≈ 2×forward),我们测出来的 1.8~1.9x 跟这个吻合。

   再看 optimizer step 的开销(full 减去 forward_backward):

   ```
           optimizer+zero_grad 耗时
   small       12.69ms
   medium      41.20ms
   large       92.65ms
   ```

   这部分也随模型变大而明显增长,因为 AdamW 要对**每一个参数**都做一次更新计算(还要维护一阶、二阶动量),参数越多这一步越慢。xl 在 full 模式下直接 OOM 了,这也是之前预期到的:AdamW 要多存两份跟参数一样大的动量,显存直接翻三倍左右,撑不住。

   ##### (c) warmup 步数的影响

   ```
   warmup=0   mean=949.24ms   std=119.05ms   ← 波动巨大
   warmup=1   mean=909.54ms   std=0.41ms
   warmup=2   mean=910.46ms   std=0.44ms
   warmup=5   mean=911.35ms   std=0.91ms
   ```

   这组数据说明得特别清楚:**只要有 1 步 warmup,数据就立刻稳定下来**(std 从 119ms 掉到 0.4ms 左右),warmup=1/2/5 之间的差别几乎可以忽略。而 warmup=0 不仅波动巨大,均值也偏高了将近 40ms。

   原因是第一次调用会触发一堆"一次性"的初始化开销:CUDA context 初始化、cuDNN/cuBLAS 挑选最优算子实现(autotuning)、显存分配器第一次申请大块显存等等——这些都只发生在第一次,后面的调用会复用。所以不热身直接测,测到的其实是"正常计算时间 + 这些一次性开销"的混合体,数据不干净;哪怕只热身 1 步,这些开销就已经消化完了。

2. ### Nsight Systems Profiling

- (a) nsys 测得 forward 单独的 NVTX 区间是 33.78ms,和 Python 标准库测的 52.52ms 对不上,原因是 CPU/GPU 异步执行、forward 和 backward 标签之间没插入 synchronize(),所以 forward 标签测到的更多是 CPU 发指令的时间而非 GPU 真正算完的时间。但整个 step 的总时长(197.26ms)和 Python 测的 full 模式(192.51ms)基本吻合(差 2.5%,是 profiling 本身的开销),说明只有插了 synchronize() 的端到端总时长才可信。

- (b) 排名第一的 kernel 是 cutlass::Kernel2<cutlass_80_simt_sgemm_128x256...>,是一个矩阵乘法(GEMM)kernel,整次运行(5次热身+5次测量)里累计耗时约 258ms、被调用 1440 次。因为 backward 里同样有大量矩阵乘法(对 Q/K/V/权重求梯度),所以 forward-only 和 forward+backward 一起看,占用时间最多的都是同一类 GEMM kernel。

- (c) 除了矩阵乘法,multi_tensor_apply_kernel(约占 6.3%,PyTorch 用来批量更新一堆参数张量,AdamW 更新参数时会用到)和 vectorized_elementwise_kernel(约占 5~6%,处理加法、激活函数这类逐元素操作)也占了不小比例。

- (d) 完整训练步(forward+backward+optimizer)相比只做 forward,矩阵乘法的时间占比会有所下降——因为 optimizer step 几乎全部由 multi_tensor_apply/elementwise 这类完全不含矩阵乘法的 kernel 组成,它拉长了总时长,但对矩阵乘法的绝对耗时没有任何贡献,相当于把矩阵乘法的占比"稀释"了。

- (e) 用之前那 3 层的数据:矩阵乘法(两次加起来)耗时始终是 softmax 的 3~4 倍左右。但矩阵乘法的 FLOPs 相对 softmax 要高出好几个数量级——实际耗时的差距远小于理论计算量的差距,说明 softmax 相对它的计算量"不成比例地慢",因为它是访存密集型操作、吃不到 Tensor Core 加速,这正是 FlashAttention 要解决的问题。