# 多Token GQA Prefill Attention融合与Triton Lowering

## 图级融合

Prefill导出图原本为：

```text
key.repeat_interleave(groups)
value.repeat_interleave(groups)
key.transpose(-2, -1)
score = query @ keyᵀ
scaled = score * scale
masked = scaled.float() + additive_mask.float()
probability = softmax(masked).to(query.dtype)
context = probability @ value
```

`FusePrefillAttentionPass`验证非交换Matmul的Operand顺序、GQA重复维度、正数Scale、
Softmax轴、Q/K/V/Mask Shape以及中间值没有逃逸，然后替换为：

```text
serve.prefill_attention(query, key, value, mask) {
    groups = 2,
    scale = 0.353553,
    causal = "mask",
    algorithm = "online_softmax"
}
```

两层Qwen2中，每层10个展开操作被替换为1个领域操作，总操作数从128降到110。

## Causal语义为什么记录为Mask

当前前端接收四维加性Causal Mask，但仅凭Shape无法证明任意运行时Mask一定是下三角。
如果融合Pass看到方阵就让Kernel强制下三角，会错误改变全零双向Mask的语义。

因此当前契约是：

```text
causal = "mask"
```

Kernel完整加载Mask；Qwen输入用负无穷屏蔽未来位置。测试同时覆盖全零Mask，确保融合
不会私自增加Causal限制。未来如果Causal Mask由编译器内部生成，可以使用
`causal=true`并省略完整Mask读取。

## Triton Online Softmax

Grid为：

```text
axis 0：Query Token Block
axis 1：Batch × Query Head
```

每个Program加载`BLOCK_M×D`的Query Tile，随后用`BLOCK_N`遍历Key/Value：

```text
score_block = dot(query_tile, key_tileᵀ) * scale + mask
next_max = max(running_max, max(score_block))
probability = exp(score_block - next_max)
running_sum = running_sum * alpha + sum(probability)
accumulator = accumulator * alpha + dot(probability, value_tile)
```

最终：

```text
context = accumulator / running_sum
```

这避免了：

- 物化GQA重复后的K/V；
- 写出`B×H×T×T`完整Score；
- 写出完整Softmax概率矩阵。

当前支持：

- FP16、BF16、FP32；
- 动态Batch和Prompt长度；
- GQA；
- 任意四维加性Mask；
- Head Dim连续但其他Stride可以非连续；
- FP32 `tl.dot`使用IEEE输入精度。

当前Prompt长度作为Triton编译期特化参数，不同T可能产生不同Kernel版本。后续要增加
Shape Bucket，避免为每个Prompt长度重新JIT。

## 编译覆盖率

两层Qwen2ForCausalLM Prefill：

```text
融合前：128个操作，已Lower 20个，覆盖率15.625%
融合后：110个操作，已Lower 22个，覆盖率20.000%
```

已Lower：

```text
kernel.triton.linear             15
kernel.triton.rms_norm            5
kernel.triton.prefill_attention   2
```

## RTX 4060 Laptop性能

配置为FP16、4个Query Head、2个KV Head、Head Dim 64：

| B | T | 展开GQA | PyTorch SDPA GQA | Triton | 对展开加速 | 对SDPA加速 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 28.94 μs | 48.09 μs | 7.89 μs | 3.67× | 6.10× |
| 1 | 64 | 24.67 μs | 52.93 μs | 7.51 μs | 3.28× | 7.05× |
| 1 | 128 | 30.03 μs | 64.14 μs | 10.67 μs | 2.81× | 6.01× |
| 2 | 16 | 23.23 μs | 38.80 μs | 6.59 μs | 3.53× | 5.89× |
| 2 | 64 | 27.20 μs | 56.03 μs | 8.18 μs | 3.33× | 6.85× |
| 2 | 128 | 35.21 μs | 72.14 μs | 11.22 μs | 3.14× | 6.43× |

六组几何平均：

```text
Triton相对展开GQA：3.281×
Triton相对SDPA GQA：6.372×
```

这里的SDPA结果只代表当前GPU、PyTorch版本和`enable_gqa`路径，不能泛化为所有
FlashAttention实现。真正有价值的结论是：直接在Kernel中映射Query Head到KV Head，
可以避免当前基线的GQA物化开销。

复现实验：

```bash
PYTHONPATH=src python benchmarks/bench_prefill_attention.py \
  --batches 1,2 \
  --tokens 16,64,128 \
  --query-heads 4 \
  --kv-heads 2 \
  --head-dim 64 \
  --out artifacts/prefill_attention_benchmark_v1.json
```

## 下一步

当前Prefill返回逻辑`B×KVH×T×D` Tensor Tuple，Decode使用物理
`PreallocatedKVCacheState`。下一阶段要加入Prefill状态化与批量KV Store，使两个阶段
共享同一个Buffer、Length和Layer Slot ABI，消除阶段切换时的Cache复制。
