# 直接消费物理 KV Buffer 的 Decode Attention

## 动机

KV Bufferization 已经把增长式：

```text
torch.cat(past, current)
```

改成了预分配 Buffer 上的位置写入。但旧 Attention 仍然经过：

```text
serve.kv.read
  → repeat_interleave(K/V)
  → QK Matmul
  → Softmax
  → PV Matmul
```

这有三个问题：

1. `kv.read` 根据动态 Length 创建逻辑 B×H×S×D View，CUDA 参考路径需要把最大
   Length 同步回 CPU；
2. GQA 的 `repeat_interleave` 会把 KV Head 扩展到 Query Head，产生额外 Tensor；
3. Score、Probability 和展开后的 K/V 都作为独立 Tensor 存在，增加 Kernel Launch
   和显存流量。

## IR 融合

`FuseDecodeAttentionPass` 对 Use-Def 链做保守匹配。只有完整子图没有中间结果逃逸、
GQA Group 一致、Softmax 位于最后一维，并且 KV Layout 已经是
`contiguous_bshd` 时才融合。当前 Triton ABI 还要求 Query 为 `B×QH×1×D`、Mask 为
`B×1×1×S`；两个 Matmul 必须严格保持 `Query @ Keyᵀ` 和 `Probability @ Value` 的
非交换顺序，Operand 与 FX Attribute 参数树也必须一致。

融合前：

```text
%key, %value = serve.kv.read(%state)
%key_gqa = aten.repeat_interleave(%key)
%value_gqa = aten.repeat_interleave(%value)
%scores = aten.matmul(%query, transpose(%key_gqa))
%scaled = aten.div(%scores)
%masked = aten.add(%scaled, %mask)
%probability = aten.softmax(%masked)
%context = aten.matmul(%probability, %value_gqa)
```

融合后：

```text
%context = serve.decode_attention(
    %state,
    %query,
    %mask
) {
    slot = 0,
    groups = 2,
    scale = 0.353553,
    layout = "contiguous_bshd",
    algorithm = "online_softmax"
} effects[read(kv.layer0)]
```

该操作保留了以下高层信息：

- Attention 正在读取哪个 KV Slot；
- Query Head 与 KV Head 的 GQA 映射；
- 缩放因子；
- KV 物理 Layout；
- 对 KV 资源的只读副作用。

这些信息如果过早 Lower 成普通 Matmul 和 Load，很难再由底层编译器恢复。

## Triton Lowering

Triton Kernel 采用一个 Program 对应一个 `(batch, query_head)`：

1. 根据 `query_head // groups` 计算对应的 KV Head，不执行
   `repeat_interleave`；
2. 按 32 Token 分块，从 B×Capacity×KVHead×HeadDim Buffer 直接读取 K/V；
3. 使用设备上的 `lengths[batch]` 屏蔽无效位置，不读取 Length 到 CPU；
4. 在每一块内计算 QK；
5. 使用 Online Softmax 维护 Running Max、Running Sum 和 Context Accumulator；
6. 只把最终 Context 写回显存。

Online Softmax 的更新为：

```text
m_new = max(m_old, max(scores_block))
alpha = exp(m_old - m_new)
l_new = alpha * l_old + sum(exp(scores_block - m_new))
acc_new = alpha * acc_old + Σ(exp(scores_block - m_new) * value_block)
```

最终：

```text
context = acc / l
```

因此临时状态只与 Token Block 和 Head Dim 有关，不需要分配完整
`B×QueryHead×Sequence` Score/Probability Tensor。

## 正确性

当前验证覆盖：

- CPU/PyTorch 参考执行器与原始 `ExportedProgram` 多轮 Decode 差分；
- Triton 端到端 Bufferized Decode；
- 每 Batch 不同有效长度 `[7, 31, 65]`；
- GQA Query Head 到 KV Head 映射；
- KV Buffer 地址在多轮 Decode 中保持不变；
- IR 副作用、类型和 SSA 支配关系校验。

运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## RTX 4060 Laptop GPU 实验

实验配置：

```text
dtype = fp16
query_heads = 4
kv_heads = 2
head_dim = 128
batch = 1, 8
length = 64, 256, 1024
```

结果保存在 `artifacts/decode_attention_benchmark.json`。

| Batch | Length | 展开 Eager | PyTorch SDPA | Triton | 对展开路径 |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 32.37 μs | 24.31 μs | 14.18 μs | 2.28× |
| 1 | 256 | 31.83 μs | 30.29 μs | 26.98 μs | 1.18× |
| 1 | 1024 | 52.45 μs | 81.19 μs | 87.30 μs | 0.60× |
| 8 | 64 | 41.09 μs | 38.36 μs | 16.74 μs | 2.45× |
| 8 | 256 | 77.99 μs | 83.58 μs | 38.94 μs | 2.00× |
| 8 | 1024 | 195.63 μs | 262.30 μs | 114.56 μs | 1.71× |

六组配置几何平均：

```text
Triton vs 展开 Eager：1.545×
Triton vs PyTorch SDPA：1.650×
最大绝对误差：0.001220703125
```

结果并非所有 Shape 都占优：`B=1, L=1024` 时 Triton 比展开 Eager 慢。这说明
当前每个 Query Head 独立读取 K/V，在小 Batch、长序列下并行度不足且重复读取较多。
下一轮需要研究：

- 多 Query Head 共享同一 KV Head 的数据加载；
- 根据 Length 选择 `block_tokens` 和 Warp 数量；
- 长上下文使用 Split-K/Split-Sequence 后归并 Online Softmax 状态；
- 将 KV Store 和 Attention 的调度边界纳入成本模型；
- 与 FlashAttention/Paged Attention ABI 做更公平的 Serving 对比。

## 当前边界

- 只支持单 Token Query；
- 只支持连续 `B×Capacity×KVHead×HeadDim` Layout；
- 尚未支持 Paged KV、Block Table 和非连续请求页；
- 尚未实现多层 Slot 自动规划；
- Kernel 参数仍是固定 `block_tokens=32`，没有 Profile-Guided 调度；
- 当前实验是单算子延迟，不是完整模型的端到端 Token/s。
