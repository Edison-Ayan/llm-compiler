# Stateful Decode 与 KV Cache

## 目标

前六个阶段已经建立动态模型导出、ServeIR、图优化、参考执行、Triton Lowering 和
Profile-Guided Selection，但 Decoder 仍然是无状态函数：

```text
hidden_states + attention_mask → hidden_states
```

第七阶段引入跨推理步骤存活的 KV Cache，把 Decode 改成：

```text
hidden_states + attention_mask + kv_state
    → output + next_kv_state
```

这一步的重点是状态语义和正确性闭环。后续的 KV Bufferization 已在
`docs/kv-bufferization.md` 中实现。

## PyTorch 前端契约

`StatefulTinyDecoderBlock` 使用普通 Tensor 表达前端边界：

```text
hidden_states : [batch, 1, hidden_size]
attention_mask: [batch, 1, 1, past_sequence + 1]
past_key      : [batch, num_kv_heads, past_sequence, head_dim]
past_value    : [batch, num_kv_heads, past_sequence, head_dim]
```

输出：

```text
output       : [batch, 1, hidden_size]
present_key  : [batch, num_kv_heads, past_sequence + 1, head_dim]
present_value: [batch, num_kv_heads, past_sequence + 1, head_dim]
```

Batch 和历史 Cache 长度是动态符号，当前 Token 数固定为 1。Mask 的最后一维使用
`past_sequence + 1` 派生符号，保留 Cache 长度和 Attention Key 长度之间的关系。

前端 Cache 保留 GQA 的原始 KV Head 数量。只有执行 Attention 前，才通过
`repeat_interleave` 扩展到 Query Head 数量，避免在 Cache 中重复存储相同数据。

## 增量 Decode 正确性

为了证明 Cache 语义正确，测试使用相同权重比较：

```text
一次执行完整因果 Attention
             vs
逐 Token 执行 Stateful Decode 并传递 KV Cache
```

6 Token、Batch 2 的 FP32 测试最大绝对误差为：

```text
4.76837158203125e-07
```

这说明逐 Token Cache 路径和完整因果计算保持数值一致。

## Tensor Cache 到状态 IR

PyTorch 导出图里，Cache 追加表现为两个普通操作：

```text
present_key   = aten.cat(past_key, current_key, axis=2)
present_value = aten.cat(past_value, current_value, axis=2)
```

`MaterializeKVStatePass` 联合匹配两个 `cat`，并完成以下签名转换：

```text
转换前：
func decode(..., past_key: tensor, past_value: tensor)
    -> (output, present_key, present_value)

转换后：
func decode(..., kv_state: !serve.kv_state)
    -> (output, next_kv_state)
```

计算图中的两个 `cat` 被替换为：

```text
next_state = serve.kv.append(
    state,
    current_key,
    current_value
) effects[read(kv.layer0), write(kv.layer0)]

key, value = serve.kv.read(
    next_state
) effects[read(kv.layer0)]
```

Pass 只有同时满足以下条件时才改写：

- 同时存在 `past_key` 和 `past_value`；
- 两者均为四维 Tensor；
- KV Head 数量和 Head Dim 是静态维度；
- 每个历史 Tensor 只被唯一的 `aten.cat` 使用；
- 每个 `cat` 恰好包含一个历史值和一个当前值。

无法证明这些条件时，Pass 拒绝改写，避免错误地引入状态语义。

## SSA 与副作用

`KVCacheState` 在 Python Runtime 中是不可变对象。`append` 不修改旧对象，而是返回：

```text
generation = old_generation + 1
```

这与 ServeIR 的 SSA 表达一致：

```text
%state1 = serve.kv.append(%state0, ...)
%state2 = serve.kv.append(%state1, ...)
```

与此同时，Operation 仍显式声明底层逻辑资源 `kv.layer0` 的读写副作用。后续 Pass
不能把两次 Append 交换，也不能把 Read 移动到它依赖的 Append 之前。

## 参考执行器

参考执行器新增：

- `KVCacheState.from_tensors`：从前端 Tensor Cache 构造状态；
- `serve.kv.append`：检查 Batch、KV Heads、Head Dim、DType 和 Device 后追加；
- `serve.kv.read`：按 Layer Slot 读取 K/V；
- KV Runtime 类型检查；
- `bind_stateful_decode_arguments`：把两个前端 Tensor 参数绑定为一个状态参数。

多轮差分测试从 Cache 长度 3 开始：

```text
Step 0: 3 → 4，generation 0 → 1
Step 1: 4 → 5，generation 1 → 2
Step 2: 5 → 6，generation 2 → 3
Step 3: 6 → 7，generation 3 → 4
```

四轮的 Output、Key Cache、Value Cache 最大绝对误差均为 0，旧 Cache 的长度仍为
3，证明 Append 没有破坏旧 SSA 状态。

## GPU 联合验证

Stateful ServeIR 也通过了 GPU 联合测试：

```text
动态 Stateful Decode
    + serve.kv.append/read
    + 两个 serve.rms_norm
    + Triton RMSNorm Lowering
    + 完整 Attention/MLP 图
```

两轮 Decode 后 Cache 长度从 4 增长到 6，状态版本号为 2；四次 RMSNorm 均实际选择
Triton，完整输出与 CUDA 上的 `torch.export` 程序满足 `3e-5` 误差要求。

## 使用方法

导出 Stateful Decode：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.stateful_frontend \
  --out artifacts/stateful_decode.json \
  --graph-out artifacts/stateful_decode.txt \
  --program-out artifacts/stateful_decode.pt2 \
  --example-past-length 4 \
  --max-cache-length 64
```

生成带显式 KV 状态的 ServeIR：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.optimizer \
  artifacts/stateful_decode.pt2 \
  --stateful-decode \
  --before-out artifacts/stateful_decode.before.serveir \
  --out artifacts/stateful_decode.optimized.serveir \
  --stats-out artifacts/stateful_decode_stats.json
```

执行多轮状态差分：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.stateful_check \
  artifacts/stateful_decode.pt2 \
  --batch 3 \
  --past-length 5 \
  --steps 4 \
  --out artifacts/stateful_differential_results.json
```

## 当前边界

- 当前只有一个 Decoder Layer 和一个 KV Slot；
- 高层 Reference Runtime 的 Append 仍保留 `torch.cat` 作为语义基线；
- `BufferizeKVCachePass` 已能生成预分配连续 Buffer 和 Triton 位置写入；
- 还没有 Paged KV Cache、Block Table、内存分配和回收；
- 还没有 RoPE，因此 Cache 中保存的是未旋转模型产生的简单 K/V；
- 只导出了单 Token Decode，Prefill 仍使用原来的无状态完整序列路径；
- KV Store 已有 Triton Kernel，但 Attention 尚未直接消费物理 Buffer；
- 尚未实现多层状态的别名分析和跨 Layer 内存规划。

下一阶段应该把 Attention Lower 成直接读取预分配 Buffer 和 Lengths，再扩展为分页
Cache，消除逻辑 KV Tensor View 和普通 Matmul 路径。
