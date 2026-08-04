# KV Cache Bufferization 与 Triton Store

## 目标

Stateful Decode 阶段已经把两个 Tensor `cat` 恢复为高层状态语义：

```text
%next = serve.kv.append(%state, %current_key, %current_value)
```

但参考 Runtime 的 Append 仍调用 `torch.cat`。每个 Decode Step 都复制完整历史 Cache：

```text
单步复杂度：O(历史长度)
完整生成：  O(序列长度²)
```

Bufferization 的目标是让编译器自动把 Functional Tensor Cache Lower 成预分配 Buffer
的位置写入，而不是要求用户手工修改模型 Forward。

## 自动编译路径

当前完整路径为：

```text
torch.export Tensor Cache 图
    ↓
MaterializeKVStatePass
    ↓
serve.kv.append
    ↓
BufferizeKVCachePass
    ↓
serve.kv.length
serve.kv.store
serve.kv.advance
    ↓
预分配 B×Capacity×H×D Buffer
    ↓
Triton KV Store Kernel
```

原始 PyTorch 模型只包含普通 `torch.cat`，不需要知道预分配 Buffer、物理 Layout 或
Triton Kernel。

## Capacity 推导

Stateful 前端的动态约束为：

```text
1 <= past_sequence <= 64
2 <= past_sequence + 1 <= 65
```

`MaterializeKVStatePass` 从 Present Key 的派生符号上界自动推导：

```text
capacity = 65
```

生成的状态类型为：

```text
!serve.kv_state<
    f32,
    layers=1,
    heads=2,
    head_dim=16,
    layout=contiguous_bshd,
    resource=kv.layer0,
    capacity=65
>
```

也可以通过 `--kv-capacity` 覆盖导出上界。覆盖值适用于 Serving Runtime 已知拥有更大
物理池、但模型导出样例使用较小动态范围的情况。

## Bufferization IR

高层 Append：

```text
%next = serve.kv.append(%state, %key, %value)
```

被分解为：

```text
%positions = serve.kv.length(%state)
    effects[read(kv.layer0)]

%stored = serve.kv.store(
    %state,
    %key,
    %value,
    %positions
)
    effects[read(kv.layer0), write(kv.layer0)]

%next = serve.kv.advance(%stored) {delta=当前 K/V 的静态 Token 数}
    effects[read(kv.layer0), write(kv.layer0)]
```

`positions` 是 `[batch]` 的 i64 Tensor，因此不同 Batch Item 可以拥有不同写入位置。
当前单 Token Decode 的 `delta=1`；低层 Store 同时支持多个新 Token 和 Ragged Position。
Pass 会先验证函数中的全部 Append，只有全部满足类型、Axis、Capacity 和静态 Token 数约束
时才统一提交改写，避免留下部分 Bufferize 的混合 IR。

Pass 前后 Operation 数量：

```text
导入：             85
删除断言：         57
融合 RMSNorm：     45
状态语义恢复：     45
KV Bufferization： 47
```

## 逻辑 SSA 与物理别名

预分配状态由两部分组成：

```text
共享物理 Buffer：B×Capacity×H×D
逻辑状态版本：    Lengths + Generation
```

Store 会写入旧逻辑长度之外的位置，但不推进 Length：

```text
state0.length = 5
store(state0, position=5)
state0 仍只能读取 [0:5]
```

Advance 返回新的逻辑状态：

```text
state1.length = 6
```

因此 `state0` 和 `state1` 可以共享物理 Buffer，旧状态的可见前缀仍保持不变。这是
Bufferization 中常见的“逻辑 SSA + 物理别名”设计。读写副作用保证 Store/Advance
不能被错误重排。

## 物理 Layout

ServeIR 对 Attention 暴露的逻辑 KV Layout 仍是：

```text
B×H×S×D
```

预分配物理 Layout 选择：

```text
B×Capacity×H×D
```

它和 OmniServe/FlashAttention 使用的 Block Layout 更接近。固定 Capacity 使每个
Batch Slot 的地址稳定，也为 CUDA Graph 和 Paged KV Cache 提供基础。

## Triton KV Store

Triton Kernel 的 Grid 为：

```text
batch × num_kv_heads × num_new_tokens
```

每个 Program 负责一个：

```text
(batch, head, token)
```

并行写 Key 和 Value 的整个 Head Dim：

```text
physical_offset =
    ((batch × capacity + position) × heads + head) × head_dim
```

Kernel 支持：

- 每个 Batch 不同的 Position；
- 多个新 Token；
- 非连续输入 K/V 的真实 Stride；
- 连续 `B×Capacity×H×D` 输出；
- Capacity Mask，避免越界地址写入；
- Key 和 Value 在同一个 Kernel 中写入。

非连续 Stride 很重要。模型中的 K/V 来自 `view + transpose`，如果强制
`.contiguous()`，虽然只复制当前 Token，但会增加一个不必要的 Kernel 和显存流量。

## 正确性验证

CPU Reference Runtime 从历史长度 3 执行四轮：

```text
Length：     3 → 4 → 5 → 6 → 7
Generation：0 → 1 → 2 → 3 → 4
```

结果：

```text
Output 最大绝对误差：0
Key 最大绝对误差：   0
Value 最大绝对误差： 0
```

四轮前后 Key/Value Buffer 的 `data_ptr` 完全相同，证明没有随着 Sequence 增长重新
分配物理 Cache。

GPU 联合测试包含：

```text
Bufferized Stateful Decode
+ Triton KV Store
+ Triton RMSNorm
+ 完整 Attention/MLP
```

两轮 Decode 后输出和原始 `torch.export` CUDA 图一致，并验证每 Batch 不同 Position
的 Triton 写入结果。

## RTX 4060 性能

配置：

```text
GPU：NVIDIA GeForce RTX 4060 Laptop
DType：FP16
KV Heads：2
Head Dim：128
Warmup：20 ms
Rep：80 ms
```

| Batch | 历史长度 | torch.cat | index_copy_ | Triton Store | 相比 cat |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 10.33 μs | 17.81 μs | 4.75 μs | 2.17× |
| 1 | 256 | 11.55 μs | 15.71 μs | 4.23 μs | 2.73× |
| 1 | 1024 | 17.16 μs | 15.52 μs | 4.46 μs | 3.85× |
| 8 | 64 | 12.58 μs | 16.63 μs | 4.62 μs | 2.72× |
| 8 | 256 | 27.20 μs | 16.61 μs | 4.46 μs | 6.10× |
| 8 | 1024 | 90.69 μs | 16.74 μs | 4.43 μs | 20.47× |
| 32 | 64 | 28.08 μs | 16.67 μs | 5.30 μs | 5.30× |
| 32 | 256 | 91.35 μs | 16.84 μs | 5.11 μs | 17.87× |
| 32 | 1024 | 323.90 μs | 17.48 μs | 5.44 μs | 59.56× |

九组配置的几何平均：

```text
Triton vs torch.cat：   7.064×
Triton vs index_copy_： 3.514×
最大绝对误差：          0
```

Triton 延迟保持在 `4.23～5.44 μs`，而 Cat 随 Batch 和历史长度从 `10.33 μs`
增长到 `323.90 μs`。这验证了复杂度变化：

```text
Cat：   复制历史，流量随 Length 增长
Store：只写当前 Token，流量与历史 Length 无关
```

## 使用方法

生成 Bufferized ServeIR：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.optimizer \
  artifacts/stateful_decode.pt2 \
  --preallocate-kv \
  --out artifacts/stateful_decode.bufferized.serveir \
  --stats-out artifacts/stateful_bufferize_stats.json
```

执行多轮差分：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.stateful_check \
  artifacts/stateful_decode.pt2 \
  --batch 3 \
  --past-length 5 \
  --steps 4 \
  --preallocate-kv \
  --out artifacts/stateful_bufferized_differential.json
```

运行 GPU Benchmark：

```bash
PYTHONPATH=src python benchmarks/bench_kv_store.py \
  --batches 1,8,32 \
  --lengths 64,256,1024 \
  --out artifacts/kv_store_benchmark.json
```

## 已完成的后续衔接

`serve.kv.read → repeat_interleave → matmul → softmax → matmul` 已进一步融合为
`serve.decode_attention`。新操作直接接收状态、Query 和 Mask，Triton Lowering
直接读取物理 Buffer 和设备 Length，不再通过 `read()` 构造逻辑 KV View。

具体设计、正确性和 GPU 结果见 `docs/decode-attention.md`。

## 当前边界

- 已能事务式 Bufferize 多个连续 Decoder Layer Slot；任一 Append 不满足契约时，
  整个函数都不会留下部分 Lower 的混合状态；
- 非 Bufferized Stateful 路径仍保留 `serve.kv.read` 作为清晰的参考语义；
- Bufferized Decode 已由 `serve.decode_attention` 直接消费 Buffer、Lengths 和 Slot；
- Store 已支持每 Batch 不同 Position，但当前前端 Mask Fixture 仍使用统一 Past Length；
- Capacity 来自导出上界或命令行，还没有独立的全局显存规划器；
- 尚未实现 KV Buffer 分配/释放 Operation；
- 尚未实现 Paged Layout 和 Block Table；
- Benchmark 只测 Cache 更新，不包含完整 Attention 和端到端 Token 延迟。

下一阶段应把连续 Buffer 扩展为 Paged Layout，并让 Attention 接收 Block Table，
同时增加长上下文 Split-Sequence 和 Profile-Guided Kernel 配置。
