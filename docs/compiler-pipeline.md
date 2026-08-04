# 对标 `torch.compile` 的编译流水线

## 项目目标

本项目的目标不是只实现若干独立 Triton Kernel，而是实现一条可验证的整图编译链：

```text
PyTorch Module
    ↓ torch.export：捕获带动态 Shape 的 Functional ATen 图
ServeIR
    ↓ 图级分析与优化：融合、状态化、Bufferization、布局规划
优化后的 ServeIR
    ↓ Lowering：为每个高层操作选择并生成后端实现
KernelIR
    ↓ Triton / cuBLAS / Runtime
GPU 结果
```

这里与 `torch.compile` 的对应关系是：

| 本项目 | `torch.compile` 体系中的近似角色 |
| --- | --- |
| `torch.export` 前端 | Dynamo/AOTAutograd 后的可编译图输入 |
| ServeIR | FX/ATen 之上的 Serving 专用编译 IR |
| ServeIR Pass | Inductor 的图优化、融合和内存规划阶段 |
| KernelIR | Inductor Scheduler IR / Triton Lowering 边界 |
| Triton Kernel | Inductor 生成的 GPU Kernel |
| Stateful KV Runtime | 面向 LLM Serving 的专用运行时能力 |

本项目不会复刻 `torch.compile` 的所有通用训练能力。研究重点是动态、有状态的 LLM
推理，特别是 Prefill、Decode、KV Cache、GQA 和 Paged Attention。

## 为什么需要 KernelIR 边界

此前 `TritonExecutor` 继承参考执行器：RMSNorm、KV Store 和 Decode Attention 使用
Triton，其余 `aten.*` 节点直接由 PyTorch 执行。它适合先验证优化正确性，但不能证明
整张图已被编译。

现在已支持以下显式 Lowering：

```text
serve.rms_norm         -> kernel.triton.rms_norm
serve.linear           -> kernel.triton.linear / kernel.cublas.linear
serve.rope             -> kernel.triton.rope
serve.prefill_attention -> kernel.triton.prefill_attention
serve.kv.store         -> kernel.triton.kv_store
serve.kv.prefill_store -> kernel.triton.kv_prefill_store
serve.kv.init          -> runtime.kv.init
serve.decode_attention -> kernel.triton.decode_attention
serve.kv.length        -> runtime.kv.length
serve.kv.advance       -> runtime.kv.advance
```

`kernel.triton.*`和`kernel.cublas.*`是GPU计算，`runtime.*`是KV状态和元数据操作。
Linear会根据静态矩阵形状选择后端：当前任一特征维达到4096时选择
`kernel.cublas.linear`，其余选择`kernel.triton.linear`。前者由执行器显式调用
PyTorch的`F.linear`进入CUDA库路径，是cuBLAS Dialect原型，尚未直接绑定底层cuBLAS
API。View/Reshape等纯元数据操作后续可Lower为`kernel.metadata.*`。这些都属于编译
后端选择，不应与未Lower ATen节点的兼容回退混为一谈。

## 覆盖率与严格模式

`LoweringCoverage` 对整张 KernelIR 统计：

- `lowered_operations`：已进入 `kernel.*` 或 `runtime.*` 的节点；
- `unlowered_operations`：仍停留在 `aten.*`、`serve.*` 等高层 Dialect 的节点；
- `coverage`：已 Lower 节点数除以总节点数；
- `unlowered_by_name`：每种缺失 Lowering 的准确数量。

兼容模式允许产生“部分 Lower”的 IR，方便逐个实现后端。严格模式要求
`unlowered_operations == 0`，否则编译或执行会在运行任何算子前失败。这样不会把
PyTorch fallback 误报成编译器能力。

生成 KernelIR 和覆盖报告：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.optimizer \
  artifacts/stateful_decode.pt2 \
  --preallocate-kv \
  --lower-kernel-ir \
  --out artifacts/stateful_decode.kernelir \
  --stats-out artifacts/stateful_decode.compile_stats.json
```

检查零回退；当前完整 Qwen2 图仍会失败，并列出需要继续实现的操作：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.optimizer \
  artifacts/stateful_decode.pt2 \
  --preallocate-kv \
  --require-full-lowering \
  --out artifacts/stateful_decode.kernelir
```

Python 入口：

```python
from stateful_llm_compiler import CompileOptions, compile_exported_program

artifact = compile_exported_program(
    exported_program,
    options=CompileOptions(
        preallocate_kv=True,
        require_full_lowering=False,
    ),
)
print(artifact.coverage.to_dict())
```

执行 KernelIR 时也可再次启用严格检查：

```python
from stateful_llm_compiler.backends import TritonExecutor

result = TritonExecutor(strict=True).run(
    artifact.module,
    runtime_arguments,
)
```

## 后续完整 Prefill 的工作顺序

1. 把Embedding、SwiGLU和RoPE系数生成Lower到Triton或Runtime；
2. 把剩余View、Transpose、Reshape和Cast规范为元数据操作；
3. 对已捕获的`input_ids→logits`整图逐步清除全部回退；
4. 让完整图在 strict 模式下达到零回退；
5. 再引入 Paged KV、Block Table 和长序列切分优化。

因此，当前阶段的价值不是“已经完成完整编译”，而是建立了以后衡量完整编译的硬性
标准。每补齐一种 Lowering，覆盖报告都会真实减少对应缺口，直到严格模式通过。
