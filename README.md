# StatefulLLM-Compiler

面向动态、有状态 LLM Serving 工作负载的研究型编译器。

当前已经完成九个里程碑：

1. 使用 `torch.export` 捕获带动态 Batch/序列长度的 Qwen 风格 Decoder；
2. 将 Functional ATen 图导入自研 ServeIR，显式建模 SSA、动态类型和 KV 副作用；
3. 实现 Use-Def、PassManager、导出断言清理和 RMSNorm 融合；
4. 实现 ServeIR 参考执行器和动态 Shape 数值差分验证；
5. 将 `serve.rms_norm` Lower 到 Triton 并完成 GPU 性能实验；
6. 基于目标 GPU Profile 生成动态 Shape 多版本 Lowering 计划并在运行时分派；
7. 导出 Stateful 单 Token Decode，把 Tensor KV Cache 改写为显式状态并完成
   CPU/GPU 多轮差分验证；
8. 自动把逻辑 KV Append Bufferize 为预分配位置写入，并 Lower 到 Triton KV Store；
9. 把 GQA Decode Attention 融合为 `serve.decode_attention`，通过 Online Softmax
   Triton Kernel 直接消费物理 KV Buffer 和设备 Length。

项目仍保持 CPU 可运行；GPU Profile、Triton Kernel 和运行时分派是可选能力。

## 运行前端导出

使用包含 PyTorch 2.8 或更新版本的环境：

```bash
cd stateful-llm-compiler
PYTHONPATH=src python -m stateful_llm_compiler.frontend \
  --out artifacts/decoder_graph.json \
  --graph-out artifacts/decoder_graph.txt \
  --program-out artifacts/decoder.pt2
```

将 `.pt2` 导入 ServeIR：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.importer \
  artifacts/decoder.pt2 \
  --out artifacts/decoder.serveir
```

运行默认优化流水线：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.optimizer \
  artifacts/decoder.pt2 \
  --before-out artifacts/decoder.before.serveir \
  --out artifacts/decoder.optimized.serveir \
  --stats-out artifacts/optimization_stats.json
```

使用 GPU Profile 运行 Profile-Guided Lowering：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.optimizer \
  artifacts/decoder.pt2 \
  --profile artifacts/rmsnorm_benchmark_v1.json \
  --out artifacts/decoder.profile.optimized.serveir \
  --stats-out artifacts/profile_optimization_stats.json
```

导出带动态 KV Cache 的单 Token Decode：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.stateful_frontend \
  --out artifacts/stateful_decode.json \
  --graph-out artifacts/stateful_decode.txt \
  --program-out artifacts/stateful_decode.pt2 \
  --example-past-length 4 \
  --max-cache-length 64
```

把 Tensor KV Cache 改写为显式 ServeIR 状态：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.optimizer \
  artifacts/stateful_decode.pt2 \
  --stateful-decode \
  --out artifacts/stateful_decode.optimized.serveir \
  --stats-out artifacts/stateful_decode_stats.json
```

运行四轮 Stateful Decode 差分验证：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.stateful_check \
  artifacts/stateful_decode.pt2 \
  --batch 3 \
  --past-length 5 \
  --steps 4 \
  --out artifacts/stateful_differential_results.json
```

生成预分配 KV Buffer IR 并验证：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.optimizer \
  artifacts/stateful_decode.pt2 \
  --preallocate-kv \
  --out artifacts/stateful_decode.bufferized.serveir \
  --stats-out artifacts/stateful_bufferize_stats.json

PYTHONPATH=src python -m stateful_llm_compiler.stateful_check \
  artifacts/stateful_decode.pt2 \
  --batch 3 \
  --past-length 5 \
  --steps 4 \
  --preallocate-kv \
  --out artifacts/stateful_bufferized_differential.json
```

验证优化后 ServeIR 与原始程序数值等价：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.reference_check \
  artifacts/decoder.pt2 \
  --shapes 1x1,2x8,3x13,4x17,8x32 \
  --out artifacts/differential_results.json
```

运行 RMSNorm GPU 基准：

```bash
PYTHONPATH=src python benchmarks/bench_rmsnorm.py \
  --rows 1,8,32,128 \
  --hidden-sizes 64,1536 \
  --dtypes fp16,fp32 \
  --inductor \
  --out artifacts/rmsnorm_benchmark.json
```

运行 Decode Attention GPU 基准：

```bash
PYTHONPATH=src python benchmarks/bench_decode_attention.py \
  --batches 1,8 \
  --lengths 64,256,1024 \
  --out artifacts/decode_attention_benchmark.json
```

运行测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

导出程序的输入契约：

```text
hidden_states:  [batch, sequence, hidden_size]
attention_mask: [batch, 1, sequence, sequence]

1 <= batch <= 8
1 <= sequence <= 128
```

两个维度都是符号维度。`attention_mask` 的三个动态轴复用同一个 `batch/sequence`
符号，因此导出程序保留了它们之间的相等约束。

设计文档：

- `docs/frontend-contract.md`：前端输入和正确性契约；
- `docs/serveir.md`：ServeIR、SSA 和 KV 副作用设计。
- `docs/passes.md`：Use-Def、PassManager 和 RMSNorm 融合设计。
- `docs/reference-execution.md`：参考执行器、Shape Guard 和数值差分。
- `docs/triton-lowering.md`：Triton Lowering、调度实验和 GPU 性能结果。
- `docs/profile-guided-lowering.md`：Target Profile、成本模型、动态分桶与运行时分派。
- `docs/stateful-decode.md`：Stateful Decode、KV 状态改写、副作用和多轮验证。
- `docs/kv-bufferization.md`：KV Bufferization、物理 Layout、Triton Store 和性能结果。
- `docs/decode-attention.md`：Attention IR 融合、Online Softmax Triton Lowering 和
  GPU 性能结果。

## 当前边界

在当前 PyTorch 2.8 环境中，ServeIR 能无 fallback 地导入 76 个 Decoder 操作。
默认优化流水线删除 14 个
导出期断言并融合两个 RMSNorm，将 IR 降至 44 个 Operation。参考执行器已在 FP32/FP16
和多组动态 Shape 下验证优化前后误差为 0。`serve.rms_norm` 已支持 Triton、Inductor
和 PyTorch Native 多后端选择；本机 Profile 证明最优后端会随 Shape 改变。单 Token
Decode 已支持动态历史长度，并将 Tensor Cache 改写为带读写副作用的显式 KV 状态。
KV Append 已能自动 Bufferize 为预分配位置写入；RTX 4060 上 Triton Store 相比
`torch.cat` 九组配置几何平均加速 7.064×。Decode Attention 已直接消费物理 KV
Buffer，六组配置相比原展开路径几何平均加速 1.545×；当前仍只支持单 Token、连续
Layout，Paged KV Cache、Block Table 和长上下文 Split-Sequence 属于后续阶段。
