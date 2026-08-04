# StatefulLLM-Compiler

面向动态、有状态 LLM Serving 工作负载的研究型编译器。

当前已经完成十九个里程碑：

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
   Triton Kernel 直接消费物理 KV Buffer 和设备 Length；
10. 导出 Hugging Face 风格的嵌套多层 KV Cache，自动合并为一个多 Slot 状态，并
    完成两层 Decoder 的 Bufferization、Attention 融合和 CPU/GPU 端到端差分；
11. 对齐 Hugging Face Qwen2 的独立 Q/K/V、默认 RoPE、SwiGLU 和多层 Cache，完成
    官方随机权重逐层零误差对照及 ServeIR/Triton GPU 差分；
12. 建立 ServeIR→KernelIR 后端边界、统一编译入口、Lowering 覆盖报告和严格零
    PyTorch 回退模式，开始从“部分 Triton 加速”进入“整图编译”阶段；
13. 把 Qwen2 的全部 Linear 规范化为 `serve.linear`，Lower 到支持动态 M 和可选
    Bias 的 Triton tiled GEMM，使两层 Decode 后端覆盖率从11.71%提升到24.32%；
14. 增加完整Qwen2ForCausalLM边界，捕获动态Input IDs、Embedding、Prefill、LM Head、
    Logits和KV整图，并验证Prefill Cache可零误差衔接下一次Decode；
15. 将多Token GQA Attention融合为`serve.prefill_attention`并Lower到Triton Online
    Softmax Kernel，Prefill图从128降至110个操作，实测相比展开GQA几何平均加速3.281×；
16. 将Prefill多层Tensor Cache物化为预分配状态并用Triton批量写入，完整CausalLM
    Decode可在相同Buffer地址上从Length=T继续追加，实现阶段间零历史Cache复制；
17. 将每层16个Qwen2 RoPE展开节点融合为双结果`serve.rope`并Lower到单Launch Triton
    Kernel，按Decode/Prefill选择调度变体，相比TorchInductor几何平均加速1.247×；
18. 加载真实Qwen2-0.5B BF16 Checkpoint，在24层、4.94亿参数上完成官方模型零误差
    权重转换，以及状态化Prefill到两步连续Decode；引入Triton/cuBLAS显式Linear后端选择，
    并记录整图覆盖率、数值误差、执行时间和显存峰值；
19. 增加`fast`与`pytorch_compatible`双数值模式，在KernelIR中显式区分融合Triton和
    保留BF16舍入边界的CUDA复合后端；兼容模式在真实24层Prefill及两步Decode上实现
    Logits和全部KV Cache逐元素零误差。

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

继续 Lower 到 KernelIR 并输出后端覆盖报告：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.optimizer \
  artifacts/decoder.pt2 \
  --lower-kernel-ir \
  --out artifacts/decoder.kernelir \
  --stats-out artifacts/decoder_compile_stats.json
```

`--require-full-lowering` 会启用零回退检查。当前完整 Decoder 图仍会报告未实现的
ATen Lowering，这是后续补齐Embedding、SwiGLU、RoPE系数生成和元数据操作的验收标准。

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
  --num-layers 2 \
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

运行 Linear GPU 基准：

```bash
PYTHONPATH=src python benchmarks/bench_linear.py \
  --rows 1,8,32 \
  --shapes 512x512,1536x1536 \
  --bias \
  --out artifacts/linear_benchmark_v1.json
```

运行 Prefill Attention GPU基准：

```bash
PYTHONPATH=src python benchmarks/bench_prefill_attention.py \
  --batches 1,2 \
  --tokens 16,64,128 \
  --query-heads 4 \
  --kv-heads 2 \
  --head-dim 64 \
    --out artifacts/prefill_attention_benchmark_v1.json
```

运行RoPE与TorchInductor对比基准：

```bash
PYTHONPATH=src python benchmarks/bench_rope.py \
  --batches 1,8 \
  --tokens 1,128,512 \
  --query-heads 14 \
  --kv-heads 2 \
  --head-dim 64 \
  --out artifacts/rope_benchmark.json
```

验证真实Qwen2-0.5B Checkpoint的转换、编译和连续推理：

```bash
PYTHONPATH=src python benchmarks/validate_qwen2_checkpoint.py \
  --local-files-only \
  --numerical-mode pytorch_compatible \
  --decode-steps 2 \
  --out artifacts/qwen2_0_5b_compatible_validation.json
```

首次运行如本地没有权重，可移除`--local-files-only`。该脚本需要CUDA GPU及
`transformers`，默认使用BF16、Batch 2、2 Token Prompt和2步Decode。

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
- `docs/qwen2-compatibility.md`：官方 Qwen2 权重映射、RoPE、多层逐项差分和编译结果。
- `docs/compiler-pipeline.md`：对标 `torch.compile` 的整图编译目标、KernelIR 和严格
  零回退契约。
- `docs/linear-lowering.md`：Linear IR契约、Triton tiled GEMM、Qwen2覆盖率和性能结果。
- `docs/qwen2-prefill.md`：完整CausalLM边界、动态Prefill、KV输出和Decode衔接验证。
- `docs/prefill-attention.md`：多Token GQA融合、Triton Online Softmax和性能实验。
- `docs/prefill-kv-state.md`：Prefill状态化、批量KV Store和Decode共享物理ABI。
- `docs/rope-lowering.md`：Qwen2 RoPE子图融合、双结果IR、动态Triton调度和
  TorchInductor性能对比。
- `docs/qwen2-0.5b-validation.md`：真实24层Qwen2-0.5B权重转换、整图编译、
  多步Decode、覆盖率与BF16数值审计。
- `docs/numerical-modes.md`：融合Triton与PyTorch兼容CUDA后端的数值契约、
  KernelIR选择和真实24层逐层误差定位。

## 当前边界

在当前 PyTorch 2.8 环境中，ServeIR 能无 fallback 地导入 76 个 Decoder 操作。
默认优化流水线删除 14 个
导出期断言并融合两个 RMSNorm，将 IR 降至 44 个 Operation。参考执行器已在 FP32/FP16
和多组动态 Shape 下验证优化前后误差为 0。`serve.rms_norm` 已支持 Triton、Inductor
和 PyTorch Native 多后端选择；本机 Profile 证明最优后端会随 Shape 改变。单 Token
Decode 已支持动态历史长度，并将多层嵌套 Tensor Cache 改写为带读写副作用的单一
多 Slot KV 状态；当前两层路径已完成 CPU/GPU 多轮差分。
KV Append 已能自动 Bufferize 为预分配位置写入；RTX 4060 上 Triton Store 相比
`torch.cat` 九组配置几何平均加速 7.064×。Decode Attention 已直接消费物理 KV
Buffer，六组配置相比原展开路径几何平均加速 1.545×；当前仍只支持单 Token、连续
Layout。两层 Qwen2 兼容路径已无外部算子地导入 ServeIR 并完成 GPU 数值差分，但当前
执行器仍会让未 Lower 的 ATen 节点走 PyTorch 参考实现，不能称为完整后端编译。新增的
KernelIR 覆盖报告和 strict 模式已经把这部分缺口显式化。Qwen2中的14个Decode
Linear和两层RoPE已全部进入显式GPU后端，两层Decoder Decode的当前后端覆盖率为
29/76（38.16%）。完整
Qwen2ForCausalLM Prefill已经从Input IDs导出到Logits和多层KV Cache；15个Linear和
两个Prefill Attention及两层RoPE均进入Triton。状态化Prefill会创建同Decode一致的
预分配KV Buffer并批量写入，Decode随后在原地址继续追加；状态化Prefill覆盖率为
27/78（34.62%）。RoPE Kernel在RTX 4060六组配置中相比PyTorch Eager几何平均加速
4.830×、相比TorchInductor加速1.247×。Embedding、SwiGLU和Cosine/Sine生成仍未完成
后端化。真实Qwen2-0.5B已经完成24层BF16官方权重零误差转换，并运行状态化Prefill和
两步连续Decode；真实Prefill覆盖291/738（39.43%），Decode覆盖338/760（44.47%）。
融合Triton路径Top-1与项目Eager一致，但BF16舍入顺序使结果尚非逐元素零误差；新增的
`pytorch_compatible`模式在相同真实模型上已实现Prefill、两步Decode和全部KV逐元素
零误差。详细边界见`docs/numerical-modes.md`。
Paged KV Cache、Block Table和长上下文Split-Sequence属于后续阶段。
