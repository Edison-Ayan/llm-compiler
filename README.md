# StatefulLLM-Compiler

面向动态、有状态 LLM Serving 工作负载的研究型编译器。

当前已经完成两个里程碑：

1. 使用 `torch.export` 捕获带动态 Batch/序列长度的 Qwen 风格 Decoder；
2. 将 Functional ATen 图导入自研 ServeIR，显式建模 SSA、动态类型和 KV 副作用。

项目刻意保持 CPU 可运行。GPU 代码生成、Triton Lowering 和 MLIR 对接属于后续阶段。

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

## 当前边界

ServeIR 能无 fallback 地导入当前 67 个 Decoder 计算操作。默认优化流水线删除 11 个
导出期断言并融合两个 RMSNorm，将 IR 降至 44 个 Operation。Attention 尚未改写为显式
KV 状态操作，`serve.rms_norm` 也尚未 Lower 到可执行 Kernel。
