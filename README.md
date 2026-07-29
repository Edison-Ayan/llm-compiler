# StatefulLLM-Compiler

A research compiler for dynamic and stateful LLM serving workloads.

The first milestone fixes the compiler's frontend contract:

1. construct a small Qwen-style decoder block;
2. capture it as a normalized ATen graph with `torch.export`;
3. preserve symbolic batch and sequence dimensions;
4. validate the exported program at shapes other than the capture example;
5. emit a deterministic JSON graph summary for the future ServeIR importer.

This is deliberately CPU-runnable. GPU code generation, KV-cache effects and
MLIR lowering are later milestones.

## Run milestone 1

Use a Python environment containing PyTorch 2.8 or newer:

```bash
cd stateful-llm-compiler
PYTHONPATH=src python -m stateful_llm_compiler.frontend \
  --out artifacts/decoder_graph.json \
  --graph-out artifacts/decoder_graph.txt \
  --program-out artifacts/decoder.pt2
```

Run the tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The exported input contract is:

```text
hidden_states:  [batch, sequence, hidden_size]
attention_mask: [batch, 1, sequence, sequence]

1 <= batch <= 8
1 <= sequence <= 128
```

Both dimensions are symbolic. Reusing the same symbolic `sequence` dimension
across the three relevant axes of `attention_mask` records their equality in
the exported program.

## Current boundary

Milestone 1 captures a pure decoder computation. It intentionally does not yet
model KV-cache mutation. Milestone 2 will introduce a small typed ServeIR with
explicit `kv.read` and `kv.append` effects, then import this exported ATen graph
into that IR.

