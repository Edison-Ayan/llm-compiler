# Linear 从 ATen 到 Triton 的完整 Lowering

## 编译路径

本阶段把 Qwen2 中所有 Functional ATen Linear 经过两层 IR 下沉：

```text
aten.linear.default
    ↓ NormalizeLinearPass
serve.linear
    ↓ LowerToKernelIRPass
kernel.triton.linear
    ↓ TritonExecutor
Triton tiled GEMM
```

`serve.linear` 保留模型级语义和类型契约；`kernel.triton.linear` 表示后端已经确定，
strict 模式不再允许调用 `torch.nn.functional.linear`、`torch.matmul` 或
`torch.compile` 承接计算。

## IR 契约

当前支持二维和三维输入：

```text
input:  [..., K]
weight: [N, K]
bias:   [N]，可选
output: [..., N]
```

三维 Decoder 输入 `[batch, tokens, K]` 在 Kernel 中逻辑展平为：

```text
M = batch × tokens
output[M, N] = input[M, K] × weight[N, K]ᵀ + bias[N]
```

Verifier会检查：

- 输入只允许二维或三维；
- Weight必须是二维静态 `N×K`；
- 输入最后一维等于 K；
- 输出前导维与输入一致，最后一维等于 N；
- Bias存在时必须是一维 N；
- Input、Weight、Bias和 Output 的 DType、Device 一致；
- `has_bias`、`input_features`、`output_features` Attribute 与类型一致。

## Triton Kernel

Grid 的两个轴分别覆盖 M 和 N：

```text
program_id(0) → 输出行 Tile
program_id(1) → 输出列 Tile
```

每个 Program 按 `BLOCK_K` 遍历归约维：

```text
input_tile:  BLOCK_M × BLOCK_K
weight_tile: BLOCK_K × BLOCK_N
accumulator: BLOCK_M × BLOCK_N，FP32
```

Weight保持 PyTorch 的物理 `N×K` 布局。地址计算直接把它作为 `K×N` Tile加载，
因此不创建转置 Tensor。Bias也在同一个 Kernel 中融合，避免额外逐元素 Kernel。

FP32路径为 `tl.dot` 指定 IEEE 输入精度，避免默认 TF32 舍入影响 Qwen2 高精度差分。
当前调度使用简单的 16/32/64 Tile规则，尚未实现 Autotune和硬件 Profile。

## Qwen2 覆盖率变化

两层 Qwen2 Decode 优化后仍为111个操作：

```text
Linear Lowering 前：13 / 111 = 11.71%
Linear Lowering 后：27 / 111 = 24.32%
```

新增的14个后端操作来自：

- 6个 Q/K/V Projection；
- 2个 Attention Output Projection；
- 2个 Gate Projection；
- 2个 Up Projection；
- 2个 Down Projection。

## RTX 4060 Laptop 初步结果

FP16、融合 Bias，单位为微秒；PyTorch Eager Linear通常由 cuBLAS承接：

| M | K=N | PyTorch/cuBLAS | Triton | Triton相对速度 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 12.01 | 13.59 | 0.88× |
| 8 | 512 | 12.89 | 12.97 | 0.99× |
| 32 | 512 | 12.63 | 14.30 | 0.88× |
| 1 | 1536 | 36.88 | 34.07 | 1.08× |
| 8 | 1536 | 34.24 | 34.13 | 1.00× |
| 32 | 1536 | 34.40 | 35.04 | 0.98× |

结果说明初版 Kernel已经达到接近库实现的量级，但没有全面超过 cuBLAS。后续正确方向是：

1. 增加 `BLOCK_M/N/K`、Warp数量和 Grouped Ordering 调优；
2. 为 Decode小 M 和 Prefill大 M 建立不同配置；
3. Profile Triton与 cuBLAS，生成 Shape相关后端选择；
4. 优先把 Bias、激活或后续逐元素计算融合进 Triton，获得单独 GEMM没有的优势。

复现实验：

```bash
PYTHONPATH=src python benchmarks/bench_linear.py \
  --rows 1,8,32 \
  --shapes 512x512,1536x1536 \
  --bias \
  --out artifacts/linear_benchmark_v1.json
```
