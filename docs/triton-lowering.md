# `serve.rms_norm` Triton Lowering

## 目标

前几个阶段已经证明：

```text
ATen 子图 → serve.rms_norm
```

在结构和参考语义上正确。本阶段增加第一个真正的 GPU Lowering：

```text
serve.rms_norm
  ↓ Triton Lowering
单 Kernel RMSNorm
```

实验设备：

```text
GPU：NVIDIA GeForce RTX 4060 Laptop 8GB
Compute Capability：8.9
PyTorch：2.9.0+cu128
Triton：3.5.0
```

## Kernel 设计

一个 Triton Program 处理一行 Hidden State：

```text
读取一行 input
  ↓ 转 FP32
计算 sum(x²) / hidden_size
  ↓
rsqrt(variance + epsilon)
  ↓
乘 input 和 weight
  ↓
按 output_dtype 写回
```

Hidden Size 向上取二次幂作为 `block_size`，超出真实 Hidden Size 的位置使用 Mask。

该实现支持：

- 任意前导维度，内部展平为 Rows；
- 动态 Rows；
- FP16、BF16、FP32；
- FP32 累加；
- 独立指定输出 DType；
- `hidden_size <= 65536`。

## Shape 感知调度

Triton 的 `num_warps` 不是越大越好。实验比较了 N=1536 时的 4/8 warps：

- 4 warps 对 M=8 更好；
- 8 warps 对 M=1、32、128 更稳定；
- N=64 时 Reduction 并行度有限，4 warps 足够。

最终规则：

```text
block_size < 2048       → 4 warps
block_size >= 2048
  且 4 <= rows <= 16    → 4 warps
其余                     → 8 warps
```

这还是手工启发式，后续应替换为 Autotune 或 Profile-Guided Cost Model。

## GPU 执行器

`TritonExecutor` 继承 Reference Executor：

- `serve.rms_norm` 使用 Triton；
- 尚未 Lower 的 ATen Operation 继续使用 PyTorch；
- ServeIR 类型和动态 Shape Guard 继续生效。

因此可以在不一次实现完整 GPU 后端的情况下，逐个把高层 ServeIR Operation Lower 到
Triton，并持续进行完整 Decoder 差分测试。

## 正确性

Kernel 独立测试覆盖：

```text
DType：FP16、FP32
Rows：1、7、64、257
Hidden Size：1536
FP32 Compute → FP16 Output
```

完整 Decoder 测试覆盖：

```text
DType：FP16、FP32
Shape：1×1、2×8、3×11
```

所有 GPU 测试通过。16 个性能 Shape 中最大绝对误差：

```text
1.953125e-3
```

该最大值来自 FP16；相对 L2 误差仍处于低量级。FP32 最大绝对误差约 `9.54e-7`。

## 性能基线

基准同时比较：

1. Expanded Eager：与前端一致的多个 PyTorch Operation；
2. Native Eager：`torch.nn.functional.rms_norm`；
3. TorchInductor：编译 Expanded 形式；
4. StatefulLLM Triton Lowering。

首次 JIT/Inductor 编译不计入 Kernel 延迟。每个 Shape 使用 20ms Warmup 和 80ms
重复测量，报告中位数。

16 个 Shape 的几何平均：

| 对比对象 | Triton 加速 |
|---|---:|
| Expanded Eager | 3.95× |
| PyTorch Native | 1.14× |
| TorchInductor | 0.991× |

Triton 在 16 个 Shape 中：

- 相比 Native 胜出 14 个；
- 相比 Inductor 胜出 8 个。

## Qwen Hidden Size

对更接近 Qwen2 的 `N=1536`：

| DType | M | Inductor | Triton | Triton/Inductor |
|---|---:|---:|---:|---:|
| FP16 | 1 | 4.95us | 4.83us | 1.03× |
| FP16 | 8 | 4.01us | 3.64us | 1.10× |
| FP16 | 32 | 4.12us | 4.12us | 约 1.00× |
| FP16 | 128 | 5.31us | 4.50us | 1.18× |
| FP32 | 1 | 4.94us | 5.12us | 0.96× |
| FP32 | 8 | 4.23us | 3.81us | 1.11× |
| FP32 | 32 | 4.09us | 4.26us | 0.96× |
| FP32 | 128 | 6.18us | 5.15us | 1.20× |

N=1536 的几何平均为 Inductor 的 `1.064×`，8 个 Shape 中胜出 6 个。

## 诚实结论

当前 Kernel 已经：

- 显著超过展开 Eager；
- 整体超过 PyTorch Native；
- 在目标 Qwen Hidden Size 上略胜 Inductor。

但它没有全局支配 Inductor。N=64、M=1、FP16 时只有 Inductor 的约 `0.77×`。

因此正确的编译策略不是“所有 RMSNorm 都强制 Triton”，而是：

```text
根据 Shape、DType 和目标 GPU
选择 Triton / Inductor / Native
```

这组负结果支持项目后续的核心方向：Shape-Aware Lowering 和 Cost Model。

## 当前限制

- 当前只实现 RMSNorm；
- 没有与下一层 Linear 继续融合；
- Warp 选择仍是手工规则；
- 性能数据是单 Kernel Micro-benchmark；
- 还没有完整 Decoder 的端到端性能提升数据；
- 暂未记录编译时间和缓存命中率。

