# BF16双数值模式与后端选择

## 为什么需要数值模式

图融合不仅改变Kernel数量，也可能改变浮点程序。Qwen2 BF16原图在以下位置会物化
中间Tensor并发生舍入：

```text
RoPE：Mul → BF16 Tensor → Add
Attention：QK Matmul → BF16 → Scale → BF16 → FP32 Softmax
           → BF16 Probability → PV Matmul → BF16
RMSNorm：FP32归约/Rsqrt → BF16 Normalized → Weight Mul
```

单核Triton RoPE和Online Softmax Attention会消除部分中间Tensor；Triton RMSNorm的
归约和`rsqrt`也不保证与PyTorch CUDA逐位相同。数学公式没有变化，但24层残差网络会
放大一个BF16量化步长的局部差异。因此编译器不能只提供一个没有精度契约的后端。

## 两种模式

`CompileOptions.numerical_mode`当前支持：

| 模式 | 目标 | 数值契约 |
| --- | --- | --- |
| `fast` | 保留融合Triton研究路径 | Top-1一致并保持KV状态不变量；记录实际误差 |
| `pytorch_compatible` | 保留官方CUDA算子与BF16舍入边界 | 项目Eager逐元素零误差 |

`fast`表示面向融合和性能实验的候选路径，不代表在所有Shape上已经快于CUDA库；稳定
性能必须由预热Benchmark和成本模型决定。

## KernelIR中的显式区别

融合模式：

```text
serve.linear              → kernel.triton.linear / kernel.cublas.linear
serve.rms_norm            → kernel.triton.rms_norm
serve.rope                → kernel.triton.rope
serve.prefill_attention   → kernel.triton.prefill_attention
serve.decode_attention    → kernel.triton.decode_attention
```

兼容模式：

```text
serve.linear              → kernel.cublas.linear
serve.rms_norm            → kernel.cuda.rms_norm
serve.rope                → kernel.cuda.rope
serve.prefill_attention   → kernel.cuda.prefill_attention
serve.decode_attention    → kernel.cuda.decode_attention
```

每个受影响的KernelIR操作都带有`numerical_mode` Attribute，Linear还会记录
`backend_selection`。因此覆盖报告可以区分真实Triton Kernel、cuBLAS库调用和保留原图
舍入边界的CUDA复合操作。

`kernel.cuda.*`当前由执行器使用PyTorch CUDA算子组合实现。它是编译器显式选择的
可审计后端，不是未Lower节点的意外回退；但它仍依赖PyTorch Runtime，不能表述成已经
生成了独立CUDA/C++ Kernel。后续可把这些复合操作Lower到CUDA Graph或独立Runtime。

## 根因定位过程

真实Qwen2-0.5B两步Decode揭示了逐层放大过程：

1. 只把RoPE和Attention切换到兼容路径后，Prefill与第1步Decode为零误差；
2. 第2步从第16层K/V开始出现`0.015625`，随后增长至`0.125`；
3. 把全部Linear切到cuBLAS、恢复原图`repeat_interleave`后，偏差位置完全不变；
4. 最后把RMSNorm切换为CUDA兼容路径，24层所有Logits和KV误差全部归零。

这说明首个跨步偏差来自特定输入上的Triton RMSNorm归约/`rsqrt`结果，而RoPE和
Attention是此前Prefill大误差的主要来源。不能因为单个随机测试通过，就宣称一个BF16
Kernel在所有输入上逐位兼容。

## Qwen2-0.5B结果

配置为BF16、Batch 2、2 Token Prompt和连续2步Decode：

| 指标 | `fast` | `pytorch_compatible` |
| --- | ---: | ---: |
| Prefill Logits最大误差 | 0.703125 | 0 |
| Prefill KV最大误差 | 0.5 | 0 |
| Decode两步Logits最大误差 | 1.375 | 0 |
| 最终KV最大误差 | 0.5546875 | 0 |
| 两步Top-1 | 全部一致 | 全部一致 |
| Buffer地址与Length | 正确 | 正确 |

两种模式的图级覆盖率相同：Prefill为291/738（39.43%），Decode为338/760
（44.47%）；不同的是已Lower操作选择了哪个后端。

单次首轮执行数据中兼容模式没有比融合模式更慢，但该输入只有2个Prompt Token，且没有
预热和重复采样，不能据此做性能结论。后续必须在长Prefill和多轮Decode上分别测量。

## 使用方式

Python编译入口：

```python
artifact = compile_exported_program(
    program,
    options=CompileOptions(
        numerical_mode="pytorch_compatible",
    ),
)
```

命令行：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.optimizer \
  artifacts/qwen2_decode.pt2 \
  --lower-kernel-ir \
  --numerical-mode pytorch_compatible \
  --out artifacts/qwen2_decode.compatible.kernelir
```

真实Checkpoint验证：

```bash
PYTHONPATH=src python benchmarks/validate_qwen2_checkpoint.py \
  --local-files-only \
  --numerical-mode pytorch_compatible \
  --out artifacts/qwen2_0_5b_compatible_validation.json
```

兼容模式下脚本会把Prefill/Decode Logits或KV任何非零误差视为失败；融合模式则保留
误差明细，并要求每一步Top-1及KV地址、Length不变量正确。

## 后续研究方向

1. 为RMSNorm增加FP32 Reference误差测试，而不只比较PyTorch BF16位模式；
2. 分别测量融合与兼容后端的稳态延迟、Launch数和临时显存；
3. 把模式从整图选项细化为每类算子的数值预算与成本模型决策；
4. 实现两阶段Triton RoPE，研究能否在少量中间Buffer代价下复刻BF16舍入；
5. 为Attention增加高精度Triton变体，并与官方BF16和FP32参考同时比较。
