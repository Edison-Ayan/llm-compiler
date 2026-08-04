# 真实Qwen2-0.5B Checkpoint验证

## 这一步要回答什么

此前两层小模型适合快速验证Pass和IR，但不能回答以下问题：

- 官方预训练权重能否完整映射，而不只是随机小配置；
- 24层图是否能在合理内存内完成Export、优化和Lowering；
- Prefill产生的物理KV状态能否直接被多步Decode继续修改；
- 自研Triton Kernel在真实BF16模型上是否仍保持数值一致；
- 当前到底有多少操作进入了编译后端，还有多少依赖兼容执行。

本阶段使用`Qwen/Qwen2-0.5B`公开Checkpoint建立可重复的端到端验证脚本。它不是
吞吐Benchmark，而是对真实模型规模的结构、状态、数值和资源闭环。

## 模型与实验配置

```text
模型：                 Qwen/Qwen2-0.5B
参数量：               494,032,768
Decoder层数：          24
Hidden Size：          896
Intermediate Size：    4864
Query / KV Head：      14 / 2
Head Dim：             64
Vocab Size：           151,936
DType：                BF16
GPU：                  RTX 4060 Laptop 8GB
输入：                 Batch=2，Prompt=2 Token，连续Decode=2步
KV Capacity：           8
```

短Prompt是为了在8GB显存上低成本重复检查整条链路，不代表编译器只支持两个Token。
前端仍导出动态Batch和动态序列长度；捕获样例Batch必须至少为2，是PyTorch 2.8会把
Batch=1特化造成的导出约束，而不是运行时不能执行Batch=1。

## 权重转换中发现的两个真实问题

### RMSNorm的BF16舍入顺序

项目原来计算：

```text
(normalized_fp32 * weight_fp32).to(bf16)
```

Hugging Face Qwen2实际计算：

```text
weight_bf16 * normalized_fp32.to(bf16)
```

两者在FP32或两层小模型上不容易暴露，但BF16误差经过24层会累积，曾造成Logits最大
误差2.73、KV最大误差6.0。现在模型实现、RMSNorm融合Attribute、参考执行器和Triton
Kernel均保留`round_before_weight`语义。

### RoPE频率Buffer的DType

官方`inv_freq`是非持久化FP32 Buffer。项目模型整体执行`.bfloat16()`时曾把它一起
转换为BF16，导致频率在计算三角函数前已经损失精度。现在设备或DType迁移后会根据
`head_dim`和`rope_theta`重新构造FP32频率。

修复后，官方模型与项目Eager在真实24层上的结果为：

```text
Prefill Logits最大误差：       0
Prefill全部24层K/V最大误差：   0
Decode第1步Logits最大误差：    0
Decode第2步Logits最大误差：    0
最终全部24层K/V最大误差：      0
```

这证明模型适配和权重转换是逐元素一致的，不再只是Shape能够运行。

## Linear后端选择

真实模型还暴露出初版Triton GEMM在`4864→896`等大归约BF16形状上的误差会明显放大。
逐节点审计发现，169个Linear中只有24个`down_proj`出现显著差异。当前Lowering把后端
选择显式写入KernelIR：

```text
大静态GEMM： kernel.cublas.linear    73个
其余投影：   kernel.triton.linear    96个
```

73个库路径包括每层Gate、Up、Down Projection和LM Head；96个Triton路径包括每层
Q/K/V/O Projection。`kernel.cublas.linear`目前由执行器调用`F.linear`进入CUDA库，
是显式可统计的cuBLAS Lowering原型，并非直接绑定cuBLAS C API。后续应把它迁入独立
Runtime，避免执行器层依赖PyTorch API。

## 真实整图覆盖率

状态化Prefill：

```text
总操作：       738
已Lower：      291
未Lower：      447
覆盖率：       39.43%
```

已Lower操作包括73个cuBLAS Linear、96个Triton Linear、49个RMSNorm、24个RoPE、
24个Prefill Attention、24个KV Prefill Store和1个KV Init。

完整Decode：

```text
总操作：       760
已Lower：      338
未Lower：      422
覆盖率：       44.47%
```

已Lower操作包括73个cuBLAS Linear、96个Triton Linear、49个RMSNorm、24个RoPE、
24个Decode Attention、24个KV Store、24个Length和24个Advance。

未Lower部分仍由兼容执行器运行，主要是Embedding、SwiGLU的SiLU/Mul、RoPE的
Cosine/Sine生成、Cast以及View/Transpose等元数据操作。因此目前是“真实整图已捕获、
关键算子已编译、仍有显式回退”，还不能宣称达到`torch.compile`的完整零回退能力。

## BF16编译路径数值审计

编译后的结果与项目Eager对比：

| 阶段 | 最大绝对误差 | 平均绝对误差 | Top-1一致 |
| --- | ---: | ---: | --- |
| Prefill | 0.703125 | 0.035508 | 是 |
| Decode第1步 | 1.375 | 0.123950 | 是 |
| Decode第2步 | 1.1875 | 0.140772 | 是 |

Prefill物理KV Cache最大绝对误差为0.5，最终两步Decode后为0.5546875。所有48个K/V
Buffer在Prefill和两步Decode前后地址不变，24层Length都从2正确递增到4。

这组误差不能表述成“数值等价”。逐算子替换实验得到：

```text
Triton RoPE + Triton Attention：    Prefill Logits最大误差0.703125
参考RoPE + Triton Attention：       最大误差0.53125
Triton RoPE + 参考Attention：       最大误差0.21875
参考RoPE + 参考Attention：          最大误差0
```

因此剩余差异来自Triton RoPE和Attention在BF16下的融合乘加、累加与中间舍入顺序；
RMSNorm和当前选择后的Linear已经排除。Top-1一致说明链路可运行，但下一阶段仍需建立
明确的精度预算，并决定是复刻官方舍入、在关键位置提高精度，还是采用端到端容差标准。

## 时间与显存

以下是一次首轮、包含必要同步的测量，不是预热后的稳定吞吐数据：

| 项目 | 时间 |
| --- | ---: |
| 加载官方权重 | 0.096 s |
| 转换项目模型 | 3.071 s |
| Prefill Export | 6.170 s |
| Prefill Compile | 1.212 s |
| Prefill首次执行 | 299.87 ms |
| Decode Export | 9.030 s |
| Decode Compile | 1.400 s |
| Decode第1步执行 | 23.33 ms |
| Decode第2步执行 | 18.17 ms |

峰值GPU已分配显存约为：官方Eager 961.1MiB、项目Eager 961.5MiB、编译Prefill
968.3MiB、编译Decode 966.2MiB。数据说明0.5B真实模型能稳定放入8GB显存完成编译
验证，但不能据此评价稳态Tokens/s。

## 复现方式

```bash
PYTHONPATH=src python benchmarks/validate_qwen2_checkpoint.py \
  --local-files-only \
  --decode-steps 2 \
  --out artifacts/qwen2_0_5b_validation.json
```

脚本的硬通过条件包括：官方与项目Eager逐元素零误差、每一步编译Decode的Top-1一致、
KV Buffer地址稳定以及所有层Length正确。详细原始结果写入指定JSON。

## 下一阶段

当前最值得继续的方向不是马上宣称“完整推理已完成”，而是同时推进：

1. 融合并Lower SwiGLU，减少每层`SiLU + Mul`及相关中间Tensor；
2. 把View、Transpose、Reshape和Cast归入Metadata/Conversion Dialect；
3. 为RoPE和Attention建立BF16逐算子误差预算与可选择高精度变体；
4. 将Embedding和RoPE Cache后端化，继续降低整图回退数量；
5. 在更长Prompt、更多Decode步数上做预热后的TorchInductor对照Benchmark。
