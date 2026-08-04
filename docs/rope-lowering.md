# Qwen2 RoPE融合与Triton Lowering

## 为什么不能只Lower单个Mul

Qwen2的旋转位置编码在Functional ATen图中展开为：

```text
cosine = unsqueeze(cosine, head_axis)
sine = unsqueeze(sine, head_axis)

rotate_half(x) = cat(-x[..., D/2:], x[..., :D/2])
output = x * cosine + rotate_half(x) * sine
```

Query和Key各自需要两个Slice、一个Neg、一个Cat、两个Mul和一个Add，再加上共享的两个
Unsqueeze。每层一共16个操作。逐个Lower这些节点无法表达它们属于同一个位置编码，也会
保留多次GPU Kernel Launch和中间Tensor。

`FuseRoPEPass`把每层完整子图改写为一个双结果领域操作：

```text
%rotated_query, %rotated_key = serve.rope(
    %query,
    %key,
    %cosine,
    %sine
) {
    head_dim = 64,
    variant = "qwen2_half_rotation"
}
```

两层测试模型因此减少30个IR操作：每层16个展开操作替换为1个`serve.rope`。

## 为什么保留Cosine和Sine输入

Qwen2在Decoder Layer外根据Position ID生成一次Cosine/Sine，所有层共享。如果把
Position ID和Inverse Frequency也融合进每层RoPE Kernel，每层都会重复执行三角函数。

当前边界选择保留：

```text
Position ID → Frequency → Cosine/Sine
                          ↓ 所有层共享
                    serve.rope
```

后续可以把上半部分Lower成位置表查询或运行时RoPE Cache，不需要改变`serve.rope`契约。

## 融合安全性

Pass不是按操作名称数量盲目替换，而是反向验证：

- Query和Key必须共享完全相同的Cosine/Sine Broadcast节点；
- Slice必须发生在最后一维，分割点必须等于`head_dim / 2`；
- Cat顺序必须严格为`[-high_half, low_half]`；
- Head Dim必须是静态正偶数；
- Query、Key、Cosine和Sine的Batch、Token、DType与Device必须一致；
- 除两个最终结果外，待删除中间值不能逃逸出融合子图。

这些约束可以防止普通的Slice、Cat或其他RoPE布局被错误识别为Qwen2半维旋转。

## KernelIR与执行

```text
serve.rope → kernel.triton.rope
```

Kernel同时产生旋转后的Query和Key，不创建Rotate-Half中间Tensor。运行时根据Token数量
选择两个实现变体：

- `tokens == 1`：一个Program处理一个Head行，使用单Warp，优化Decode启动延迟；
- `tokens > 1`：把Query和Key逻辑元素合并为256元素Block，减少长Prefill的小Program
  调度开销。

两个变体都在一次Kernel Launch中覆盖Query和Key。物理输入可以来自
`view(...).transpose(1, 2)`，Kernel通过显式Stride访问，不要求四维Tensor整体连续；只
要求Head Dim连续。

## 正确性

专项GPU测试覆盖：

```text
DType：FP16、FP32
Batch/Token/Head Dim：
    (1, 1, 8)
    (2, 7, 16)
    (1, 33, 32)
```

其中`T=1`覆盖Decode，`T=7/33`覆盖非2次幂动态Prefill。另有严格KernelIR测试保证
独立RoPE图能够在`TritonExecutor(strict=True)`下零回退执行。完整两层Qwen2的
Prefill、Decode、KV状态衔接及Hugging Face差分测试也全部通过。

真实Qwen2-0.5B BF16审计发现，Triton RoPE与官方Eager会因融合乘加和中间舍入顺序
产生最多约一个到两个BF16量化步长的局部差异；差异继续经过24层Attention后会放大。
因此上面的专项FP16/FP32正确性不能外推成BF16逐元素零误差，真实整图结果单独记录在
`qwen2-0.5b-validation.md`。编译器现在提供`pytorch_compatible`模式，把RoPE Lower为
显式`kernel.cuda.rope`复合路径，保留Mul和Add之间的BF16物化边界；真实24层两步
Decode验证为逐元素零误差。该路径不是单Launch Triton实现，详细权衡见
`numerical-modes.md`。

## RTX 4060性能结果

测试配置使用Qwen2-0.5B风格的14个Query Head、2个KV Head和64 Head Dim，FP16，
编译时间不计入延迟：

| Batch | Token | PyTorch Eager | TorchInductor | Triton | Triton/Inductor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 26.64 us | 7.15 us | 4.28 us | 1.67× |
| 1 | 128 | 26.78 us | 6.89 us | 5.41 us | 1.27× |
| 1 | 512 | 44.99 us | 9.40 us | 10.22 us | 0.92× |
| 8 | 1 | 22.86 us | 5.97 us | 3.85 us | 1.55× |
| 8 | 128 | 68.09 us | 16.40 us | 16.53 us | 0.99× |
| 8 | 512 | 259.83 us | 84.74 us | 67.90 us | 1.25× |

几何平均：

```text
Triton / PyTorch Eager：4.830×
Triton / TorchInductor：1.247×
```

结果没有掩盖单项劣势：`B=1,T=512`仍比Inductor慢约8%，`B=8,T=128`基本持平。
这说明后续仍可以针对较小Batch的长序列调整Block大小或交给成本模型选择Inductor。

## 覆盖率变化

两层完整CausalLM Prefill：

```text
RoPE前：22 / 105 = 20.95%
RoPE后：24 / 75  = 32.00%
```

状态化Prefill：

```text
RoPE前：25 / 108 = 23.15%
RoPE后：27 / 78  = 34.62%
```

完整CausalLM Decode：

```text
RoPE前：28 / 108 = 25.93%
RoPE后：30 / 78  = 38.46%
```

覆盖率同时受“融合减少总节点数”和“新增两个已Lower节点”影响，因此不能简单理解为
只多支持了两个ATen操作。

## 当前限制

- 只识别Qwen2默认的前后半维旋转，不支持交错偶奇维RoPE；
- Head Dim必须是静态偶数且不超过65536；
- Cosine/Sine生成仍是ATen回退，尚未接入运行时RoPE Cache；
- 只实现相同DType的Query、Key、Cosine和Sine；
- 当前变体选择只依据Token数量，尚未接入GPU Profile成本模型。

下一项高价值图融合是SwiGLU：把Gate/Up两条Linear后的SiLU和逐元素乘法表达为显式
`serve.swiglu`，再与Triton Kernel及覆盖率闭环连接。
