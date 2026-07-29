# ServeIR 参考执行与语义验证

## 目的

结构合法不等于语义正确。RMSNorm Pattern 即使通过 SSA 校验，也可能因为以下问题改变
结果：

- 漏掉 DType Cast；
- Epsilon、Axis 或 Compute DType 错误；
- FP16 与 FP32 导出结构不同；
- 重写后操作数顺序错误；
- 动态 Shape 符号绑定错误。

因此在 Triton Lowering 前实现一个 CPU/PyTorch 参考执行器，用它回答：

> 优化后的 ServeIR 在不同 DType 和动态 Shape 下，是否与原始 ExportedProgram 数值等价？

## Reference Executor

参考执行器按函数中的 Operation 顺序维护运行时环境：

```text
SSA Value Name → PyTorch Runtime Value
```

每执行一个 Operation：

1. 从 Attribute 参数树解码位置参数和关键字参数；
2. 将 `{"ssa": "%v0"}` 解析为对应运行时 Value；
3. 调用 PyTorch 参考语义；
4. 检查结果数量；
5. 用 ServeIR Type 校验结果；
6. 写回运行时环境。

当前覆盖优化后 Decoder 所需的全部 17 类操作：

```text
sym_size, to, linear, split, getitem, view, transpose
repeat_interleave, matmul, div, add, softmax, reshape
chunk, silu, mul, serve.rms_norm
```

参考执行器不用于性能测试。它的作用是给后续 Lowering 提供可信的语义基线。

## ExportedProgram 参数绑定

ServeIR 函数同时包含：

- Lifted Parameter；
- Buffer；
- Constant Tensor；
- User Input。

`bind_exported_program_arguments` 根据 Graph Signature 自动按顺序绑定：

```text
PARAMETER       → program.state_dict[target]
BUFFER          → program.state_dict[target]
CONSTANT_TENSOR → program.constants[target]
USER_INPUT      → 调用方输入
```

因此差分测试不需要手工维护 Parameter 名称和顺序。

## 运行期 Shape Guard

参考执行器会检查：

- Tensor Rank；
- DType；
- Device；
- 静态维度；
- 符号维度范围；
- 同名符号维度的相等约束。

例如：

```text
hidden_states  : [s_batch, s_sequence, hidden]
attention_mask : [s_batch, 1, s_sequence, s_sequence]
```

如果 Hidden 的 Sequence 为 8、Mask 的 Sequence 为 7，执行器会报告符号相等约束失败。
如果 Batch 为 9，而导出范围是 `[1, 8]`，执行器会在执行任何 Operation 前拒绝输入。

## FP16 Pattern 变体

差分测试发现 PyTorch 的 FP32 和 FP16 RMSNorm 导出图不同：

```text
FP32:
  input_cast_2(input_cast_1(input))

FP16:
  input_cast_1(input)
  input_cast_2(input)
```

两者语义上都从同一个原始输入生成 FP32 计算值，但结构不同。RMSNorm Pattern 现在同时
接受：

```text
cast_2.input == cast_1.result
或
cast_2.input == cast_1.input
```

融合 Operation 还显式记录：

```text
compute_dtype = f32
output_dtype  = f16 / f32
```

避免 FP16 输入经 FP32 归一化后忘记 Cast 回 FP16。

## 差分结果

测试 Shape：

```text
1×1
2×8
3×13
4×17
8×32
```

FP32 和 FP16 均得到：

```text
max_abs_error    = 0.0
relative_l2_error = 0.0
执行 Operation    = 44
```

这说明当前输入范围和参考后端下，优化后的 ServeIR 与原始 ExportedProgram 逐元素一致。

## 当前限制

- 参考执行器只覆盖当前优化后 Decoder 使用的操作；
- 只实现单函数、单 Block、无控制流执行；
- `serve.kv.read/append` 已在 Stateful Decode 里接入不可变 KV Runtime；当前仍使用
  `torch.cat` 追加，尚未 Lower 到预分配或分页 Cache；
- 当前结果来自 CPU PyTorch，不是 GPU Kernel；
- 数值等价不能证明未来 Triton Lowering 也等价，Triton 仍需独立差分测试。
