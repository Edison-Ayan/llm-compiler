# ServeIR 第一版设计

## 目标

ServeIR 不是另一个 ATen 文本格式。它的长期目标是显式表达普通张量图无法准确表达的
Serving 语义：

- 动态 Batch 与变长序列；
- Prefill、Decode 和 Mixed 阶段；
- 跨迭代存活的 KV Cache；
- Prefix Cache 的共享与别名；
- KV 状态的读取、追加、分配和释放；
- 面向阶段与 Shape 的多版本代码生成。

当前版本先建立最小基础：类型、SSA、Operation、Function、Module、副作用和校验器。

## SSA 规则

每个 Value 只能定义一次。Operation 的操作数必须来自：

1. 当前函数的 Block 参数；
2. 当前 Operation 之前已经产生的结果。

例如：

```text
%v0 = "aten.linear.default"(%input, %weight)
%v1 = "aten.silu.default"(%v0)
return %v1
```

校验器会拒绝未定义值、使用未来值、重复 SSA 名称和无效返回值。

## 类型系统

第一版包含：

- `TensorType`：Shape、DType、Device；
- `StaticDim`：编译期常量维度；
- `SymbolicDim`：带范围约束的运行期维度；
- `ScalarType`：普通标量和 SymInt；
- `TupleType`：表达 `split/chunk` 的多值结果；
- `KVStateType`：跨推理步骤存活的 KV 状态句柄；
- `UnknownType`：尚不能精确推导的兼容边界。

动态维度直接来自 `torch.export`。同名 `SymbolicDim` 表示维度相等，例如输入和
Attention Mask 共享同一个序列长度符号。

## KV 副作用

`KVStateType` 不是普通 Tensor。它代表一个具有身份的逻辑资源：

```text
!serve.kv_state<
  f16,
  layers=28,
  heads=2,
  head_dim=128,
  layout=blocked,
  resource=kv
>
```

第一版定义两个操作：

```text
serve.kv.read   effects[read(kv)]
serve.kv.append effects[read(kv), write(kv)]
```

`kv.append` 返回一个新的 SSA 状态值，但同时声明底层 KV 资源发生了写入。这样既能用
SSA 表达状态版本，又不会错误地把它当成纯函数并跨越其他 KV 操作重排。

当前校验器保证：

- 第一个操作数必须是 `KVStateType`；
- `kv.read` 必须声明读效果；
- `kv.append` 必须声明读写效果；
- `kv.append` 必须返回同类型的新 KV 状态。

## ATen 导入

导入器读取 `.pt2` 中的 `ExportedProgram`：

1. 参数和用户输入变成函数参数；
2. 每个 FX `call_function` 变成一个 ServeIR Operation；
3. FX Node 引用变成 SSA Operand；
4. 常量和嵌套参数树保存在 Operation Attribute；
5. Tensor Metadata 变成 ServeIR Type；
6. 未识别的调用保留为 `serve.external`。

当前 Decoder 导入结果：

```text
9 个函数参数
67 个 ServeIR 操作
1 个返回值
0 个 external fallback
```

## 当前限制

- 当前导入的 Decoder 仍是纯计算图，尚未把 Attention 改写成显式 KV 操作；
- `UnknownType` 仍用于无返回值的元数据断言；
- 尚未实现 Region、控制流和多 Block；
- 尚未实现 Alias Analysis；
- 尚未实现任何优化 Pass 或代码生成。

## 优化 Pass

当前已完成：

1. `RemoveExportAssertions`：删除只用于 Export Guard 的无结果元数据断言；
2. `FuseRMSNorm`：把 `pow → mean → add → rsqrt → mul` 识别为
   `serve.rms_norm`；
3. Use-Def Analysis、IR Rewriter 和逐 Pass 校验。

详细设计和实验结果见 `docs/passes.md`。下一阶段需要为 `serve.rms_norm` 建立可执行
Reference Lowering，再实现 Triton Lowering 和数值/性能对比。
