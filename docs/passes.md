# ServeIR Pass 与 RMSNorm 融合

## 为什么需要 Pass 基础设施

把 ATen 图打印成自研 IR 不等于实现了编译器。编译器必须能在保持语义和 SSA 合法性的
前提下分析、替换和删除子图。

本阶段实现的最小优化流水线：

```text
ServeIR
  ↓ RemoveExportAssertions
清理后的 ATen IR
  ↓ FuseRMSNorm
包含 serve.rms_norm 的高层 IR
```

## Use-Def Analysis

`UseDefAnalysis` 为每个 SSA Value 建立：

- 唯一 Producer；
- 全部 Operation 使用位置；
- 函数 Return 使用位置；
- Use Count；
- 子图外逃逸判断。

Pattern Rewrite 依赖这些信息回答：

```text
一个中间值是否只在待融合子图内部使用？
删除旧 Operation 后是否仍有消费者引用它？
融合结果是否直接被函数返回？
```

## IR Rewriter

`IRRewriter.replace_subgraph` 执行以下步骤：

1. 确认所有待删除 Operation 都属于当前函数；
2. 确认新 Operation 的操作数不会随旧子图一起删除；
3. 拒绝存在中间值逃逸的子图；
4. 根据新操作数的 Producer 计算合法插入位置；
5. 替换普通 Operand、函数 Return 和 Attribute 参数树中的 SSA 引用；
6. 删除旧 Operation 并插入新 Operation。

插入位置不能简单使用匹配子图的第一个节点。RMSNorm 输入 Cast 还服务于 Residual 分支，
权重 Cast 也可能晚于部分归一化计算；新节点必须位于全部输入定义之后。

## PassManager

`PassManager` 默认在以下位置调用 IR Verifier：

```text
流水线开始
每个 Pass 结束
```

因此任何 Use-Before-Def、脏 Attribute 引用或 KV 副作用错误都会在产生问题的 Pass 后立即
失败，而不是留到代码生成阶段。

每个 Pass 返回：

- 是否改变 IR；
- Operation 数量变化；
- Pass 自定义统计数据。

## RemoveExportAssertions

`torch.export` 会插入 `aten._assert_tensor_metadata.default`，用于检查 DType、Device 和
Layout。ServeIR 已把这些信息编码进函数类型和入口契约，因此可以删除结果未被使用的断言。

Pass 采用保守策略：

- 断言结果没有任何使用：删除；
- 断言结果参与后续计算：保留并计入 `skipped`。

当前 Decoder 删除了 11 个元数据断言：

```text
67 → 56 Operations
```

## RMSNorm Pattern

当前识别的 ATen 结构：

```text
input_cast ─┬→ pow(2) → mean(-1) → add(eps) → rsqrt ─┐
            └→ input_cast ────────────────────────────×
weight_cast ──────────────────────────────────────────×
                                                       ↓
                                                  output_cast
```

融合结果：

```text
%result = "serve.rms_norm"(%input_cast, %weight_cast)
  {axis=-1, epsilon=1e-6, compute_dtype="f32"}
```

输入和权重 Cast 不包含在删除集合中，因为输入 Cast 同时被 Residual 分支使用。保留这些
共享节点，再交给后续 Cast Canonicalization/DCE 处理，可以避免错误删除合法外部使用。

当前单层 Decoder 识别出两个 RMSNorm：

```text
56 → 44 Operations
```

完整流水线：

```text
67 → 56 → 44 Operations
总计减少 23 个，下降 34.3%
```

这里的 Operation 减少表示 IR 抽象层次提高，并不等价于运行时间提升。参考执行器已经
验证融合前后数值等价；只有 `serve.rms_norm` Lower 到实际融合 Kernel 后，才能评测
Kernel 数量和性能收益。

## 负例与幂等性

测试覆盖两类容易被忽略的条件：

1. 如果 `pow` 等中间结果被匹配子图外的 Operation 使用，融合必须拒绝；
2. 同一个 Pass Pipeline 运行第二次，不应继续修改 IR。

这分别验证了重写合法性和 Pass 幂等性。

## 当前限制

- 只识别当前 Qwen 风格 RMSNorm 展开形式；
- 尚未支持 Pattern Benefit 和多个 Pattern 的优先级；
- 参考执行器已支持当前融合 IR，但不是高性能后端；
- 尚未 Lower 到 Triton；
- 尚未实现通用 DCE 和 Cast Canonicalization。
