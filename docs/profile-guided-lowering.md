# Profile-Guided Lowering

## 目标

同一个 RMSNorm 算子没有对所有 Shape 都最优的实现。上一阶段的 GPU 测量已经表明：
Triton、Inductor 和 PyTorch Native 的延迟曲线会随着 `rows`、`hidden_size` 和
`dtype` 交叉。因此，编译器不应把所有 `serve.rms_norm` 无条件 Lower 到同一个后端。

这一阶段把性能测量接入编译流水线：

```text
GPU Benchmark
    ↓
Target Profile
    ↓
RMSNorm Cost Model
    ↓
SelectRMSNormLoweringPass
    ↓
带动态 Shape 多版本计划的 ServeIR
    ↓
运行时按实际 Rows 分派
```

这里的 `rows` 是输入除最后一维外的元素总数。对于
`[batch, sequence, hidden_size]`，它等于 `batch × sequence`。

## Profile 格式

`benchmarks/bench_rmsnorm.py` 生成 schema v1 JSON。除了每个测量点的延迟，
它还记录画像适用的软硬件环境：

```json
{
  "schema_version": 1,
  "target": {
    "device_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
    "compute_capability": "8.9",
    "torch_version": "2.9.0+cu128",
    "triton_version": "3.5.0"
  },
  "benchmark": {
    "epsilon": 1e-6,
    "warmup_ms": 20,
    "rep_ms": 80,
    "compile_time_included": false
  },
  "results": []
}
```

编译时间被排除，因此这一模型优化的是预热后的稳态 Kernel 延迟，不代表第一次请求
的端到端延迟。成本模型仍能读取第五阶段生成的旧版纯数组 JSON，但旧格式没有 Target
信息。

## 动态 Shape 分桶

成本模型只在 `hidden_size` 和 `dtype` 精确匹配时使用测量点，不跨 Hidden Size
插值。当前 Profile 的 Rows 测量点为：

```text
1, 8, 32, 128
```

相邻点之间使用几何中点作为边界：

```text
boundary(a, b) = floor(sqrt(a × b))
```

最终得到：

| 运行时 Rows | 代表测量点 |
|---:|---:|
| 1～2 | 1 |
| 3～16 | 8 |
| 17～64 | 32 |
| 65 及以上 | 128 |

Rows 的候选值通常按数量级变化。几何中点等价于在对数尺度上选择最近测量点，比算术
中点更适合这种采样方式。它仍然是启发式近似，并不意味着边界一定就是后端曲线的真实
交点。

每个代表点分别比较 `native_eager_us`、`triton_us` 和 `inductor_us`，选择实测
延迟最低的后端。若选择 Triton，计划还会固化该 Shape 对应的 `num_warps`。

## 编译与运行时分派

启用 Profile 后，默认 Pass 流水线为：

```text
RemoveExportAssertionsPass
FuseRMSNormPass
SelectRMSNormLoweringPass
```

选择 Pass 不增加 Operation，而是在每个 `serve.rms_norm` 上附加
`lowering_plan`。例如，本机 `N=1536, FP32` 的一次实测产生：

| 运行时 Rows | Backend | 代表点 | 估计延迟 |
|---:|---|---:|---:|
| 1～2 | Triton | 1 | 4.70 μs |
| 3～16 | Triton | 8 | 3.94 μs |
| 17～64 | Inductor | 32 | 4.12 μs |
| 65 及以上 | Triton | 128 | 5.44 μs |

`TritonExecutor` 根据实际 Tensor Shape 计算 Rows，再解析这份计划：

- `triton`：启动自研 Triton Kernel，并使用计划中的 `num_warps`；
- `native`：调用 PyTorch Native RMSNorm；
- `inductor`：按 Device、DType、Hidden Size 和 Epsilon 缓存动态编译结果；
- 没有匹配测量点：使用计划中的 `fallback`，默认为 Inductor；
- IR 没有 Lowering 计划：保持原行为，默认使用 Triton。

执行器会把本次选择写入 `lowering_trace`，包括 Backend、Rows、代表测量点、估计延迟
和 `num_warps`，便于测试和诊断。

## 本机结果

在 RTX 4060 Laptop GPU 上重新采集 16 个 Shape/DType 组合：

- Triton 相比展开 PyTorch Eager 的几何平均加速为 `3.927×`；
- Triton 相比 PyTorch Native 的几何平均加速为 `1.135×`，16/16 个点获胜；
- Triton 相比 Inductor 的全矩阵几何平均为 `0.984×`，7/16 个点获胜；
- 仅看 `hidden_size=1536`，Triton 相比 Inductor 为 `1.048×`，5/8 个点获胜；
- 最大绝对误差为 `0.001953125`，出现在 FP16 测试中。

全矩阵上 Triton 并没有稳定超过 Inductor，这正是 Profile-Guided Selection 存在的
必要性：编译器保留两种实现，根据动态 Shape 选择，而不是用单一平均值掩盖曲线交叉。

## 使用方法

采集目标机器 Profile：

```bash
PYTHONPATH=src python benchmarks/bench_rmsnorm.py \
  --rows 1,8,32,128 \
  --hidden-sizes 64,1536 \
  --dtypes fp16,fp32 \
  --inductor \
  --warmup-ms 20 \
  --rep-ms 80 \
  --out artifacts/rmsnorm_benchmark_v1.json
```

单独查看一个签名的多版本计划：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.cost_model \
  artifacts/rmsnorm_benchmark_v1.json \
  --hidden-size 1536 \
  --dtype fp32
```

把 Profile 接入优化器：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.optimizer \
  artifacts/decoder.pt2 \
  --profile artifacts/rmsnorm_benchmark_v1.json \
  --out artifacts/decoder.profile.optimized.serveir \
  --stats-out artifacts/profile_optimization_stats.json
```

## 当前边界

- Profile 与 GPU、驱动、PyTorch、Triton 版本相关；当前会记录 Target，但还没有在加载
  时强制拒绝不匹配的运行环境；
- 分桶边界由离散采样点推导，没有主动搜索真实性能交点；
- 当前成本只包含稳态 Kernel 延迟，尚未加入首次编译成本、显存占用、代码大小和
  Serving P99 延迟；
- Inductor 动态 Kernel 在第一次进入对应签名时仍会产生编译开销；
- 目前只为 RMSNorm 建模，Attention/KV Cache 等状态型算子还没有进入成本模型。

