# Prefill与Decode统一物理KV状态ABI

## 问题

此前两个阶段虽然数学Cache兼容，但运行时表示不同：

```text
Prefill输出：tuple[(key_0, value_0), (key_1, value_1), ...]
Decode输入：PreallocatedKVCacheState
```

如果在阶段切换时调用`from_layer_tensors`重新包装，会分配新Buffer并复制整段Prompt
Cache。真实Serving中Prompt越长，这次复制越昂贵。

## Prefill状态化Pass

`MaterializePrefillKVStatePass`识别函数返回值中的多层`B×KVH×T×D` K/V Pair，构造：

```text
%state0 = serve.kv.init %layer0_key {
    num_layers = 2,
    capacity = 16,
    layout = "contiguous_bshd"
}

%state1 = serve.kv.prefill_store %state0, %key0, %value0 {slot = 0}
%state2 = serve.kv.prefill_store %state1, %key1, %value1 {slot = 1}

return %logits, %state2
```

状态类型：

```text
!serve.kv_state<
    f32,
    layers=2,
    heads=2,
    head_dim=8,
    layout=contiguous_bshd,
    resource=kv,
    capacity=16
>
```

`serve.kv.init`携带`allocate(kv)`和`write(kv)`副作用；每个Prefill Store携带
`read(kv)`和`write(kv)`，因此后续Pass不能把它们跨越依赖错误重排。

## 物理布局与长度

每层K/V Buffer：

```text
[batch, capacity, num_kv_heads, head_dim]
```

逻辑模型输出仍是：

```text
[batch, num_kv_heads, tokens, head_dim]
```

Triton Store直接完成布局写入：

```text
B×H×T×D → B×Capacity×H×D
```

Prefill开始时每层Length为0；写入T个Token后，该Slot返回新的逻辑SSA状态且Length变为
T。物理Buffer由各个SSA版本共享，旧版本只允许观察自己的逻辑前缀。

## KernelIR

```text
serve.kv.init
    → runtime.kv.init

serve.kv.prefill_store
    → kernel.triton.kv_prefill_store
```

`kernel.triton.kv_prefill_store`复用已经支持多Token的Triton KV Store Kernel，不调用
`torch.cat`，也不为每层创建新的Cache Tensor。

状态化Prefill编译统计：

```text
优化计算图：                     75
增加1个Init和2个Store：          78
已Lower：                        27
未Lower：                        51
覆盖率：                     34.62%
```

完整CausalLM Decode：

```text
总操作：78
已Lower：30
未Lower：48
覆盖率：38.46%
```

## 端到端状态传递验证

GPU测试执行：

```text
4 Token Prefill
→ 初始化两层Capacity=16 Buffer
→ 两次Triton Prefill Store
→ Length=[4,4]
→ 完整CausalLM单Token Decode
→ 在Position=4继续写入
→ Length=[5,5]
```

验证内容：

- Prefill Logits通过`3e-5`差分；
- Decode Logits通过`3e-5`差分；
- 两层完整K/V通过`3e-5`差分；
- Prefill和Decode前后的所有Key/Value Buffer `data_ptr()`完全不变；
- Capacity保持16；
- Decode没有重新包装或复制历史Cache。

## 使用方式

```python
prefill_artifact = compile_exported_program(
    prefill_program,
    options=CompileOptions(
        function_name="prefill",
        prefill_kv_state=True,
        kv_capacity=128,
    ),
)

decode_artifact = compile_exported_program(
    decode_program,
    options=CompileOptions(
        function_name="decode",
        preallocate_kv=True,
        kv_capacity=128,
    ),
)
```

命令行生成状态化Prefill KernelIR：

```bash
PYTHONPATH=src python -m stateful_llm_compiler.optimizer \
  artifacts/qwen2_prefill.pt2 \
  --prefill-kv-state \
  --kv-capacity 128 \
  --lower-kernel-ir \
  --out artifacts/qwen2_prefill.kernelir
```

## 当前限制

- Pass目前按函数返回值中的连续K/V Pair识别Layer Slot，后续应增加显式输出签名元数据；
- Capacity必须由编译选项提供，尚未自动连接Serving请求的最大序列长度；
- 每层独立分配K/V Buffer，尚未合并为统一大块Arena；
- 仍是连续布局，不是Paged KV；
- Batch内所有请求使用同一个Prompt长度符号；
- 尚未实现请求级Block Table和动态回收。

下一阶段可以在已有统一ABI和RoPE Lowering上实现Embedding、SwiGLU及RoPE系数生成；
之后再升级为Paged KV Cache，而不需要改动Prefill/Decode上层状态语义。
