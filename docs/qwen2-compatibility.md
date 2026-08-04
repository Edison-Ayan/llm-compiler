# Hugging Face Qwen2 Stateful Decode 兼容性

## 目标

早期 `StatefulTinyDecoder` 只复现了 GQA、RMSNorm、SwiGLU 和 KV Cache 的基本形状，
不能证明编译器能够处理真实模型语义。本阶段以本地 `transformers 5.9.0` 的
`Qwen2Model` 为参考，建立不下载权重也可以重复执行的随机权重差分测试。

项目实现位于`src/stateful_llm_compiler/qwen2.py`。`StatefulQwen2Model`保留直接接收
Input Embedding的Decoder边界；`StatefulQwen2ForCausalLM`进一步加入Token Embedding
和LM Head。Decoder参数名称与Hugging Face保持一致，完整模型通过显式映射复制权重。

## 已对齐的模型结构

- 独立 `q_proj`、`k_proj`、`v_proj` 和 `o_proj`；
- GQA Query Head 到 KV Head 映射；
- 默认 RoPE，包括 FP32 Frequency 计算、`rotate_half` 和 Position ID；
- Attention Scale、FP32 Softmax 和加法 Mask；
- 独立 `gate_proj`、`up_proj`、`down_proj` 的 SwiGLU；
- Pre/Post Attention RMSNorm 和最终 RMSNorm；
- Hugging Face 风格 `((key_0, value_0), ...)` 多层 Cache。

Qwen2 官方实现没有 Q/K RMSNorm，因此兼容层不会为了增加算子数量而错误加入该操作。
当前只支持默认 RoPE、SiLU 和 Full Attention；Sliding Window 与其他 RoPE Scaling
会在配置转换时明确拒绝。

## 权重映射

`StatefulQwen2Model.from_huggingface()` 从 `Qwen2Model` 提取配置，检查当前支持边界，
然后按相同参数名称复制所有 Decoder 权重。两边参数集合的关系是：

```text
project_state_dict = huggingface_state_dict - {embed_tokens.weight}
```

测试使用两层、Hidden Size 32、4 个 Query Head、2 个 KV Head 的随机初始化官方模型。
单 Token Decode 的以下结果最大绝对误差均为 0：

```text
Layer 0 Hidden State
Layer 1 Hidden State
Final Hidden State
Layer 0/1 Present Key
Layer 0/1 Present Value
```

## 编译链

Qwen2 两层单 Token Decode 导出 253 个 FX Node，导入得到 219 个 ServeIR Operation，
且没有 `serve.external` fallback。优化过程为：

```text
导入：                       219
删除断言与死 Shape 元数据：  157
融合 5 个 RMSNorm：          127
恢复两个 KV Slot：           127
Bufferize 两个 KV Append：   131
融合两个 Decode Attention：  111
Lower 5个 RMSNorm、14个 Linear、2个 KV Store、2个 Attention和4个 Runtime操作：111
```

当前29/81个操作已经进入 KernelIR 后端，覆盖率为35.80%；剩余52个操作会由严格
模式准确报告，不能再把参考执行器回退当成完整编译。

Qwen2 使用：

```text
scores = matmul(query, keyᵀ) * head_dim**-0.5
```

因此 `FuseDecodeAttentionPass` 同时支持旧图的正数除法缩放和 Qwen2 的正数乘法缩放，
并继续验证非交换 Matmul 的 Operand/FX 参数顺序。

RoPE 引入了 `slice`、`cat`、`neg`、`cos`、`sin` 和 `unsqueeze`。参考执行器已补充这些
ATen 语义。`inv_freq` 是 `persistent=False` Buffer，参数绑定器会从
`ExportedProgram.constants` 读取，而不是错误假设所有 Buffer 都位于 `state_dict`。

## 正确性

- Hugging Face 官方 Eager 与项目 PyTorch 模型：最大绝对误差 0；
- 优化后 ServeIR Reference Executor 与 ExportedProgram：最大绝对误差 0；
- 包含14个 Triton Linear的 KernelIR GPU路径通过 `3e-5` 精度差分；
- 两个 Attention 都被融合，并直接消费物理 KV Buffer。

运行：

```bash
PYTHONPATH=src python -m unittest discover \
  -s tests -p 'test_qwen2_compat.py' -v
```

安装可选参考依赖：

```bash
pip install -e '.[gpu,hf]'
```

## 当前边界

- 已捕获Input IDs、Embedding和LM Head，但Embedding仍未Lower到后端；
- 通过随机初始化官方模型验证映射，尚未加载公开预训练权重；
- 只支持默认 RoPE、Full Attention 和 SiLU；
- 已导出动态多Token Prefill并验证Tensor Cache衔接Decode，尚未统一物理状态ABI；
- 仍使用连续 KV Buffer，尚未实现 Paged KV 和 Block Table；
- 当前是两层小配置正确性实验，不是完整 Qwen2 端到端 Tokens/s Benchmark。
