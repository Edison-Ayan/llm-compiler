# Hugging Face Qwen2 Stateful Decode 兼容性

## 目标

早期 `StatefulTinyDecoder` 只复现了 GQA、RMSNorm、SwiGLU 和 KV Cache 的基本形状，
不能证明编译器能够处理真实模型语义。本阶段先以本地 `transformers 5.9.0` 的
`Qwen2Model` 建立不下载权重也可以重复执行的随机权重差分测试，随后又加载真实
Qwen2-0.5B BF16 Checkpoint完成24层验证。

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

Qwen2 两层单 Token Decode 导入得到 209 个 ServeIR Operation，
且没有 `serve.external` fallback。优化过程为：

```text
导入：                                       209
融合RMSNorm、RoPE、Decode Attention并物化KV状态
优化及Lower后的KernelIR：                     76
```

当前29/76个操作已经进入 KernelIR 后端，覆盖率为38.16%；剩余47个操作会由严格
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

真实Qwen2-0.5B进一步验证了24层、494,032,768个参数的映射。官方模型与项目Eager的
Prefill Logits、两步Decode Logits及所有层K/V均逐元素零误差。编译路径的BF16误差、
整图覆盖率和资源数据见`qwen2-0.5b-validation.md`。

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
- 已加载Qwen2-0.5B公开预训练权重；更大Qwen2模型尚未验证；
- 只支持默认 RoPE、Full Attention 和 SiLU；
- 已导出动态多Token Prefill并验证Tensor Cache衔接Decode，尚未统一物理状态ABI；
- 仍使用连续 KV Buffer，尚未实现 Paged KV 和 Block Table；
- 两层小配置仍作为快速回归测试；真实模型脚本目前是首轮执行验证，不是稳定态
  Tokens/s Benchmark。
