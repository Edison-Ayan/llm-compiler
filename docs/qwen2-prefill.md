# Qwen2ForCausalLM 动态 Prefill 前端

## 完整模型边界

本阶段把此前只接收Input Embedding的Decoder扩展为：

```text
input_ids: [B, T]
    ↓ Embedding
hidden_states: [B, T, H]
    ↓ 多层Qwen2 Decoder Prefill
hidden_states + 每层K/V: [B, KVH, T, D]
    ↓ LM Head
logits: [B, T, vocab_size]
```

`StatefulQwen2ForCausalLM`同时提供：

- `prefill`：从空状态处理完整Prompt；
- `decode`：消费Prefill产生的Cache处理下一个Token；
- `forward`：默认指向Prefill，供`torch.export`捕获完整图。

官方Qwen2ForCausalLM的Embedding、Decoder、Final Norm和LM Head权重都会被复制。对于
`tie_word_embeddings=True`的配置，项目模型也会共享Embedding和LM Head参数。

## 为什么不在图里创建空Cache

一种简单实现是先创建长度为0的K/V，再复用Decode代码：

```text
empty = new_empty([B, KVH, 0, D])
present = cat(empty, current, dim=2)
```

实验表明，两层模型会因此产生4个`aten.new_empty`和4个无意义的空拼接。项目改为在
Attention中显式区分Prefill语义：没有历史状态时，当前K/V直接成为首个Cache版本。

因此最终导出图：

- 不包含`new_empty`；
- 不包含Cache空拼接；
- 仍保留RoPE内部必要的`cat`；
- K/V直接作为函数结果，可继续进入状态化Pass。

## 动态Shape契约

```text
input_ids:      [batch, prompt_tokens]
attention_mask: [batch, 1, prompt_tokens, prompt_tokens]
position_ids:   [batch, prompt_tokens]

1 <= batch <= max_batch
2 <= prompt_tokens <= max_prompt_length
```

Attention Mask使用四维加性Causal Mask：未来Token位置为负无穷，其余位置为0。
Prefill最少两个Token；单Token路径由Decode前端负责。

## 图规模

两层、Hidden Size 32、词表128的测试模型：

```text
导入ServeIR：                         193
规范化15个Linear并融合5个RMSNorm、
2个双结果RoPE和2个Prefill Attention： 75
Lower 15个Linear、5个RMSNorm、
2个RoPE和2个Attention：               75
```

当前KernelIR覆盖率：

```text
已Lower： 24
未Lower： 51
总操作：  75
覆盖率： 32.000%
```

多TokenAttention已经融合为两个`kernel.triton.prefill_attention`。剩余缺口主要是
Embedding、SwiGLU、RoPE系数生成、Cast和元数据操作。

## 正确性闭环

当前测试完成四层差分：

1. 项目CausalLM与官方Hugging Face模型：Logits误差0；
2. 两层Prefill K/V与官方DynamicCache：误差0；
3. 优化后ServeIR在Prompt长度2、4、7下：所有输出误差0；
4. Prefill Cache输入下一次Decode：下一Token Logits误差0。

GPU KernelIR路径实际执行15个`kernel.triton.linear`、5个
`kernel.triton.rms_norm`、2个`kernel.triton.rope`和2个
`kernel.triton.prefill_attention`，其余操作暂时由
兼容执行器承接，整体通过`3e-5`差分。

## 使用方式

```python
from stateful_llm_compiler.frontend import export_qwen2_causal_lm_prefill
from stateful_llm_compiler.qwen2 import (
    StatefulQwen2ForCausalLM,
    make_qwen2_prefill_inputs,
)

model = StatefulQwen2ForCausalLM.from_huggingface(huggingface_model)
inputs = make_qwen2_prefill_inputs(model.config, batch=2, tokens=8)
program = export_qwen2_causal_lm_prefill(
    model,
    inputs,
    max_batch=8,
    max_prompt_length=128,
)
```

## 下一步

多TokenAttention融合和Prefill状态化已经完成。每层K/V现在可以直接批量写入
`PreallocatedKVCacheState`，并由完整CausalLM Decode在相同物理地址上继续追加。
下一阶段开始清除Embedding、SwiGLU和RoPE系数生成回退。
