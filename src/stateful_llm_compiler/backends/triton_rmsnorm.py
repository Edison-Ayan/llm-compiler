"""`serve.rms_norm` 的 Triton Lowering。

这一文件负责把 ServeIR 中的 `serve.rms_norm` 操作，
Lowering 成可以在 NVIDIA GPU 上执行的 Triton Kernel。

RMSNorm 的计算公式：

    mean_square = sum(x_i^2) / hidden_size
    inverse_rms = 1 / sqrt(mean_square + epsilon)
    output_i = x_i * inverse_rms * weight_i

与 LayerNorm 不同，RMSNorm 不会计算和减去输入均值。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    row_stride,
    hidden_size: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    """一个 Triton Program 处理一行 Hidden State。

    假设输入张量最终被看作二维矩阵：

        [rows, hidden_size]

    每个 Triton Program 负责其中一行。

    参数：
        input_pointer:
            输入张量首地址。

        weight_pointer:
            RMSNorm 可学习权重的首地址。
            Weight 的形状应为 [hidden_size]。

        output_pointer:
            输出张量首地址。

        row_stride:
            相邻两行之间相隔的元素数量。
            对于连续张量，它通常等于 hidden_size。

        hidden_size:
            每行真实有效的元素数量。
            它是编译期常量。

        epsilon:
            防止除零的稳定项。
            它是编译期常量。

        block_size:
            Triton 实际处理的向量长度。
            通常是大于等于 hidden_size 的最小 2 的幂。
    """

    # 获取当前 Triton Program 在第 0 维 Grid 中的编号。
    #
    # Kernel 的启动 Grid 是：
    #
    #     (rows,)
    #
    # 因此：
    #
    #     program_id = 0 处理第 0 行
    #     program_id = 1 处理第 1 行
    #     ...
    row = tl.program_id(0)

    # 创建当前 Program 要处理的列索引。
    #
    # 例如：
    #
    #     hidden_size = 1536
    #     block_size = 2048
    #
    # offsets 就是：
    #
    #     [0, 1, 2, ..., 2047]
    offsets = tl.arange(0, block_size)

    # 标记哪些位置是真实有效的 Hidden State 元素。
    #
    # 对于 hidden_size=1536、block_size=2048：
    #
    #     offsets 0~1535    对应 True
    #     offsets 1536~2047 对应 False
    #
    # Mask 用于避免越界访问。
    mask = offsets < hidden_size

    # 计算当前行的起始地址。
    #
    # 例如 row_stride=1536：
    #
    #     row=0 时，起始位置为 input_pointer + 0
    #     row=1 时，起始位置为 input_pointer + 1536
    #     row=2 时，起始位置为 input_pointer + 3072
    input_row = input_pointer + row * row_stride

    # 从全局显存加载当前行的数据。
    #
    # mask=False 的位置不会真正访问显存，
    # 而是使用 other=0.0 作为填充值。
    #
    # 补零不会影响后续平方和：
    #
    #     x^2 + 0^2 = x^2
    #
    # 加载后统一转换为 FP32，
    # 避免在 FP16/BF16 下进行长向量归约时产生较大误差。
    values = tl.load(
        input_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # 计算一行元素的平方和：
    #
    #     sum(x_i^2)
    #
    # tl.sum(..., axis=0) 会把当前 Program 中的整个向量
    # 归约成一个标量。
    square_sum = tl.sum(values * values, axis=0)

    # 计算平方均值：
    #
    #     mean_square = sum(x_i^2) / hidden_size
    #
    # 注意这里除以的是 hidden_size，而不是 block_size。
    #
    # block_size 中超过 hidden_size 的部分只是补零区域，
    # 不属于真实输入。
    mean_square = square_sum / hidden_size

    # 计算均方根的倒数：
    #
    #     inverse_rms =
    #         1 / sqrt(mean_square + epsilon)
    #
    # tl.rsqrt(x) 等价于 1 / sqrt(x)。
    inverse_rms = tl.rsqrt(mean_square + epsilon)

    # 加载 RMSNorm 的逐通道缩放权重。
    #
    # Weight 的形状为：
    #
    #     [hidden_size]
    #
    # 每一行输入都使用相同的一组 Weight。
    #
    # Weight 同样转换成 FP32参与计算。
    weights = tl.load(
        weight_pointer + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # 执行 RMSNorm：
    #
    #     output_i =
    #         input_i * inverse_rms * weight_i
    #
    # inverse_rms 是一个标量，
    # 会广播到当前行的所有元素。
    normalized = values * inverse_rms * weights

    # 计算当前输出行的起始地址。
    output_row = output_pointer + row * row_stride

    # 将归一化结果写回全局显存。
    #
    # 只写入 offsets < hidden_size 的有效位置。
    #
    # 如果 output 的 dtype 是 FP16 或 BF16，
    # Triton 会在 store 时把 FP32 normalized 转换成对应类型。
    tl.store(
        output_row + offsets,
        normalized,
        mask=mask,
    )


def triton_rms_norm(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float,
    output_dtype: str | torch.dtype | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """执行输入最后一维上的 RMSNorm。

    输入的前导维度保持不变。

    例如：

        输入 Shape:  [batch, sequence, hidden_size]
        输出 Shape:  [batch, sequence, hidden_size]

    Triton Kernel 会在逻辑上把输入展平为：

        [rows, hidden_size]

    其中：

        rows = tensor.numel() // hidden_size

    参数：
        tensor:
            输入 Tensor，必须位于 CUDA 设备。

        weight:
            RMSNorm 权重，必须是一维 Tensor，
            形状为 [hidden_size]。

        epsilon:
            RMSNorm 的数值稳定项。

        output_dtype:
            输出数据类型。

            支持：
                None
                "f16"
                "bf16"
                "f32"
                torch.float16
                torch.bfloat16
                torch.float32

            如果为 None，输出类型与输入类型相同。

        num_warps:
            每个 Triton Program 使用的 Warp 数量。

            如果不手工指定，就根据 block_size 和 rows
            使用启发式规则自动选择 4 或 8。

    返回：
        与输入 Shape 相同的 RMSNorm 输出 Tensor。
    """

    # 当前 Triton 实现要求输入和 Weight 都在 NVIDIA GPU 上。
    if not tensor.is_cuda or not weight.is_cuda:
        raise ValueError("Triton RMSNorm 要求输入和权重位于 CUDA")

    # tensor 至少要有一个维度。
    #
    # 标量 Tensor 的 Shape 是 []，
    # 没有最后一维可以执行 RMSNorm。
    if tensor.ndim < 1:
        raise ValueError("Triton RMSNorm 要求 Tensor 至少有一个维度")

    # RMSNorm 在输入的最后一个维度上执行。
    #
    # 例如：
    #
    #     tensor.shape = [2, 128, 4096]
    #
    # 则：
    #
    #     hidden_size = 4096
    hidden_size = tensor.shape[-1]

    # Weight 必须是一维向量，并且长度必须等于 hidden_size。
    #
    # 合法：
    #
    #     tensor.shape = [2, 128, 4096]
    #     weight.shape = [4096]
    #
    # 不合法：
    #
    #     weight.shape = [1, 4096]
    #     weight.shape = [2048]
    if weight.ndim != 1 or weight.numel() != hidden_size:
        raise ValueError(
            f"Weight 应为 [{hidden_size}]，实际为 {tuple(weight.shape)}"
        )

    # 当前 Kernel 使用“一个 Triton Program 处理一整行”的方式。
    #
    # hidden_size 太大时会产生很高的：
    #
    #     寄存器压力
    #     归约成本
    #     编译成本
    #
    # 因此暂时设置一个上限。
    if hidden_size > 65536:
        raise ValueError(
            "当前 Triton RMSNorm 只支持 hidden_size <= 65536"
        )

    # 当前 Kernel 使用简单的连续地址计算：
    #
    #     input_pointer + row * row_stride + offset
    #
    # 因此需要保证输入和 Weight 的内存布局连续。
    #
    # 如果原 Tensor 已经连续，不会产生复制；
    # 如果不连续，则会创建一份连续副本。
    tensor = tensor.contiguous()
    weight = weight.contiguous()

    # 根据 output_dtype 解析最终输出类型。
    #
    # 如果 output_dtype=None，则默认与输入类型相同。
    dtype = _resolve_dtype(
        output_dtype,
        tensor.dtype,
    )

    # 创建输出 Tensor。
    #
    # Shape 与输入完全一致；
    # Device 与输入相同；
    # DType 由上面的 dtype 决定。
    #
    # 使用 torch.empty 是因为 Kernel 会写入全部有效元素，
    # 不需要提前初始化。
    output = torch.empty(
        tensor.shape,
        device=tensor.device,
        dtype=dtype,
    )

    # 把所有前导维度展平成行数。
    #
    # 例如：
    #
    #     tensor.shape = [2, 128, 4096]
    #
    # 则：
    #
    #     rows = 2 * 128 = 256
    rows = tensor.numel() // hidden_size

    # Triton 通常使用 2 的幂大小处理向量和归约。
    #
    # 例如：
    #
    #     hidden_size = 1536
    #     block_size = 2048
    #
    #     hidden_size = 4096
    #     block_size = 4096
    #
    # 超出 hidden_size 的部分由 Kernel 中的 mask 屏蔽。
    block_size = triton.next_power_of_2(hidden_size)

    # 如果调用者没有指定 num_warps，
    # 使用当前根据实验结果编写的启发式规则。
    if num_warps is None:
        # 对于较窄的 Hidden State：
        #
        #     block_size < 2048
        #
        # 一行的数据量不大，使用 4 个 Warp 通常已经足够。
        if block_size < 2048:
            num_warps = 4

        # 对于较宽的 Hidden State，
        # 但总行数较少的情况：
        #
        #     4 <= rows <= 16
        #
        # 根据实测结果，使用 4 个 Warp 可能比 8 个 Warp 更好，
        # 因为它能够降低线程协调和寄存器压力。
        elif 4 <= rows <= 16:
            num_warps = 4

        # 其他较宽、行数较多的情况使用 8 个 Warp，
        # 提高单行内部的并行度。
        else:
            num_warps = 8

    # 启动 Triton Kernel。
    #
    # Grid：
    #
    #     (rows,)
    #
    # 表示一共启动 rows 个 Triton Program，
    # 每个 Program 处理一行 Hidden State。
    _rms_norm_kernel[(rows,)](
        # input_pointer
        tensor,

        # weight_pointer
        weight,

        # output_pointer
        output,

        # row_stride
        #
        # 对于 ndim > 1 的连续张量：
        #
        #     tensor.stride(-2) == hidden_size
        #
        # 对于一维输入，不存在倒数第二维，
        # 因此直接使用 hidden_size。
        tensor.stride(-2)
        if tensor.ndim > 1
        else hidden_size,

        # 以下三个参数标记为 tl.constexpr，
        # Triton 会针对它们的具体值生成专门的 Kernel。
        hidden_size=hidden_size,
        epsilon=epsilon,
        block_size=block_size,

        # num_warps 是 Triton Kernel 启动配置，
        # 表示每个 Program 使用多少个 Warp。
        num_warps=num_warps,
    )

    return output


def _resolve_dtype(
    value: str | torch.dtype | None,
    default: torch.dtype,
) -> torch.dtype:
    """将字符串或 torch.dtype 转换成统一的 torch.dtype。

    参数：
        value:
            用户指定的输出数据类型。

        default:
            当 value=None 时使用的默认类型。

    返回：
        对应的 torch.dtype。
    """

    # 没有显式指定输出类型，
    # 则使用默认值，通常也就是输入 Tensor 的 dtype。
    if value is None:
        return default

    # 如果调用者直接传入的是 torch.dtype，
    # 则不需要进一步转换。
    if isinstance(value, torch.dtype):
        return value

    # ServeIR 风格的 dtype 字符串到 PyTorch dtype 的映射。
    mapping = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
    }

    try:
        # 根据字符串查找对应的 torch.dtype。
        return mapping[value]

    except KeyError as error:
        # 如果传入未支持的字符串，例如：
        #
        #     "fp8"
        #     "int8"
        #
        # 则抛出更容易理解的 ValueError，
        # 同时保留原始 KeyError 作为异常原因。
        raise ValueError(
            f"不支持的输出 DType：{value}"
        ) from error