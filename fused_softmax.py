import torch
import triton
import triton.language as tl

properties = triton.runtime.driver.active.utils.get_device_properties(0)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
_TUNED_LAUNCH_CONFIGS = {
    256: (2, 4),
    384: (1, 2),
    512: (1, 1),
    640: (1, 1),
    768: (1, 2),
    896: (1, 1),
    1024: (1, 1),
    1152: (1, 1),
    1280: (1, 2),
    1408: (1, 2),
    1536: (1, 2),
    1664: (1, 2),
    1792: (1, 2),
    1920: (1, 2),
    2048: (1, 2),
    2176: (1, 4),
    2304: (1, 8),
    2432: (1, 8),
    2560: (1, 4),
    2688: (1, 8),
    2816: (1, 4),
    2944: (1, 8),
    3072: (1, 8),
    3200: (1, 8),
    3328: (1, 8),
    3456: (1, 8),
    3584: (1, 8),
    3712: (1, 8),
    3840: (1, 8),
    3968: (1, 8),
    4096: (1, 8),
}


def test_softmax(size, atol=1e-6, rtol=1e-6):
    x = torch.randn(size).cuda()
    # y1 = naive_softmax(x)
    y1 = torch.softmax(x, dim=-1)
    y2 = softmax(x)

    torch.testing.assert_close(y1, y2, atol=atol, rtol=rtol)

    print("passed")


def softmax(x):
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 4096:
        raise ValueError(f"n_cols={n_cols} is too large for this kernel")
    rows_per_program, num_warps = _get_launch_config(n_cols, block_size)

    if rows_per_program == 1:
        _softmax_kernel_single[(n_rows,)](
            x,
            output,
            x.stride(0),
            output.stride(0),
            n_cols,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=2,
        )
    else:
        _softmax_kernel_multi[(triton.cdiv(n_rows, rows_per_program),)](
            x,
            output,
            x.stride(0),
            output.stride(0),
            n_rows,
            n_cols,
            ROWS_PER_PROGRAM=rows_per_program,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=2,
        )

    return output


def _get_launch_config(n_cols, block_size):
    config = _TUNED_LAUNCH_CONFIGS.get(n_cols)
    if config is not None:
        return config

    if block_size <= 512:
        return 1, 2
    if block_size <= 2048:
        return 1, 4
    return 1, 8


@triton.jit
def _softmax_kernel_single(
    x_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(x_ptr + pid * input_row_stride + col_offsets, mask=mask, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.math.exp2(row_minus_max * 1.4426950408889634)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(output_ptr + pid * output_row_stride + col_offsets, softmax_output, mask=mask)


@triton.jit
def _softmax_kernel_multi(
    x_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]
    mask = (row_offsets < n_rows) & (col_offsets < n_cols)

    row = tl.load(x_ptr + row_offsets * input_row_stride + col_offsets, mask=mask, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=1)[:, None]
    numerator = tl.math.exp2(row_minus_max * 1.4426950408889634)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = numerator / denominator

    tl.store(output_ptr + row_offsets * output_row_stride + col_offsets, softmax_output, mask=mask)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[128 * k for k in range(2, 33)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["triton", "torch"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="softmax-kernel",
        args={"M": 256},
    )
)
def benchmark_softmax(M, size, provider):

    x = torch.randn(M, size).cuda()

    quantiles = [0.5, 0.05, 0.95]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    # test_softmax(size=(1823, 781))

    benchmark_softmax.run(save_path=".", print_data=True)
