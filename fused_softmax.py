import torch
import triton
import triton.language as tl

properties = triton.runtime.driver.active.utils.get_device_properties(0)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]


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

    if block_size <= 1024:
        num_warps = 4
    elif block_size <= 2048:
        num_warps = 8
    else:
        num_warps = 16

    grid = (n_rows, 1, 1)

    _softmax_kernel[grid](
        x,
        output,
        x.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return output


@triton.jit
def _softmax_kernel(
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
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(output_ptr + pid * output_row_stride + col_offsets, softmax_output, mask=mask)


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
        args={"M": 4096},
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
