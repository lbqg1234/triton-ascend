"""Thread SIMT Warp sums: independent Warps, native dtype, and all-lane results."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import check_kernel_binary, compile_offline, get_stages, lower_offline


@tla.jit
def warp_sum(X, Y, N, WARPS: tl.constexpr):
    with tla.scope.vector():
        with tla.scope.simt(num_warps=WARPS, mode="thread"):
            tid = tla.simt.thread_id()
            value = tl.load(X + tid, mask=tid < N, other=0)
            result = tla.simt.warp_reduce_sum(value)
            tl.store(Y + tid, result)


# Floating-point inputs make every intermediate exactly representable, including
# FP16, so the check is independent of the reduction tree. Int32 also tests wrap.
CASES = [("fp16", 1, 31), ("fp32", 4, 97), ("i32", 4, 127)]


def _torch_dtype(name):
    import torch
    return {"fp16": torch.float16, "fp32": torch.float32, "i32": torch.int32}[name]


def test_ir():
    module, ctx = compile_offline(warp_sum, {"X": "*fp32", "Y": "*fp32", "N": "i32"}, {"WARPS": 4})
    module = lower_offline(module, ctx)
    text = str(module)
    assert module.verify()
    assert text.count("ascend_dpx.reduce.add") == 1
    assert text.count("tt.load") == text.count("tt.store") == 1
    assert 'tla.simt_mode = "thread"' in text and "tla.simt_num_warps = 4" in text
    assert "tensor<" not in text and "memref.alloc" not in text


@pytest.mark.skipif(not get_stages()[1], reason="target compile stage disabled")
@pytest.mark.parametrize("dtype,warps,valid", CASES)
def test_compile(dtype, warps, valid):
    import torch
    import torch_npu  # noqa: F401

    x = torch.empty(warps * 32, dtype=_torch_dtype(dtype), device="npu")
    y = torch.empty_like(x)
    # Each VF selects its own Warp count independently of the kernel default.
    check_kernel_binary(warp_sum.warmup(x, y, valid, warps, grid=(1, ), num_warps=8))


@pytest.mark.skipif(not get_stages()[2], reason="device run stage disabled")
@pytest.mark.parametrize("dtype,warps,valid", CASES)
def test_run(dtype, warps, valid):
    import torch
    import torch_npu  # noqa: F401

    index = torch.arange(warps * 32, dtype=torch.int32)
    source = ((index * 3) % 17 - 8 + (index // 32) * 2).to(_torch_dtype(dtype))
    if dtype == "i32":
        # Force 32-bit wraparound without changing the expected dtype.
        source += 1 << 30
    x = source.to("npu")
    y = torch.empty_like(x)
    warp_sum[(1, )](x, y, valid, warps, num_warps=8)
    torch.npu.synchronize()

    masked = source.clone()
    masked[valid:] = 0
    expected = masked.reshape(warps, 32).sum(dim=1).to(source.dtype).repeat_interleave(32)
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
