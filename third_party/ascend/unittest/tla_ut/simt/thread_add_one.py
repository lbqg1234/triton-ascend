"""Thread SIMT VF: per-thread FP32 GM access and a divergent scalar branch."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import check_kernel_binary, compile_offline, get_stages, lower_offline


@tla.jit
def thread_add_one(X, Y, N):
    with tla.scope.vector():
        with tla.scope.simt(num_warps=4, mode="thread"):
            tid = tla.simt.thread_id()
            if tid < N:
                x = tl.load(X + tid)
                tl.store(Y + tid, x + 1.0)


def test_ir():
    module, ctx = compile_offline(thread_add_one, {"X": "*fp32", "Y": "*fp32", "N": "i32"}, {})
    module = lower_offline(module, ctx)
    text = str(module)
    assert module.verify()
    assert "ascend_dpx.thread_id_x" in text
    assert "scf.if" in text
    assert text.count("tt.load") == text.count("tt.store") == 1
    assert 'tla.simt_mode = "thread"' in text
    assert "tla.simt_num_warps = 4" in text
    assert "tensor<" not in text
    assert "memref.alloc" not in text


@pytest.mark.skipif(not get_stages()[1], reason="target compile stage disabled")
def test_compile():
    import torch
    import torch_npu  # noqa: F401

    x = torch.empty(160, dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    # The VF's explicit four warps must take precedence over the kernel default.
    check_kernel_binary(thread_add_one.warmup(x, y, 127, grid=(1,), num_warps=8))


@pytest.mark.skipif(not get_stages()[2], reason="device run stage disabled")
@pytest.mark.parametrize("valid", [0, 1, 31, 32, 33, 127, 128, 160])
def test_run(valid):
    import torch
    import torch_npu  # noqa: F401

    source = torch.arange(160, dtype=torch.float32) + 0.25
    x = source.to("npu")
    y = torch.full((160,), -1024.0, dtype=torch.float32, device="npu")
    thread_add_one[(1,)](x, y, valid, num_warps=8)
    torch.npu.synchronize()
    expected = torch.full((160,), -1024.0, dtype=torch.float32)
    written = min(valid, 128)
    expected[:written] = source[:written] + 1.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
