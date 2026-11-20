"""Compose cross-Warp sums with explicit UB scratch and Thread VF barriers."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import check_kernel_binary, compile_offline, get_stages, lower_offline


@tla.jit
def cross_warp_sum(X, Y, N, ROUNDS, WARPS: tl.constexpr):
    scratch = tla.alloc((WARPS, ), X.dtype.element_ty)
    with tla.scope.vector():
        with tla.scope.simt(num_warps=WARPS, mode="thread"):
            p = tla.simt.to_ptr(scratch)
            tid = tla.simt.thread_id()
            lane = tid % 32
            warp = tid // 32
            program = tl.program_id(0)
            for iteration in range(ROUNDS):
                row = program * ROUNDS + iteration
                value = tl.load(X + row * (WARPS * 32) + tid, mask=tid < N, other=0)
                partial = tla.simt.warp_reduce_sum(value)
                if lane == 0:
                    tl.store(p + warp, partial)
                tla.simt.sync_threads()
                if warp == 0:
                    total = tl.full((), 0, X.dtype.element_ty)
                    # Also handles more than 32 partials without a Tile Tensor.
                    for index in range(lane, WARPS, 32):
                        total += tl.load(p + index)
                    result = tla.simt.warp_reduce_sum(total)
                    if lane == 0:
                        tl.store(Y + row, result)
                # Every reader must finish before the next iteration overwrites p.
                tla.simt.sync_threads()


CASES = [
    ("fp16", 4, 113, 1, 1),
    ("fp32", 1, 29, 1, 1),
    ("i32", 64, 2031, 1, 1),
    ("fp32", 4, 117, 3, 3),
]


def _torch_dtype(name):
    import torch
    return {"fp16": torch.float16, "fp32": torch.float32, "i32": torch.int32}[name]


def test_ir():
    signature = {"X": "*i32", "Y": "*i32", "N": "i32", "ROUNDS": "i32"}
    module, ctx = compile_offline(cross_warp_sum, signature, {"WARPS": 64})
    module = lower_offline(module, ctx)
    text = str(module)
    assert module.verify()
    assert text.count("memref.alloc") == 1 and "memref<64xi32" in text
    assert "!tt.ptr<i32, 6>" in text
    assert text.count("ascend_dpx.reduce.add") == 2
    assert text.count("gpu.barrier") == 2
    assert "scf.for" in text and "tensor<" not in text
    ops = []
    module.walk(ops.append)
    scope, = [op for op in ops if op.get_name() == "scope.scope" and op.get_str_attr("vector_mode") == "simt"]
    allocation, = [op for op in ops if op.get_name() == "memref.alloc"]
    assert allocation.get_block().get_parent().id() != scope.get_region(0).id()


@pytest.mark.skipif(not get_stages()[1], reason="target compile stage disabled")
def test_compile():
    import torch
    import torch_npu  # noqa: F401

    x = torch.empty(2048, dtype=torch.int32, device="npu")
    y = torch.empty(1, dtype=torch.int32, device="npu")
    check_kernel_binary(cross_warp_sum.warmup(x, y, 2031, 1, 64, grid=(1, ), num_warps=8))


@pytest.mark.skipif(not get_stages()[2], reason="device run stage disabled")
@pytest.mark.parametrize("dtype,warps,valid,rounds,programs", CASES)
def test_run(dtype, warps, valid, rounds, programs):
    import torch
    import torch_npu  # noqa: F401

    threads = warps * 32
    index = torch.arange(programs * rounds * threads, dtype=torch.int32)
    source = ((index * 5) % 13 - 6).reshape(programs, rounds, threads).to(_torch_dtype(dtype))
    x = source.to("npu")
    # Extra elements detect unintended writeback by another lane or program.
    y = torch.full((programs * rounds + 3, ), -1024, dtype=source.dtype, device="npu")
    cross_warp_sum[(programs, )](x, y, valid, rounds, warps, num_warps=8)
    torch.npu.synchronize()

    expected = torch.full((programs * rounds + 3, ), -1024, dtype=source.dtype)
    expected[:programs * rounds] = source[:, :, :valid].sum(dim=2).to(source.dtype).reshape(-1)
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
