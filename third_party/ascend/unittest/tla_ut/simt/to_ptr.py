"""SIMT root-Buffer pointer conversion with explicit offsets and strides."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import check_kernel_binary, compile_offline, get_stages, lower_offline

N = 256
SIGNATURE = {"X": "*fp32", "Y": "*fp32"}


@tla.jit
def root_pointer_load_store(X, Y):
    ub = tla.alloc_buffer((256,), tl.float32, space="ub")
    x_gm = tla.view(X, (256,), strides=(1,))
    y_gm = tla.view(Y, (256,), strides=(1,))
    with tla.scope.vector():
        tla.copy(x_gm, ub)
        with tla.scope.simt():
            i = 1 + 2 * tl.arange(0, 128)
            p = tla.simt.to_ptr(ub)
            a = tl.load(p + i)
            b = tl.load(X + i)
            tl.store(p + i, a + b + 1.0)
        tla.copy(ub, y_gm)


def test_ir(output_dir=None):
    module, ctx = compile_offline(root_pointer_load_store, SIGNATURE, {}, output_dir=output_dir)
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    assert text.count("memref.alloc") == 1
    assert text.count("tt.load") == 2 and text.count("tt.store") == 1
    assert "!tt.ptr<f32, 6>" in text
    assert "tla.simt_buffer" not in text
    assert "memref.extract_strided_metadata" not in text
    ops = []
    module.walk(ops.append)
    scope, = [op for op in ops if op.get_name() == "scope.scope" and op.get_str_attr("vector_mode") == "simt"]
    for op in ops:
        if op.get_name() in ("memref.subview", "memref.alloc"):
            assert op.get_block().get_parent().id() != scope.get_region(0).id()
    assert "tt.addptr" in text
    assert "memref.subview" not in text
    print("IR PASS: one explicit UB, root Buffer capture, pointer offset/stride inside SIMT")


def test_compile():
    import torch
    import torch_npu  # noqa: F401

    x = torch.empty(N, dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    check_kernel_binary(root_pointer_load_store.warmup(x, y, grid=(1,), num_warps=4))


def test_run():
    import torch
    import torch_npu  # noqa: F401

    x = torch.arange(N, dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    root_pointer_load_store[(1,)](x, y, num_warps=4)
    torch.npu.synchronize()
    expected = x.cpu().clone()
    expected[1::2] = 2 * expected[1::2] + 1
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@triton.jit
def masked_update(buf, offset, stride, valid):
    p = tla.simt.to_ptr(buf)
    i = tl.arange(0, 128)
    ptrs = p + offset + i * stride
    value = tl.load(ptrs, mask=i < valid, other=-99.0)
    tl.store(ptrs, value + 3.0, mask=i < valid)


@tla.jit
def dynamic_masked_load_store(X, Y, offset, stride, valid):
    ub = tla.alloc((512,), tl.float32)
    with tla.scope.vector():
        tla.copy(tla.view(X, (512,)), ub)
        with tla.scope.simt():
            masked_update(ub, offset, stride, valid)
        tla.copy(ub, tla.view(Y, (512,)))


@pytest.mark.skipif(not get_stages()[2], reason="device run stage disabled")
@pytest.mark.parametrize("offset,stride,valid", [(1, 2, 0), (3, 2, 77), (17, 2, 128), (2, 3, 117)])
def test_run_dynamic_mask_and_helper(offset, stride, valid):
    import torch
    import torch_npu  # noqa: F401

    x = torch.arange(512, dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    dynamic_masked_load_store[(1,)](x, y, offset, stride, valid, num_warps=4)
    torch.npu.synchronize()
    expected = x.cpu().clone()
    expected[offset:offset + stride * valid:stride] += 3.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@tla.jit
def static_masked_load_store(X, Y, valid: tl.constexpr):
    ub = tla.alloc((512,), tl.float32)
    with tla.scope.vector():
        tla.copy(tla.view(X, (512,)), ub)
        with tla.scope.simt():
            masked_update(ub, 3, 2, valid)
        tla.copy(ub, tla.view(Y, (512,)))


@pytest.mark.skipif(not get_stages()[2], reason="device run stage disabled")
@pytest.mark.parametrize("valid", [0, 77, 128])
def test_run_static_mask(valid):
    import torch
    import torch_npu  # noqa: F401

    x = torch.arange(512, dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    static_masked_load_store[(1,)](x, y, valid, num_warps=4)
    torch.npu.synchronize()
    expected = x.cpu().clone()
    expected[3:3 + 2 * valid:2] += 3.0
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@tla.jit
def scalar_pointer_load_store(X, Y):
    ub = tla.alloc((128,), tl.float32)
    with tla.scope.vector():
        tla.copy(tla.view(X, (128,)), ub)
        with tla.scope.simt():
            p = tla.simt.to_ptr(ub)
            tl.store(p + 9, tl.load(p + 7) + 5.0)
        tla.copy(ub, tla.view(Y, (128,)))


@pytest.mark.skipif(not get_stages()[2], reason="device run stage disabled")
def test_run_scalar_pointer():
    import torch
    import torch_npu  # noqa: F401

    x = torch.arange(128, dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    scalar_pointer_load_store[(1,)](x, y, num_warps=4)
    torch.npu.synchronize()
    expected = x.cpu().clone()
    expected[9] = expected[7] + 5
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@tla.jit
def two_roots(X, Y):
    first = tla.alloc((128,), tl.float32)
    second = tla.alloc((128,), tl.float32)
    with tla.scope.vector():
        source = tla.view(X, (128,))
        tla.copy(source, first)
        tla.copy(source, second)
        with tla.scope.simt():
            i = tl.arange(0, 128)
            p = tla.simt.to_ptr(first)
            q = tla.simt.to_ptr(second)
            tl.store(q + i, tl.load(p + i) + tl.load(q + i) + 5.0)
        tla.copy(second, tla.view(Y, (128,)))


@pytest.mark.skipif(not get_stages()[2], reason="device run stage disabled")
def test_run_distinct_root_allocations():
    import torch
    import torch_npu  # noqa: F401

    x = torch.arange(128, dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    two_roots[(1,)](x, y, num_warps=4)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), 2 * x.cpu() + 5, rtol=0, atol=0)


if __name__ == "__main__":
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
