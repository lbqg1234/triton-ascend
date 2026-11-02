"""Root-Buffer to pointer contracts, without device execution."""
import pytest
import triton
import triton.language as tl
from triton.language.extra import tla

from utils.tla_test_utils import compile_offline, lower_offline


def _lower(kernel, signature, constants=None):
    module, ctx = compile_offline(kernel, signature, constants or {})
    return lower_offline(module, ctx)


def _ops(module, name):
    ops = []
    module.walk(lambda op: ops.append(op) if op.get_name() == name else None)
    return ops


def test_public_api():
    assert tla.simt.__all__ == ["to_ptr"]
    assert not hasattr(tla.simt, "load") and not hasattr(tla.simt, "store")


@triton.jit
def root_helper(buf, offset, stride, valid):
    p = tla.simt.to_ptr(buf)
    i = tl.arange(0, 32)
    values = tl.load(p + offset + i * stride, mask=i < valid, other=0)
    tl.store(p + offset + i * stride, values + 2, mask=i < valid)


@tla.jit
def root_kernel(X, Y, offset, stride, valid, DTYPE: tl.constexpr, MATRIX: tl.constexpr, HELPER: tl.constexpr):
    if MATRIX:
        buf = tla.alloc((3, 32), DTYPE)
    else:
        buf = tla.alloc((96,), DTYPE)
    tla.copy(tla.view(X, buf.shape), buf)
    with tla.scope.simt():
        if HELPER:
            root_helper(buf, offset, stride, valid)
        else:
            p = tla.simt.to_ptr(buf)
            i = tl.arange(0, 32)
            values = tl.load(p + offset + i * stride, mask=i < valid, other=0)
            tl.store(p + offset + i * stride, values + 2, mask=i < valid)
    tla.copy(buf, tla.view(Y, buf.shape))


@pytest.mark.parametrize("dtype,typ", [(tl.int8, "i8"), (tl.int32, "i32"), (tl.float16, "fp16"),
                                      (tl.bfloat16, "bf16"), (tl.float32, "fp32")])
@pytest.mark.parametrize("matrix,helper", [(False, False), (True, False), (True, True)])
def test_root_buffer_preserves_type_and_needs_no_tile_shape(dtype, typ, matrix, helper):
    module = _lower(root_kernel, {"X": "*" + typ, "Y": "*" + typ, "offset": "i32", "stride": "i32", "valid": "i32"},
                    {"DTYPE": dtype, "MATRIX": matrix, "HELPER": helper})
    assert len(_ops(module, "memref.alloc")) == 1
    assert len(_ops(module, "tt.load")) == len(_ops(module, "tt.store")) == 1
    assert not _ops(module, "tt.call")
    assert not _ops(module, "memref.extract_strided_metadata")
    assert "tla.simt_buffer" not in str(module)
    scope, = [op for op in _ops(module, "scope.scope") if op.get_str_attr("vector_mode") == "simt"]
    bridge, = _ops(module, "builtin.unrealized_conversion_cast")
    assert bridge.get_block().get_parent().id() == scope.get_region(0).id()
    assert ", 6>" in str(module)


@triton.jit
def pointer_helper(p, Y):
    i = tl.arange(0, 32)
    tl.store(Y + i, tl.load(p + i, mask=None, other=None), mask=None)


@tla.jit
def pointer_helper_kernel(X, Y):
    buf = tla.alloc((32,), tl.float32)
    tla.copy(tla.view(X, (32,)), buf)
    with tla.scope.simt():
        pointer_helper(tla.simt.to_ptr(buf), Y)


def test_returned_pointer_can_be_passed_to_helper():
    module = _lower(pointer_helper_kernel, {"X": "*fp32", "Y": "*fp32"})
    assert not _ops(module, "tt.call")
    assert len(_ops(module, "tt.load")) == len(_ops(module, "tt.store")) == 1


@triton.jit
def convert(buf):
    return tla.simt.to_ptr(buf)


@tla.jit
def invalid_view(X, offset, KIND: tl.constexpr, HELPER: tl.constexpr):
    buf = tla.alloc((64,), tl.float32)
    if KIND == "zero_subview":
        buf = tla.subview(buf, (0,), (32,))
    elif KIND == "offset_subview":
        buf = tla.subview(buf, (3,), (32,))
    elif KIND == "dynamic_subview":
        buf = tla.subview(buf, (offset,), (32,))
    elif KIND == "reshape":
        buf = tla.reshape(buf, (2, 32))
    elif KIND == "gm_view":
        buf = tla.view(X, (64,))
    with tla.scope.simt():
        if HELPER:
            p = convert(buf)
        else:
            p = tla.simt.to_ptr(buf)
        tl.store(p, 1.0)


@pytest.mark.parametrize("kind", ["zero_subview", "offset_subview", "dynamic_subview", "reshape", "gm_view"])
@pytest.mark.parametrize("helper", [False, True])
def test_views_are_rejected_even_when_their_offset_is_zero(kind, helper, capfd):
    try:
        _lower(invalid_view, {"X": "*fp32", "offset": "i32"}, {"KIND": kind, "HELPER": helper})
    except Exception as exc:
        diagnostics = capfd.readouterr().err
        while exc is not None:
            diagnostics += str(exc)
            exc = exc.__cause__
        assert "root Buffer" in diagnostics
    else:
        pytest.fail("view unexpectedly accepted")


@tla.jit
def invalid_input(X, KIND: tl.constexpr):
    buf = tla.alloc((32,), tl.float32, space="l1" if KIND == "space" else "ub")
    with tla.scope.simt():
        if KIND == "pointer":
            p = tla.simt.to_ptr(X)
        elif KIND == "scalar":
            p = tla.simt.to_ptr(1)
        else:
            p = tla.simt.to_ptr(buf)


@pytest.mark.parametrize("kind,message", [("space", "GM and UB"), ("pointer", "root tla.buffer"),
                                          ("scalar", "root tla.buffer")])
def test_invalid_inputs(kind, message):
    with pytest.raises(Exception, match=message):
        compile_offline(invalid_input, {"X": "*fp32"}, {"KIND": kind})


@tla.jit
def outside_vf(HELPER: tl.constexpr, SIMD: tl.constexpr):
    buf = tla.alloc((32,), tl.float32)
    if SIMD:
        with tla.scope.simd():
            p = convert(buf)
    elif HELPER:
        p = convert(buf)
    else:
        p = tla.simt.to_ptr(buf)


@pytest.mark.parametrize("helper,simd", [(False, False), (True, False), (True, True)])
def test_to_ptr_requires_simt_vf(helper, simd):
    with pytest.raises(Exception) as caught:
        compile_offline(outside_vf, {}, {"HELPER": helper, "SIMD": simd})
    exc = caught.value
    diagnostics = ""
    while exc is not None:
        diagnostics += str(exc)
        exc = exc.__cause__
    assert "must be called inside SIMT VF" in diagnostics


@tla.jit
def allocation_inside_vf(Y):
    with tla.scope.simt():
        buf = tla.alloc((32,), tl.float32)
        tl.store(tla.simt.to_ptr(buf), 1.0)


def test_root_must_be_allocated_outside_vf(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _lower(allocation_inside_vf, {"Y": "*fp32"})
    assert "allocated outside SIMT VF" in capfd.readouterr().err


@tla.jit
def loop_and_branch(X, Y, count, flag):
    buf = tla.alloc((256,), tl.float32)
    tla.copy(tla.view(X, (256,)), buf)
    for step in range(count):
        if flag:
            with tla.scope.simt():
                p = tla.simt.to_ptr(buf)
                i = step + 2 * tl.arange(0, 32)
                tl.store(p + i, tl.load(p + i) + 1.0)
    tla.copy(buf, tla.view(Y, (256,)))


def test_external_control_flow_keeps_root_capture_dominance():
    module = _lower(loop_and_branch, {"X": "*fp32", "Y": "*fp32", "count": "i32", "flag": "i1"})
    assert module.verify()
    assert len(_ops(module, "scf.for")) == len(_ops(module, "scf.if")) == 1
    assert len(_ops(module, "memref.alloc")) == 1
