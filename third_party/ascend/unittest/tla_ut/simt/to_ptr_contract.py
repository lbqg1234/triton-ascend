"""UB Buffer to pointer contracts, without device execution."""
import pytest
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


@tla.jit
def buffer_kernel(offset, stride, valid, DTYPE: tl.constexpr, SHAPE: tl.constexpr):
    buf = tla.alloc(SHAPE, DTYPE)
    with tla.scope.simt():
        p = tla.simt.to_ptr(buf)
        i = tl.arange(0, 32)
        values = tl.load(p + offset + i * stride, mask=i < valid, other=0)
        tl.store(p + offset + i * stride, values + 2, mask=i < valid)


@pytest.mark.parametrize("dtype,element,shape", [
    (tl.int8, "i8", (96,)), (tl.int32, "i32", (96,)),
    (tl.float16, "f16", (96,)), (tl.bfloat16, "bf16", (96,)),
    (tl.float32, "f32", (3, 32)),
])
def test_buffer_preserves_type_and_needs_no_tile_shape(dtype, element, shape):
    module = _lower(buffer_kernel, {"offset": "i32", "stride": "i32", "valid": "i32"},
                    {"DTYPE": dtype, "SHAPE": shape})
    assert len(_ops(module, "memref.alloc")) == 1
    assert len(_ops(module, "tt.load")) == len(_ops(module, "tt.store")) == 1
    assert not _ops(module, "memref.extract_strided_metadata")
    scope, = _ops(module, "scope.scope")
    bridge, = _ops(module, "builtin.unrealized_conversion_cast")
    assert bridge.get_block().get_parent().id() == scope.get_region(0).id()
    assert f"!tt.ptr<{element}, 6>" in str(module)


@tla.jit
def viewed_buffer(X, KIND: tl.constexpr):
    buf = tla.alloc((64,), tl.float32)
    if KIND == "subview":
        buf = tla.subview(buf, (0,), (32,))
    elif KIND == "reshape":
        buf = tla.reshape(buf, (2, 32))
    elif KIND == "gm_view":
        buf = tla.view(X, (64,))
    with tla.scope.simt():
        tl.store(tla.simt.to_ptr(buf), 1.0)


@pytest.mark.parametrize("kind,message", [("subview", "known logical offset zero"),
                                         ("gm_view", "only supports UB Buffers")])
def test_unknown_offset_or_gm_views_are_rejected(kind, message):
    with pytest.raises(Exception, match=message):
        compile_offline(viewed_buffer, {"X": "*fp32"}, {"KIND": kind})


def test_identity_reshape_is_accepted():
    module = _lower(viewed_buffer, {"X": "*fp32"}, {"KIND": "reshape"})
    assert module.verify()
    assert len(_ops(module, "memref.alloc")) == len(_ops(module, "tt.store")) == 1


@tla.jit
def pointer_input(X):
    with tla.scope.simt():
        tla.simt.to_ptr(X)


def test_to_ptr_requires_buffer_input():
    with pytest.raises(Exception, match="expects a tla.buffer"):
        compile_offline(pointer_input, {"X": "*fp32"}, {})


@tla.jit
def outside_vf(SIMD: tl.constexpr):
    buf = tla.alloc((32,), tl.float32)
    if SIMD:
        with tla.scope.simd():
            p = tla.simt.to_ptr(buf)
    else:
        p = tla.simt.to_ptr(buf)


@pytest.mark.parametrize("simd", [False, True])
def test_to_ptr_requires_simt_vf(simd):
    with pytest.raises(Exception, match="must be called inside SIMT VF"):
        compile_offline(outside_vf, {}, {"SIMD": simd})


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


def test_external_control_flow_keeps_capture_dominance():
    module = _lower(loop_and_branch, {"X": "*fp32", "Y": "*fp32", "count": "i32", "flag": "i1"})
    assert module.verify()
    assert len(_ops(module, "scf.for")) == len(_ops(module, "scf.if")) == 1
    assert len(_ops(module, "memref.alloc")) == 1
