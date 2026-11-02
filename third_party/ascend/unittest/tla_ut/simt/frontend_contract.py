"""TLA SIMT scope and affinity adapter contracts; no NPU launch."""
import pytest
import triton
import triton.language as tl
from triton.language.extra import tla
from triton._C.libtriton import ir
from triton.language.extra.tla.compiler import check_ttadapter
from utils.tla_test_utils import compile_offline, lower_offline


def _ops(module, name):
    result = []
    module.walk(lambda op: result.append(op) if op.get_name() == name else None)
    return result


def _lower(kernel, signature, constants=None):
    module, ctx = compile_offline(kernel, signature, constants or {})
    return lower_offline(module, ctx)


@tla.jit
def simt_copy(X, Y, BLOCK: tl.constexpr):
    with tla.scope.simt():
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(Y + offsets, tl.load(X + offsets))


@pytest.mark.parametrize("dtype,backing", [("i1", "i8"), ("i32", "i32"), ("fp16", "f16"), ("bf16", "bf16"),
                                           ("fp32", "f32")])
def test_simt_preserves_ttir_and_entry_pointer_abi(dtype, backing):
    module = _lower(simt_copy, {"X": "*" + dtype, "Y": "*" + dtype}, {"BLOCK": 32})
    text = str(module)
    assert f"memref<?x{backing}>" in text
    assert 'hacc.mode = "tla"' in text
    assert len(_ops(module, "tt.load")) == len(_ops(module, "tt.store")) == 1
    assert len(_ops(module, "tt.get_program_id")) == 1
    scope, = [op for op in _ops(module, "scope.scope") if op.get_str_attr("vector_mode") == "simt"]
    bridges = _ops(module, "builtin.unrealized_conversion_cast")
    assert len(bridges) == 2
    assert all(op.get_block().get_parent().id() == scope.get_region(0).id() for op in bridges)
    assert not _ops(module, "memref.alloc") and not _ops(module, "memref.load")


@tla.jit
def repeated_simt(X, Y):
    with tla.scope.vector():
        with tla.scope.simt():
            offsets = tl.arange(0, 32)
            tl.store(Y + offsets, tl.load(X + offsets))
        with tla.scope.simt():
            offsets = tl.arange(0, 32)
            tl.store(X + offsets, tl.load(Y + offsets))


def test_pointer_bridges_are_local_to_each_simt_vf():
    module = _lower(repeated_simt, {"X": "*fp32", "Y": "*fp32"})
    scopes = [op for op in _ops(module, "scope.scope") if op.get_str_attr("vector_mode") == "simt"]
    bridges = _ops(module, "builtin.unrealized_conversion_cast")
    assert len(scopes) == 2 and len(bridges) == 4
    for scope in scopes:
        assert sum(op.get_block().get_parent().id() == scope.get_region(0).id() for op in bridges) == 2
    assert not _ops(module, "memref.alloc")


@tla.jit
def simt_loop(X, Y, iterations):
    with tla.scope.simt():
        offsets = tl.arange(0, 32)
        values = tl.load(X + offsets)
        for i in range(iterations):
            values = values + 1.0
        tl.store(Y + offsets, values)


def test_simt_loop_carried_tensors_remain_in_scope():
    module = _lower(simt_loop, {"X": "*fp32", "Y": "*fp32", "iterations": "i32"})
    assert len(_ops(module, "scf.for")) == 1
    assert len(_ops(module, "tt.load")) == len(_ops(module, "tt.store")) == 1


@triton.jit
def add_one(values):
    return values + 1.0


@tla.jit
def helper_in_simt(X, Y):
    with tla.scope.simt():
        offsets = tl.arange(0, 32)
        tl.store(Y + offsets, add_one(tl.load(X + offsets)))


def test_jit_helper_is_inlined_before_affinity_conversion():
    module = _lower(helper_in_simt, {"X": "*fp32", "Y": "*fp32"})
    assert not _ops(module, "tt.call") and not _ops(module, "tt.func")
    assert len(_ops(module, "tt.load")) == len(_ops(module, "tt.store")) == 1


@tla.jit
def mixed_simd_simt(X, Y):
    local = tla.alloc((64, ), tl.float32)
    out = tla.view(Y, (64, ))
    with tla.scope.vector():
        scalar = tl.load(X)
        bounded = tla.clamp(tla.cast(scalar, tl.float32), -1.0, 1.0)
        with tla.scope.simd():
            tla.simd.vsts(local[0], tla.simd.vdup(bounded, tl.float32))
        tla.store_ub_to_gm(local, out)
        with tla.scope.simt():
            offsets = tl.arange(0, 64)
            tl.store(Y + offsets, tl.load(X + offsets) + 1.0)


def test_simt_preservation_keeps_existing_affinity_conversions():
    module = _lower(mixed_simd_simt, {"X": "*fp32", "Y": "*fp32"})
    assert len(_ops(module, "memref.load")) == 1
    assert len(_ops(module, "memref.alloc")) == 1
    assert len(_ops(module, "hivm.hir.store")) == 1
    assert len(_ops(module, "tt.load")) == len(_ops(module, "tt.store")) == 1
    assert not _ops(module, "tt.clampf") and not _ops(module, "tt.ptr_to_int")


@tla.jit
def tensor_capture(X, Y):
    offsets = tl.arange(0, 32)
    with tla.scope.simt():
        tl.store(Y + offsets, tl.load(X + offsets))


def test_non_constant_tensor_captures_are_rejected(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _lower(tensor_capture, {"X": "*fp32", "Y": "*fp32"})
    assert "cannot capture a non-constant Tensor" in capfd.readouterr().err


@tla.jit
def constant_capture(X, Y):
    ones = tl.full((32, ), 1.0, tl.float32)
    with tla.scope.simt():
        offsets = tl.arange(0, 32)
        tl.store(Y + offsets, tl.load(X + offsets) + ones)


def test_constant_tensor_captures_need_no_implicit_buffer():
    module = _lower(constant_capture, {"X": "*fp32", "Y": "*fp32"})
    assert not _ops(module, "memref.alloc")
    assert "tensor<32xf32>" in str(module)


def test_unconverted_main_module_ops_remain_invalid(tmp_path):
    module = _lower(simt_copy, {"X": "*fp32", "Y": "*fp32"}, {"BLOCK": 32})
    text = module.str_nodebug().replace("    return\n", "    %unconverted = tt.get_program_id x : i32\n    return\n", 1)
    assert "%unconverted" in text
    path = tmp_path / "unconverted-main.mlir"
    path.write_text(text)
    invalid = ir.parse_mlir_module(str(path), module.context)
    with pytest.raises(ValueError, match="unconverted TLA adapter operations.*tt.get_program_id"):
        check_ttadapter(invalid)


@tla.jit
def store_outside_simt(X, Y):
    with tla.scope.simt():
        offsets = tl.arange(0, 32)
        tl.store(Y + offsets, tl.load(X + offsets))
    tl.store(Y, 0.0)


def test_simt_scope_does_not_enable_main_module_tt_store():
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _lower(store_outside_simt, {"X": "*fp32", "Y": "*fp32"})
