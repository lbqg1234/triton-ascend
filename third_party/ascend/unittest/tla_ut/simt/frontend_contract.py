"""TLA SIMT scope and affinity adapter contracts; no NPU launch."""
import pytest
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


def _run_bridge_adapter(tmp_path, text):
    # Raw IR isolates compiler-owned bridge placement from frontend generation.
    _, ctx = compile_offline(simt_copy, {"X": "*fp32", "Y": "*fp32"}, {"BLOCK": 32})
    path = tmp_path / "bridge.mlir"
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), ctx)
    from triton._C.libtriton import ascend as ascend_binding
    pm = ir.pass_manager(ctx)
    ascend_binding.passes.ttir.add_tla_to_linalg(pm)
    pm.run(module, "module")
    return module


def _lower_bridge_module(tmp_path, layout="identity", placement="outside", dead=False):
    root_type = "memref<32xf32, #hivm.address_space<ub>>"
    offset, sizes, strides, view_type = {
        "identity": ("0", "32", "1", root_type),
        "offset": ("1", "16", "1", "memref<16xf32, strided<[1], offset: 1>, #hivm.address_space<ub>>"),
        "strided": ("0", "16", "2", "memref<16xf32, strided<[2]>, #hivm.address_space<ub>>"),
        # Contiguous data alone is insufficient: the layout must be identity.
        "singleton": ("0", "4, 1, 8", "8, 99, 1",
                      "memref<4x1x8xf32, strided<[8, 99, 1]>, #hivm.address_space<ub>>"),
    }[layout]
    bridge = f"%p = builtin.unrealized_conversion_cast %view : {view_type} to !tt.ptr<f32, 6>"
    text = f'''module {{
  tt.func public @bridge() {{
    %root = memref.alloc() : {root_type}
    %view = memref.reinterpret_cast %root to offset: [{offset}], sizes: [{sizes}], strides: [{strides}] : {root_type} to {view_type}
    {bridge if placement == "outside" else ""}
    scope.scope : () -> () {{
      {bridge if placement == "inside" else ""}
      {"" if dead else "%value = tt.load %p : !tt.ptr<f32, 6>"}
      scope.return
    }} {{hivm.vf_mode = #hivm.vf_mode<SIMT>, vector_mode = "simt"}}
    tt.return
  }}
}}'''
    return _run_bridge_adapter(tmp_path, text)


@pytest.mark.parametrize("placement", ["inside", "outside"])
def test_identity_alias_bridge_is_localized(tmp_path, placement):
    module = _lower_bridge_module(tmp_path, placement=placement)
    assert module.verify()
    scope, = _ops(module, "scope.scope")
    bridge, = _ops(module, "builtin.unrealized_conversion_cast")
    assert bridge.get_block().get_parent().id() == scope.get_region(0).id()


@pytest.mark.parametrize("layout,placement", [("offset", "outside"), ("strided", "inside"),
                                             ("singleton", "outside")])
def test_nonidentity_bridge_layouts_are_rejected(tmp_path, capfd, layout, placement):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _lower_bridge_module(tmp_path, layout=layout, placement=placement)
    assert "zero-offset identity layout" in capfd.readouterr().err


def test_unused_bridge_outside_scope_is_removed(tmp_path):
    module = _lower_bridge_module(tmp_path, dead=True)
    assert module.verify()
    assert not _ops(module, "builtin.unrealized_conversion_cast")


@pytest.mark.parametrize("inside", [False, True])
def test_non_pointer_ucc_is_rejected_during_collection(tmp_path, capfd, inside):
    bridge = "%invalid = builtin.unrealized_conversion_cast %arg : memref<?xf32> to memref<?xf32>"
    text = f'''module {{
  tt.func public @bridge(%arg: memref<?xf32>) {{
    {"" if inside else bridge}
    scope.scope : () -> () {{
      {bridge if inside else ""}
      scope.return
    }} {{hivm.vf_mode = #hivm.vf_mode<SIMT>, vector_mode = "simt"}}
    tt.return
  }}
}}'''
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _run_bridge_adapter(tmp_path, text)
    assert "one-to-one memref-to-pointer bridge" in capfd.readouterr().err
