"""Minimal TLA SIMT VF: GM load -> Tensor add -> GM store.

Run with TLA_TEST_STAGE=ir / compile / run, or leave it unset for all stages.
The case is a development target for SIMT VF lowering and device execution.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (check_kernel_binary, compile_offline, get_stages, lower_offline)

N = 128
SIGNATURE = {"X_ptr": "*fp32", "O_ptr": "*fp32"}


@tla.jit
def gm_add_one(X_ptr, O_ptr, BLOCK: tl.constexpr):
    with tla.scope.vector():
        with tla.scope.simt():
            offsets = tl.arange(0, BLOCK)
            values = tl.load(X_ptr + offsets)
            tl.store(O_ptr + offsets, values + 1.0)


def test_ir(output_dir=None):
    module, ctx = compile_offline(gm_add_one, SIGNATURE, {"BLOCK": N}, output_dir=output_dir)
    text = str(module)
    assert "%X_ptr: !tt.ptr<f32>" in text and "%O_ptr: !tt.ptr<f32>" in text
    assert text.count("tt.load") == 1 and text.count("tt.store") == 1
    assert "#hivm.vf_mode<SIMT>" in text
    assert "memref.alloc" not in text and "hivm.hir.load" not in text

    lowered_module = lower_offline(module, ctx, output_dir=output_dir)
    lowered = str(lowered_module)
    assert 'hacc.mode = "tla"' in lowered
    assert "func.func @gm_add_one" in lowered
    assert "#hivm.vf_mode<SIMT>" in lowered
    assert "memref<?xf32>" in lowered
    assert lowered.count("tt.load") == 1 and lowered.count("tt.store") == 1
    assert lowered.count("builtin.unrealized_conversion_cast") == 2
    assert "arith.constant dense<1.000000e+00> : tensor<128xf32>" in lowered
    ops = []
    lowered_module.walk(ops.append)
    simt_scope, = [op for op in ops if op.get_name() == "scope.scope" and op.get_str_attr("vector_mode") == "simt"]
    bridges = [op for op in ops if op.get_name() == "builtin.unrealized_conversion_cast"]
    assert len(bridges) == 2
    assert all(op.get_block().get_parent().id() == simt_scope.get_region(0).id() for op in bridges)
    for forbidden in ("memref.alloc", "memref.copy", "bufferization.materialize_in_destination"):
        assert forbidden not in lowered, forbidden
    print("IR PASS: GM -> SIMT Tensor -> GM")


def _launch():
    import torch
    import torch_npu  # noqa: F401

    x = torch.arange(N, dtype=torch.float32, device="npu") / 8
    out = torch.empty_like(x)
    gm_add_one[(1, )](x, out, BLOCK=N, num_warps=4)
    torch.npu.synchronize()
    return x, out


def test_compile():
    import torch
    import torch_npu  # noqa: F401

    x = torch.empty(N, dtype=torch.float32, device="npu")
    out = torch.empty_like(x)
    check_kernel_binary(gm_add_one.warmup(x, out, BLOCK=N, num_warps=4, grid=(1, )))
    print("COMPILE PASS: gm_add_one")


def test_run():
    import torch

    x, out = _launch()
    torch.testing.assert_close(out, x + 1.0, rtol=0, atol=0)
    print("GM_ADD_ONE OK")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: gm_add_one", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
