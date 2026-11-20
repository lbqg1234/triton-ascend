# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Pointer bridges and explicit Thread collectives for SIMT VFs."""

from triton.language import core as tl

from .core import buffer

__all__ = ["to_ptr", "thread_id", "warp_reduce_sum", "sync_threads"]


@tl.builtin
def to_ptr(src, _semantic=None):
    """Return a scalar pointer to a contiguous UB Buffer with logical offset zero.

    Call inside a SIMT VF with a UB Buffer defined outside it. Its
    layout must be identity (zero offset and contiguous row-major data). Use kernel
    GM pointers directly. The pointer preserves the Buffer's element type
    and uses address space 6; it does not encode shape or strides.
    Compute element offsets and strides
    on the returned pointer and use standard ``tl.load/store``.

    This operation allocates no Buffer and performs no DMA or synchronization.
    """
    context = getattr(_semantic.builder, "_tla_context", None)
    if context is None or context.vf_mode != "simt":
        raise ValueError("tla.simt.to_ptr must be called inside SIMT VF")
    if not isinstance(src, buffer):
        raise TypeError("tla.simt.to_ptr expects a tla.buffer")
    if src.space != "ub":
        raise ValueError("tla.simt.to_ptr only supports UB Buffers")
    if src.offset != 0:
        raise ValueError("tla.simt.to_ptr requires a Buffer with known logical offset zero")
    ptr_ty = tl.pointer_type(src.dtype, address_space=6)
    return tl.tensor(_semantic.builder.create_tla_simt_pointer(src.handle, ptr_ty.to_ir(_semantic.builder)), ptr_ty)


@tl.builtin
def thread_id(_semantic=None):
    """Return the current Thread VF thread index as an i32 scalar.

    Only valid inside ``tla.scope.simt(mode="thread")``. The one-dimensional
    index is in ``[0, num_warps * 32)`` for the current VF.
    """
    _check_thread_vf("tla.simt.thread_id", _semantic)
    return tl.tensor(_semantic.builder.create_tla_simt_thread_id(), tl.int32)


def _check_thread_vf(name, semantic):
    context = getattr(semantic.builder, "_tla_context", None)
    if (context is None or context.vf_mode != "simt" or context.simt_mode != "thread"):
        raise ValueError(f"{name} must be called inside Thread SIMT VF")


@tl.builtin
def warp_reduce_sum(value, _semantic=None):
    """Return the sum of all 32 lanes in the current Warp to every lane.

    Only valid inside a Thread SIMT VF. Each lane must reach the same
    dynamic call; inactive elements must contribute a same-type zero.
    Accepts float16, float32 and int32 scalars and preserves their dtype.
    Floating-point addition order is implementation-defined. Int32 addition
    wraps at 32 bits. No implicit shared Buffer or synchronization between
    Warps is introduced.
    """
    _check_thread_vf("tla.simt.warp_reduce_sum", _semantic)
    value = tl._unwrap_if_constexpr(value)
    if not isinstance(value, tl.tensor) and type(value) not in (int, float):
        raise TypeError("tla.simt.warp_reduce_sum expects a float16, float32 or int32 scalar")
    value = _semantic.to_tensor(value)
    if value.type.is_block() or value.dtype not in (tl.float16, tl.float32, tl.int32):
        raise TypeError("tla.simt.warp_reduce_sum expects a float16, float32 or int32 scalar")
    return tl.tensor(_semantic.builder.create_tla_simt_warp_reduce_sum(value.handle), value.dtype)


@tl.builtin
def sync_threads(_semantic=None):
    """Synchronize all threads of the current Thread SIMT VF.

    Every thread must reach the same dynamic barrier instance. UB writes
    before the barrier are visible to reads after it. When reusing scratch,
    synchronize again after all consumers have finished reading. This
    operation allocates no Buffer and does not synchronize different VFs.
    """
    _check_thread_vf("tla.simt.sync_threads", _semantic)
    _semantic.builder.create_tla_simt_sync_threads()
