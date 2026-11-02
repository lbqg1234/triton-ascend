# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Bridge zero-offset contiguous UB Buffers to standard Triton pointers inside SIMT VFs."""

from triton.language import core as tl

from .core import buffer

__all__ = ["to_ptr"]


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
