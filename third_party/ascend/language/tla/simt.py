# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Bridge explicit root Buffers to standard Triton pointers inside SIMT VFs."""

from triton.language import core as tl

from .builder import ensure_tla_builder
from .core import buffer

__all__ = ["to_ptr"]


@tl.builtin
def to_ptr(src, _semantic=None):
    """Return a scalar pointer to a root Buffer with logical offset zero.

    Call inside a SIMT VF with a Buffer allocated outside it. Only GM/UB
    address spaces are supported. Views (including zero-offset subviews and
    reshapes) are not roots. Kernel GM pointers can be used directly instead.
    The pointer preserves the Buffer's element type and address space; it
    does not encode shape or strides. Compute element offsets and strides
    on the returned pointer and use standard ``tl.load/store``.

    This operation allocates no Buffer and performs no DMA or synchronization.
    """
    context = getattr(_semantic.builder, "_tla_context", None)
    if context is None or context.vf_mode != "simt":
        raise ValueError("tla.simt.to_ptr must be called inside SIMT VF")
    if not isinstance(src, buffer):
        raise TypeError("tla.simt.to_ptr expects a root tla.buffer")
    if src.space not in ("gm", "ub"):
        raise ValueError("tla.simt.to_ptr only supports GM and UB Buffers")
    if src.offset != 0:
        raise ValueError("tla.simt.to_ptr requires a root Buffer with logical offset zero; views are not supported")
    ensure_tla_builder(_semantic.builder)
    ptr_ty = tl.pointer_type(src.dtype, address_space=6 if src.space == "ub" else 1)
    return tl.tensor(_semantic.builder.create_tla_simt_pointer(src.handle, ptr_ty.to_ir(_semantic.builder)), ptr_ty)
