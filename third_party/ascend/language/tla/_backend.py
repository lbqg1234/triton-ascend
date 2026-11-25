# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Adapt the Ascend compilation stages for a single TLA compilation."""
from triton.backends.ascend import compiler as ascend
from triton.compiler import compiler as triton_compiler

from ._options import _EXTRA_OPTIONS, normalize_extra_options


class _TLABackend(ascend.AscendBackend):
    # The generic reuse handler rebuilds kernels with a plain ASTSource.
    get_jit_reuse_handler = None

    def parse_options(self, opts):
        options = super().parse_options(opts)
        if options.block_batching_factor != 1:
            raise ValueError("TLA kernels do not support automatic block batching")
        object.__setattr__(options, "enable_tla_mode", True)
        shared_size = opts.get("shared_mem_dynamic_size")
        if shared_size is not None and ascend._is_a5_target_arch(self.target.arch):
            object.__setattr__(options, "shared_mem_dynamic_size", shared_size)
        object.__setattr__(options, _EXTRA_OPTIONS, normalize_extra_options(opts.get(_EXTRA_OPTIONS, ())))
        return options

    def add_stages(self, stages, options, language):
        super().add_stages(stages, options, language)

        def ttadapter(module, metadata):
            from .compiler import lower_for_backend

            metadata["has_auto_blockify_blacklist_op"] = False
            metadata["tla_compiler_args"] = lower_for_backend(module, options)
            ascend._export_profile_metadata(module, metadata)
            ascend._export_program_grid_metadata(module, metadata)
            text = ascend._serialize_module(module, options)
            if options.debug:
                ascend.get_dump_manager(metadata["hash"]).put(text, "kernel.ttadapter.mlir", binary=False)
            return text

        stages["ttadapter"] = ttadapter


def make_backend(target):
    return _TLABackend(target) if target.backend == "npu" else triton_compiler.make_backend(target)


def compile_tla(src, target=None, options=None, _env_vars=None):
    return triton_compiler.compile(src, target=target, options=options, _env_vars=_env_vars,
                                   _backend_factory=make_backend)
