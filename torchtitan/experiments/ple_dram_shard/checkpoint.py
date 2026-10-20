# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Retain staged PLE linear outputs through the existing selective AC policy.

Shape legend: T = tokens, D = hidden width, P = per-layer embedding width.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.models.glimmer_dense import ple_dram as dram
from torchtitan.models.glimmer_dense.ple import LayerPLE, PerLayerContext


@torch.library.custom_op("ple_dram_shard::staged_linear", mutates_args=())
def _linear(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(inputs, weight)


@_linear.register_fake
def _linear_fake(inputs, weight):
    return inputs.new_empty((*inputs.shape[:-1], weight.shape[0]))


class _SavedLinear(dram._DramLinear):
    @staticmethod
    def forward(ctx, weight, staged, inputs, mixed_precision_backward):
        ctx.save_for_backward(inputs, staged)
        ctx.cpu_dtype = weight.dtype
        ctx.mixed_precision_backward = mixed_precision_backward
        return _linear(inputs, staged)


def _staged_linear(weight, inputs, staged, mixed_precision_backward, transport=None):
    if transport is not None:
        from .replicated_gradient import ReducedLinear

        return ReducedLinear.apply(
            weight,
            dram._match_inputs(staged.value(), inputs),
            inputs,
            mixed_precision_backward,
            transport,
            _linear,
        )
    return _SavedLinear.apply(
        weight,
        dram._match_inputs(staged.value(), inputs),
        inputs,
        mixed_precision_backward,
    )


def _norm(weight, inputs, eps, staged, transport):
    if transport is None:
        return dram.dram_rms_norm(weight, inputs, eps, staged)
    from .replicated_gradient import ReducedNorm

    return ReducedNorm.apply(
        weight, dram._match_inputs(staged.value(), inputs), inputs, eps, transport
    )


_transport = None


def _get_transport():
    global _transport
    if _transport is None:
        from .replicated_gradient import ReplicatedGradientTransport

        _transport = ReplicatedGradientTransport()
    return _transport


class SavedContext(PerLayerContext):
    @dataclass(kw_only=True, slots=True)
    class Config(PerLayerContext.Config):
        activation_checkpoint: bool = False
        reuse_norm_input: bool = False

    def forward(self, hidden):
        if not self.config.async_training:
            return super().forward(hidden)
        if self.config.activation_checkpoint and torch.is_grad_enabled():
            # Keep the row lookup outside the checkpoint so its cache can die
            # after blending, instead of surviving until root recomputation.
            return checkpoint(
                self._forward, hidden, use_reentrant=False, preserve_rng_state=False
            )
        return self._forward(hidden)

    def _forward(self, hidden):
        transport = _get_transport()
        projected = _staged_linear(
            self.weight,
            hidden,
            dram.stage(self.weight, hidden.device, hidden.dtype),
            self.config.mixed_precision_backward,
            transport,
        )
        projected = projected * self.config.dim**-0.5
        projected = projected.unflatten(-1, (self.num_layers, self.ple_dim))
        staged_norm = dram.stage(self.norm_weight, projected.device, projected.dtype)
        if self.config.reuse_norm_input:
            from .replicated_gradient import ReusingContextNorm

            # Scaling above creates sole-use storage. Neither the upstream
            # scale nor linear backward saves these projected values.
            projected = ReusingContextNorm.apply(
                self.norm_weight,
                dram._match_inputs(staged_norm.value(), projected),
                projected,
                1e-6,
                transport,
            )
        else:
            projected = _norm(
                self.norm_weight,
                projected,
                1e-6,
                staged_norm,
                transport,
            )
        return projected.flatten(-2, -1)


class SavedLayerPLE(LayerPLE):
    @dataclass(kw_only=True, slots=True)
    class Config(LayerPLE.Config):
        overlap_replicated_gradient: bool = False

    def forward(self, hidden, ple, *, staged=None):
        if not self.config.async_training:
            return super().forward(hidden, ple, staged=staged)
        if ple.shape[-1] != self.config.ple_dim:
            raise ValueError(
                f"PLE slice width must be {self.config.ple_dim}, got {ple.shape[-1]}"
            )
        if staged is None:
            staged = self.prefetch(hidden)
        if ple.dtype != hidden.dtype:
            ple = ple.to(dtype=hidden.dtype)
        transport = (
            _get_transport() if self.config.overlap_replicated_gradient else None
        )
        gated_TP = (
            F.gelu(
                _staged_linear(
                    self.gate_weight,
                    hidden,
                    staged["gate"],
                    self.config.mixed_precision_backward,
                    transport,
                )
            )
            * ple
        )
        update_TD = _staged_linear(
            self.proj_weight,
            gated_TP,
            staged["proj"],
            self.config.mixed_precision_backward,
            transport,
        )
        update_TD = _norm(self.norm_weight, update_TD, 1e-6, staged["norm"], transport)
        return hidden + update_TD


class SavePLEMatmuls(SelectiveAC):
    @dataclass(kw_only=True, slots=True)
    class Config(SelectiveAC.Config):
        num_layers_to_save: int = 0

    def get_save_ops(self):
        return super().get_save_ops() | {torch.ops.ple_dram_shard.staged_linear.default}

    def _wrap_block(self, module, *, base_fqn=None):
        layer = 0 if base_fqn is None else int(base_fqn.rsplit(".", 1)[1])
        if layer < self.config.num_layers_to_save:
            wrapped = super()._wrap_block(module, base_fqn=base_fqn)
        else:
            wrapped = SelectiveAC(self.config, self.dump_folder)._wrap_block(
                module, base_fqn=base_fqn
            )
        manager = getattr(module, "_ple_saved_input_staging", None)
        if manager is not None:
            from .saved_inputs import wrap_checkpoint_inputs

            wrapped = wrap_checkpoint_inputs(wrapped, manager, layer)
        return wrapped
