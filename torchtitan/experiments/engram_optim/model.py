# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU Engram variants with native CPU and higher-order fallbacks.

Shape legend: T = tokens, D = hidden width, H = n-gram heads.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torchtitan.models.glimmer_dense.engram import (
    _CONV_DILATION,
    EngramTable,
    EngramUpdate,
    _segment_ids,
    _segmented_short_conv,
)


class FusedEngramTable(EngramTable):
    @dataclass(kw_only=True, slots=True)
    class Config(EngramTable.Config):
        fused_lookup: bool = False

    def forward(self, tokens, positions):
        if (
            not self.config.fused_lookup
            or tokens.device.type != "cuda"
            or not self.weight.is_contiguous()
            or not tokens.is_contiguous()
            or tokens.dtype not in (torch.int32, torch.int64)
        ):
            return super().forward(tokens, positions)
        from .kernels import lookup

        segments_T = _segment_ids(tokens, positions, fill_id=self.eos_token_id)
        embeddings_TD = lookup(
            self.weight, tokens, segments_T, self.rows_per_head, self.eos_token_id
        )
        if self.remainder_weight is None:
            return embeddings_TD
        width = self.remainder_weight.numel()
        head_TD = embeddings_TD[:, :width] + self.remainder_weight
        return torch.cat((head_TD, embeddings_TD[:, width:]), dim=-1)


class OptimizedEngramUpdate(EngramUpdate):
    @dataclass(kw_only=True, slots=True)
    class Config(EngramUpdate.Config):
        fused_conv: bool = False
        activation_checkpoint: bool = False

    def forward(self, hidden, embeddings, tokens, positions):
        if self.config.activation_checkpoint and torch.is_grad_enabled():
            return checkpoint(
                self._forward,
                hidden,
                embeddings,
                tokens,
                positions,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self._forward(hidden, embeddings, tokens, positions)

    def _forward(self, hidden_TD, embeddings_TD, tokens_T, positions_T):
        segments_T = _segment_ids(
            tokens_T, positions_T, fill_id=self.config.eos_token_id
        )
        keys_TD = F.linear(embeddings_TD, self.key_weight)
        values_TD = F.linear(embeddings_TD, self.value_up)
        values_TD = F.linear(values_TD, self.value_down)
        conv = _segmented_short_conv
        if self.config.fused_conv and values_TD.device.type == "cuda":
            from .kernels import segmented_conv

            conv = segmented_conv
        values_TD = conv(
            values_TD, self.conv_weight, segments_T, dilation=_CONV_DILATION
        )
        scale_width = self.value_scale.numel()
        scaled_TD = values_TD[:, :scale_width] * self.value_scale
        values_TD = torch.cat((scaled_TD, values_TD[:, scale_width:]), dim=-1)
        scale = hidden_TD.shape[-1] ** -0.5
        gate_TD = torch.sigmoid(
            (hidden_TD * keys_TD).sum(dim=-1, keepdim=True) * scale - 4
        )
        return hidden_TD + gate_TD * values_TD
