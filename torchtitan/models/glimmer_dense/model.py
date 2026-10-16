# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Text model: Muse Glimmer attention with a dense SwiGLU FFN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common.attention import (
    AttentionMasksType,
    create_attention_mask,
    create_varlen_metadata_for_document,
    FlexInnerAttention,
    get_causal_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
    get_sliding_window_mask_mod,
    VarlenInnerAttention,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.muse_glimmer.model import (
    Attention,
    MuseGlimmerTransformerBlock,
    _window_mask_key,
)
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)


class GlimmerDenseBlock(MuseGlimmerTransformerBlock):
    """Post-norm Glimmer block whose FFN is one SwiGLU (gate, up, down)."""

    @dataclass(kw_only=True, slots=True)
    class Config(MuseGlimmerTransformerBlock.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Varlen metadata already carries the document padding mask.
        del padding_mask
        return super().forward(x, attention_masks, positions)


class GlimmerDenseModel(Decoder):
    """Decoder with Muse Glimmer attention and a dense SwiGLU FFN."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 1024
        vocab_size: int = 50304

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            attention_op_flops = 0
            for layer in self.layers:
                attention = layer.attention
                head_dim = (
                    attention.head_dim
                    if attention.head_dim is not None
                    else attention.dim // attention.n_heads
                )
                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=head_dim,
                    v_head_dim=head_dim,
                    seq_len=seq_len,
                )
            return nparams, 6 * active_nparams + attention_op_flops

        def set_sharding_(self, parallelism: ParallelismConfig) -> None:
            from .sharding import set_glimmer_dense_sharding_config

            set_glimmer_dense_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
            )

    def get_attention_masks(
        self,
        positions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
    ) -> AttentionMasksType:
        attn_config = self.config.first_attention
        assert isinstance(attn_config, Attention.Config)
        inner_attn = attn_config.inner_attention
        if isinstance(inner_attn, VarlenInnerAttention.Config):
            return create_varlen_metadata_for_document(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
        if not isinstance(inner_attn, FlexInnerAttention.Config):
            raise TypeError(
                "Glimmer attention requires FlexInnerAttention or "
                f"VarlenInnerAttention, got {type(inner_attn).__name__}"
            )

        seq_len = positions.shape[0]
        base_mods = [
            get_causal_mask_mod(),
            get_efficient_causal_mask_mod_for_packed_document(positions),
        ]
        block_size = inner_attn.block_size
        separate_full_blocks = not is_in_batch_invariant_mode()

        def _build_mask(mask_mods: list) -> BlockMask:
            return create_attention_mask(
                and_masks(*mask_mods),
                1,
                None,
                seq_len,
                seq_len,
                device=positions.device,
                BLOCK_SIZE=block_size,
                separate_full_blocks=separate_full_blocks,
            )

        masks: dict[str, BlockMask] = {_window_mask_key(None): _build_mask(base_mods)}
        window_sizes = {
            layer.attention.window_size
            for layer in self.config.layers
            if isinstance(layer.attention, Attention.Config)
            and layer.attention.window_size is not None
        }
        for window_size in window_sizes:
            masks[_window_mask_key(window_size)] = _build_mask(
                [*base_mods, get_sliding_window_mask_mod(window_size)]
            )
        return masks
