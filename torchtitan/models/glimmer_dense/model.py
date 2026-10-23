# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Text model: Muse Glimmer attention with a dense SwiGLU FFN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
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
    _window_mask_key,
    Attention,
    MuseGlimmerTransformerBlock,
)
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)

from .deepseek_engram import DeepSeekEngram
from .engram import ENGRAM_LAYER_INDEX, NGramEngram
from .ple import LayerPLE, PerLayerEmbedder
from .ple_dram import (
    _iter_ple_parameters,
    move_ple_parameters_to_dram,
    preserve_ple_optimizer_implementation,
)


class GlimmerDenseBlock(MuseGlimmerTransformerBlock):
    """Post-norm Glimmer block whose FFN is one SwiGLU (gate, up, down)."""

    @dataclass(kw_only=True, slots=True)
    class Config(MuseGlimmerTransformerBlock.Config):
        layer_ple: LayerPLE.Config | None = None

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.layer_ple = None if config.layer_ple is None else config.layer_ple.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        per_layer_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Varlen metadata already carries the document padding mask.
        del padding_mask
        staged = None
        if (
            self.layer_ple is not None
            and self.layer_ple.config.async_training
            and per_layer_input is not None
        ):
            staged = self.layer_ple.prefetch(x)
        out = super().forward(x, attention_masks, positions)
        if self.layer_ple is None or per_layer_input is None:
            return out
        return self.layer_ple(out, per_layer_input, staged=staged)


class GlimmerDenseModel(Decoder):
    """Decoder with Muse Glimmer attention and a dense SwiGLU FFN."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 1024
        vocab_size: int = 100352
        ple: PerLayerEmbedder.Config | None = None
        ple_async_training: bool = False
        ple_unbind_inputs: bool = False
        engram: NGramEngram.Config | None = None
        deepseek_engram: DeepSeekEngram.Config | None = None

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            engram = getattr(model, "engram", None)
            excluded = () if engram is None else (engram.table,)
            deepseek_engram = getattr(model, "deepseek_engram", None)
            if deepseek_engram is not None:
                excluded = excluded + deepseek_engram.tables()
            nparams, active_nparams = get_nparams_and_active_nparams(
                model,
                modules_excluded_from_active_params=excluded,
            )
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

    def __init__(self, config: Config) -> None:
        if config.ple is not None:
            config.ple.async_training = config.ple_async_training
        for layer in config.layers:
            if layer.layer_ple is not None:
                layer.layer_ple.async_training = config.ple_async_training
        super().__init__(config)
        self.ple = None if config.ple is None else config.ple.build()
        self.engram = None if config.engram is None else config.engram.build()
        self.deepseek_engram = (
            None if config.deepseek_engram is None else config.deepseek_engram.build()
        )
        if config.ple_async_training:
            for parameter in _iter_ple_parameters(self):
                parameter._torchtitan_skip_fsdp = True

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        super().init_states(buffer_device=buffer_device)
        if self.ple is not None and self.config.ple_async_training:
            move_ple_parameters_to_dram(self)

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        super()._register_optimizer_hooks(optimizers, model_parts, parallelism_context)
        asynchronous_models = [
            model for model in model_parts if model.config.ple_async_training
        ]
        if asynchronous_models:
            preserve_ple_optimizer_implementation(optimizers, asynchronous_models)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ):
        have_tokens = self.tok_embeddings is not None
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        row_prefetch = (
            self.ple.prefetch_rows(tokens, dtype=h.dtype)
            if self.ple is not None and self.config.ple_async_training and have_tokens
            else None
        )
        per_layer_inputs = (
            self.ple(tokens, h, row_prefetch)
            if self.ple is not None and have_tokens
            else None
        )
        # UnbindBackward stacks layer gradients once instead of zero-filling
        # and adding a full token-by-all-layers tensor for every layer slice.
        per_layer_slices = (
            per_layer_inputs.unbind(dim=1)
            if per_layer_inputs is not None and self.config.ple_unbind_inputs
            else None
        )
        for index, layer in enumerate(self.layers.values()):
            if (
                self.engram is not None
                and index == ENGRAM_LAYER_INDEX
                and have_tokens
                and positions is not None
            ):
                h = self.engram(h, tokens, positions)
            if (
                self.deepseek_engram is not None
                and index in self.deepseek_engram.layer_indices
            ):
                if not have_tokens or positions is None:
                    raise ValueError("Engram needs token ids and positions")
                h = self.deepseek_engram(index, h, tokens, positions)
            layer_input = None
            if per_layer_inputs is not None:
                layer_input = (
                    per_layer_inputs[:, index]
                    if per_layer_slices is None
                    else per_layer_slices[index]
                )
            h = layer(
                h,
                attention_masks,
                positions,
                padding_mask=padding_mask,
                per_layer_input=layer_input,
            )
        h = self.norm(h) if self.norm is not None else h
        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h

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
