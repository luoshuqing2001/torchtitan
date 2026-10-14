# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Text model: Muse Glimmer attention with a Qwen3-style MoE block."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn as nn
import torch_remat as remat
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
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import MoE, register_moe_load_balancing_hook
from torchtitan.models.muse_glimmer.model import Attention, _window_mask_key
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)

from .diagnostics import LayerDiagnostics, routing_and_hidden_stats


class GlimmerQwen3MoEBlock(TransformerBlock):
    """Post-norm Muse Glimmer block whose FFN is a Qwen3 MoE.

    Routed experts live in ``moe``. The always-on shared experts are separate
    SwiGLU FFNs so each one keeps its own gate, matching a count of shared
    experts rather than one wider MLP.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        post_attention_norm: nn.Module
        post_ffn_norm: nn.Module
        shared_experts: list[FeedForward.Config]

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention = config.attention.build()
        assert config.moe is not None
        self.moe = config.moe.build()
        self.moe_enabled = True
        self.shared_experts = nn.ModuleList(
            expert.build() for expert in config.shared_experts
        )
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.post_attention_norm = config.post_attention_norm.build()
        self.post_ffn_norm = config.post_ffn_norm.build()
        self.diagnostics = LayerDiagnostics()
        # Full-sequence router and activation stats once per log window.
        # DIAG_EVERY=1 records every step.
        self._diag_interval = max(1, int(os.environ.get("DIAG_EVERY", "10")))
        self._diag_forwards = 0

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = self.attention(self.attention_norm(x), attention_masks, positions)
        h = x + self.post_attention_norm(attn_out)
        moe_in = self.ffn_norm(h)
        routed = self.moe(moe_in, padding_mask_T=padding_mask)
        shared = self.shared_experts[0](moe_in)
        for expert in self.shared_experts[1:]:
            shared = shared + expert(moe_in)
        moe_out = routed + shared
        out = h + self.post_ffn_norm(moe_out)
        if self.training and not remat.is_recomputing():
            self._diag_forwards += 1
            if self._diag_forwards % self._diag_interval == 0:
                self._record_diagnostics(
                    moe_in=moe_in,
                    hidden=out,
                    attn_out=attn_out,
                    moe_out=moe_out,
                    padding_mask=padding_mask,
                )
        return out

    @torch.no_grad()
    def _record_diagnostics(
        self,
        *,
        moe_in: torch.Tensor,
        hidden: torch.Tensor,
        attn_out: torch.Tensor,
        moe_out: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> None:
        router = self.moe.router
        scores_TE = router.score_func(router.gate(moe_in))
        self.diagnostics.update(
            routing_and_hidden_stats(
                scores_TE=scores_TE,
                top_k=router.top_k,
                num_experts=router.num_experts,
                hidden_TD=hidden,
                attn_out_TD=attn_out,
                moe_out_TD=moe_out,
                expert_bias_E=self.moe.expert_bias_E,
                padding_mask_T=padding_mask,
            )
        )


class GlimmerQwen3MoEModel(Decoder):
    """Decoder with Muse Glimmer attention and Qwen3 token-choice MoE."""

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 768
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
            from .sharding import set_glimmer_qwen3_moe_sharding_config

            set_glimmer_qwen3_moe_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
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
