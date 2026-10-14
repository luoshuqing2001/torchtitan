# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_inner_attention_local_spmd,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.muse_glimmer.model import Attention
from torchtitan.models.muse_glimmer.sharding import _set_attention_sharding
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from .model import GlimmerQwen3MoEBlock, GlimmerQwen3MoEModel


def set_glimmer_qwen3_moe_sharding_config(
    config: "GlimmerQwen3MoEModel.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Fill sharding configs for Muse Glimmer attention and Qwen3 MoE."""
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_layer_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)


def _set_layer_sharding(
    layer_cfg: "GlimmerQwen3MoEBlock.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    attention = layer_cfg.attention
    assert isinstance(attention, Attention.Config)
    sp_activation = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    layer_cfg.attention_norm.sharding_config = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R if enable_sp else spmd.I)
        },
        in_src_shardings={"input": sp_activation},
        out_src_shardings=sp_activation,
    )
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.ffn_norm.sharding_config = norm
    layer_cfg.post_attention_norm.sharding_config = norm
    layer_cfg.post_ffn_norm.sharding_config = norm

    _set_attention_sharding(attention, enable_sp=enable_sp)
    set_gqa_inner_attention_local_spmd(attention.inner_attention)
    if attention.qk_norm is not None:
        head_shard = attention_activation_placement()
        attention.qk_norm.sharding_config = ShardingConfig(
            in_src_shardings={"input": head_shard},
            in_dst_shardings={"input": head_shard},
            out_src_shardings=head_shard,
            out_dst_shardings=head_shard,
        )
    if attention.o_gate is not None:
        attention.o_gate.sharding_config = colwise_config(
            input_layout=dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        )

    assert layer_cfg.moe is not None
    set_moe_sharding_config(
        layer_cfg.moe,
        enable_ep=enable_ep,
        enable_sp=enable_sp,
    )
    for shared_expert in layer_cfg.shared_experts:
        set_dense_ffn_sharding(
            shared_expert,
            attn_x_layout=sp_activation,
            enable_sp=enable_sp,
        )
