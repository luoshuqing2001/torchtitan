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
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_inner_attention_local_spmd,
    token_id_placement,
)
from torchtitan.models.muse_glimmer.sharding import _set_muse_glimmer_layer_sharding
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from .model import GlimmerDenseModel


def set_glimmer_dense_sharding_config(
    config: "GlimmerDenseModel.Config",
    *,
    enable_sp: bool,
) -> None:
    """Fill sharding configs for Muse Glimmer attention and a dense SwiGLU."""
    from .factored_moe import FactoredMoE
    from .model import GlimmerAttentionBlock

    from torchtitan.models.common.attention import QKVLinear

    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        fused = isinstance(layer_cfg.attention.qkv_linear, QKVLinear.Config)
        routed = isinstance(
            getattr(layer_cfg, "feed_forward", None), FactoredMoE.Config
        )
        if isinstance(layer_cfg, GlimmerAttentionBlock.Config) or not fused or routed:
            _set_attention_only_layer_sharding(layer_cfg, enable_sp=enable_sp)
        else:
            _set_muse_glimmer_layer_sharding(layer_cfg, enable_sp=enable_sp)
    if config.ple is not None and not config.ple_async_training:
        _set_ple_sharding(config.ple, enable_sp=enable_sp)
        for layer_cfg in config.layers:
            if layer_cfg.layer_ple is not None:
                _set_layer_ple_sharding(layer_cfg.layer_ple, enable_sp=enable_sp)
    if config.engram is not None:
        _set_engram_sharding(config.engram, enable_sp=enable_sp)
    if config.deepseek_engram is not None:
        for layer in config.deepseek_engram.layers.values():
            _set_deepseek_engram_sharding(layer, enable_sp=enable_sp)


def _set_attention_only_layer_sharding(layer_cfg, *, enable_sp: bool) -> None:
    """Muse Glimmer layer sharding for unfused Q/K/V projections.

    Used for attention-only blocks and for dense blocks whose projections are
    not one fused ``wqkv`` (query MLP, SGQA). Every projection linear is placed
    like ``wqkv``. The recipe runs them with tensor parallel degree 1 (DistMuon
    requires it), so only FSDP applies.
    """
    attention = layer_cfg.attention
    x_layout = _activation_layout(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R if enable_sp else spmd.I)
        },
        in_src_shardings={"input": x_layout},
        out_src_shardings=x_layout,
    )
    layer_cfg.post_attention_norm.sharding_config = norm_config(enable_sp=enable_sp)
    attention.sharding_config = ShardingConfig(in_src_shardings={"x_TD": x_layout})
    gathered = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    qkv = attention.qkv_linear
    for name in ("wqkv", "wq", "q_up", "q_down", "wkv", "wk", "wk_rope", "wv"):
        linear = getattr(qkv, name, None)
        if linear is not None:
            linear.sharding_config = colwise_config(input_layout=gathered)
    attention.wo.sharding_config = rowwise_config(output_layout=x_layout)
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
        attention.o_gate.sharding_config = colwise_config(input_layout=gathered)
    if getattr(layer_cfg, "feed_forward", None) is not None:
        from .factored_moe import FactoredMoE

        norm = norm_config(enable_sp=enable_sp)
        layer_cfg.ffn_norm.sharding_config = norm
        layer_cfg.post_ffn_norm.sharding_config = norm
        if isinstance(layer_cfg.feed_forward, FactoredMoE.Config):
            _set_factored_moe_sharding(layer_cfg.feed_forward, enable_sp=enable_sp)
        else:
            set_dense_ffn_sharding(
                layer_cfg.feed_forward, attn_x_layout=x_layout, enable_sp=enable_sp
            )


def _set_factored_moe_sharding(moe_config, *, enable_sp: bool) -> None:
    """Replicate the routed FFN's weights and run it in one local region.

    The recipe runs it with tensor and expert parallel degree 1, so only FSDP
    shards the weights.
    """
    activation = _activation_layout(enable_sp=enable_sp)
    weight = dense_param_placement(tp=spmd.I)
    from torchtitan.models.common.moe_sharding import _tokens_per_expert_placement

    names = ["router_weight", "w_gate", "w_up", "w_down", "expert_bias"]
    counts = ["tokens_per_expert"]
    if moe_config.split:
        names += ["gate_router_weight", "gate_expert_bias"]
        counts += ["tokens_per_gate_expert"]
    # Expert loads are per-rank partial sums, as for the MoE router's.
    state = {name: weight for name in names}
    state.update({name: _tokens_per_expert_placement() for name in counts})
    moe_config.sharding_config = ShardingConfig(
        state_shardings=state,
        in_src_shardings={"x": activation},
        in_dst_shardings={"x": activation},
        out_src_shardings=activation,
        out_dst_shardings=activation,
        local_spmd=True,
    )


def _activation_layout(*, enable_sp: bool):
    if enable_sp:
        return dense_sequence_parallel_placement()
    return dense_activation_placement(tp=spmd.I, cp=spmd.S(0))


def _set_ple_sharding(ple_config, *, enable_sp: bool) -> None:
    """Place ordinary PLE parameters and activations on the GPU mesh."""
    activation = _activation_layout(enable_sp=enable_sp)
    tokens = token_id_placement(enable_sp=enable_sp)
    assert ple_config.table is not None and ple_config.context is not None
    ple_config.table.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": tokens},
        in_dst_shardings={"input": tokens},
        out_src_shardings=dense_activation_placement(tp=spmd.P, cp=spmd.S(0)),
        out_dst_shardings=activation,
        local_spmd=True,
    )
    ple_config.context.sharding_config = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.I),
            "norm_weight": dense_param_placement(tp=spmd.I),
        },
        in_src_shardings={"hidden": activation},
        in_dst_shardings={"hidden": activation},
        out_src_shardings=activation,
        out_dst_shardings=activation,
        local_spmd=True,
    )


def _set_engram_sharding(engram_config, *, enable_sp: bool) -> None:
    """Replicate the n-gram table and run the gate in one local region."""
    activation = _activation_layout(enable_sp=enable_sp)
    tokens = token_id_placement(enable_sp=enable_sp)
    weight = dense_param_placement(tp=spmd.I)
    assert engram_config.table is not None and engram_config.update is not None
    table_shardings = {"weight": weight}
    if engram_config.remainder:
        table_shardings["remainder_weight"] = weight
    engram_config.table.sharding_config = ShardingConfig(
        state_shardings=table_shardings,
        in_src_shardings={"tokens": tokens, "positions": tokens},
        in_dst_shardings={"tokens": tokens, "positions": tokens},
        out_src_shardings=activation,
        out_dst_shardings=activation,
        local_spmd=True,
    )
    engram_config.update.sharding_config = ShardingConfig(
        state_shardings={
            "key_weight": weight,
            "value_up": weight,
            "value_down": weight,
            "value_scale": weight,
            "conv_weight": weight,
        },
        in_src_shardings={
            "hidden": activation,
            "embeddings": activation,
            "tokens": tokens,
            "positions": tokens,
        },
        in_dst_shardings={
            "hidden": activation,
            "embeddings": activation,
            "tokens": tokens,
            "positions": tokens,
        },
        out_src_shardings=activation,
        out_dst_shardings=activation,
        local_spmd=True,
    )


def _set_deepseek_engram_sharding(layer_config, *, enable_sp: bool) -> None:
    """Replicate one Engram layer's table and projections; run it locally."""
    activation = _activation_layout(enable_sp=enable_sp)
    tokens = token_id_placement(enable_sp=enable_sp)
    weight = dense_param_placement(tp=spmd.I)
    layer_config.sharding_config = ShardingConfig(
        state_shardings={
            name: weight
            for name in (
                "key_weight",
                "value_weight",
                "query_norm_weight",
                "key_norm_weight",
                "conv_norm_weight",
                "conv_weight",
            )
        },
        in_src_shardings={"hidden": activation, "tokens": tokens, "positions": tokens},
        in_dst_shardings={"hidden": activation, "tokens": tokens, "positions": tokens},
        out_src_shardings=activation,
        out_dst_shardings=activation,
        local_spmd=True,
    )
    assert layer_config.table is not None
    layer_config.table.sharding_config = ShardingConfig(
        state_shardings={"weight": weight},
        in_src_shardings={"rows": tokens},
        in_dst_shardings={"rows": tokens},
        out_src_shardings=activation,
        out_dst_shardings=activation,
        local_spmd=True,
    )


def _set_layer_ple_sharding(layer_ple, *, enable_sp: bool) -> None:
    activation = _activation_layout(enable_sp=enable_sp)
    weight = dense_param_placement(tp=spmd.I)
    layer_ple.sharding_config = ShardingConfig(
        state_shardings={
            "gate_weight": weight,
            "proj_weight": weight,
            "norm_weight": weight,
        },
        in_src_shardings={"hidden": activation, "ple": activation},
        in_dst_shardings={"hidden": activation, "ple": activation},
        out_src_shardings=activation,
        out_dst_shardings=activation,
        local_spmd=True,
    )
