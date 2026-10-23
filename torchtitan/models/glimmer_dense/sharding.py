# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    set_decoder_sharding_config,
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
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
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
