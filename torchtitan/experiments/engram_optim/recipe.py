# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preserve ordinary Engram training unless an optimization is requested."""

import os
from dataclasses import fields

from torchtitan_recipes.models.glimmer_dense import glimmer_dense as base_recipe

from .model import FusedEngramTable, OptimizedEngramUpdate


def _convert(config, config_type):
    return config_type(
        **{field.name: getattr(config, field.name) for field in fields(config)}
    )


def glimmer_dense():
    flags = {}
    for name in (
        "ENGRAM_FUSED_LOOKUP",
        "ENGRAM_FUSED_CONV",
        "ENGRAM_UPDATE_CHECKPOINT",
        "ENGRAM_COMPILE_FP8",
    ):
        value = os.environ.get(name, "0")
        if value not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
        flags[name] = value == "1"
    config = base_recipe()
    if config.model.engram is None:
        raise ValueError("Set PLE=engram_legacy for the Engram optimization recipe")
    engram = config.model.engram
    if flags["ENGRAM_COMPILE_FP8"]:
        config.compile.regions.append("float8_quantize")
    if flags["ENGRAM_FUSED_LOOKUP"]:
        engram.table = _convert(engram.table, FusedEngramTable.Config)
        engram.table.fused_lookup = True
    if flags["ENGRAM_FUSED_CONV"] or flags["ENGRAM_UPDATE_CHECKPOINT"]:
        engram.update = _convert(engram.update, OptimizedEngramUpdate.Config)
        engram.update.fused_conv = flags["ENGRAM_FUSED_CONV"]
        engram.update.activation_checkpoint = flags["ENGRAM_UPDATE_CHECKPOINT"]
    return config


def glimmer_dense_dp():
    """Use the same opt-in kernels with a fully replicated DP backbone."""
    config = glimmer_dense()
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("NGPU", "1")))
    config.parallelism.data_parallel_replicate_degree = world
    config.parallelism.data_parallel_shard_degree = 1
    return config
