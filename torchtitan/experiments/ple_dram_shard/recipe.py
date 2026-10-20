# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in row-sharded variant of the local Glimmer DRAM recipe."""

import os
from dataclasses import fields, replace

from torchtitan_recipes.models.glimmer_dense import glimmer_dense as base_recipe

from .model import RowShardedGlimmer, RowShardedPLE
from .optim import RowShardedOptim
from .overlap import CPUOverlapOptimizersContainer


def _convert(config, config_type):
    return config_type(
        **{field.name: getattr(config, field.name) for field in fields(config)}
    )


def glimmer_dense():
    config = base_recipe()
    early_flag = os.environ.get("PLE_EARLY_FETCH", "0")
    if early_flag not in ("0", "1"):
        raise ValueError("PLE_EARLY_FETCH must be 0 or 1")
    if early_flag == "1" and not config.model.ple_async_training:
        raise ValueError("PLE_EARLY_FETCH=1 requires PLE_ASYNC_TRAINING=1")
    save_flag = os.environ.get("PLE_SAVE_MATMULS", "0")
    if save_flag not in ("0", "1"):
        raise ValueError("PLE_SAVE_MATMULS must be 0 or 1")
    if save_flag == "1" and not config.model.ple_async_training:
        raise ValueError("PLE_SAVE_MATMULS=1 requires PLE_ASYNC_TRAINING=1")
    replica_flag = os.environ.get("PLE_OVERLAP_REPLICATED_GRADIENT", "0")
    if replica_flag not in ("0", "1"):
        raise ValueError("PLE_OVERLAP_REPLICATED_GRADIENT must be 0 or 1")
    if replica_flag == "1" and save_flag != "1":
        raise ValueError(
            "PLE_OVERLAP_REPLICATED_GRADIENT=1 requires PLE_SAVE_MATMULS=1"
        )
    context_checkpoint_flag = os.environ.get("PLE_CONTEXT_CHECKPOINT", "0")
    if context_checkpoint_flag not in ("0", "1"):
        raise ValueError("PLE_CONTEXT_CHECKPOINT must be 0 or 1")
    if context_checkpoint_flag == "1" and replica_flag != "1":
        raise ValueError(
            "PLE_CONTEXT_CHECKPOINT=1 requires PLE_OVERLAP_REPLICATED_GRADIENT=1"
        )
    defer_flag = os.environ.get("PLE_DEFER_TABLE_GRADIENT", "0")
    if defer_flag not in ("0", "1"):
        raise ValueError("PLE_DEFER_TABLE_GRADIENT must be 0 or 1")
    if defer_flag == "1" and not config.model.ple_async_training:
        raise ValueError("PLE_DEFER_TABLE_GRADIENT=1 requires PLE_ASYNC_TRAINING=1")
    embedding_flag = os.environ.get("PLE_FUSED_EMBEDDING", "0")
    if embedding_flag not in ("0", "1"):
        raise ValueError("PLE_FUSED_EMBEDDING must be 0 or 1")
    if embedding_flag == "1" and not config.model.ple_async_training:
        raise ValueError("PLE_FUSED_EMBEDDING=1 requires PLE_ASYNC_TRAINING=1")
    gather_flag = os.environ.get("PLE_FUSED_ROW_GATHER", "0")
    if gather_flag not in ("0", "1"):
        raise ValueError("PLE_FUSED_ROW_GATHER must be 0 or 1")
    if gather_flag == "1" and not config.model.ple_async_training:
        raise ValueError("PLE_FUSED_ROW_GATHER=1 requires PLE_ASYNC_TRAINING=1")
    overlap_flag = os.environ.get("PLE_OVERLAP_CPU_OPTIMIZER", "0")
    if overlap_flag not in ("0", "1"):
        raise ValueError("PLE_OVERLAP_CPU_OPTIMIZER must be 0 or 1")
    if overlap_flag == "1" and not config.model.ple_async_training:
        raise ValueError("PLE_OVERLAP_CPU_OPTIMIZER=1 requires PLE_ASYNC_TRAINING=1")
    route_flag = os.environ.get("PLE_OWNER_ROUTING", "0")
    if route_flag not in ("0", "1"):
        raise ValueError("PLE_OWNER_ROUTING must be 0 or 1")
    if route_flag == "1" and not config.model.ple_async_training:
        raise ValueError("PLE_OWNER_ROUTING=1 requires PLE_ASYNC_TRAINING=1")
    clip_flag = os.environ.get("PLE_ASYNC_CLIP", "0")
    exchange_flag = os.environ.get("PLE_TENSOR_EXCHANGE", "0")
    direct_flag = os.environ.get("PLE_DIRECT_GRADIENT", "0")
    update_flag = os.environ.get("PLE_FUSED_TABLE_UPDATE", "0")
    release_flag = os.environ.get("PLE_RELEASE_FETCH_CACHE", "0")
    norm_flag = os.environ.get("PLE_GPU_NORM", "0")
    prepare_flag = os.environ.get("PLE_PREPARE_UPDATE", "0")
    routing_flag = os.environ.get("PLE_PREPARE_ROUTING", "0")
    allocation_flag = os.environ.get("PLE_EARLY_GRADIENT", "0")
    streaming_flag = os.environ.get("PLE_STREAMING_UPDATE", "0")
    defer_join_flag = os.environ.get("PLE_DEFER_CPU_JOIN", "0")
    reuse_norm_flag = os.environ.get("PLE_REUSE_CONTEXT_NORM_INPUT", "0")
    offload_inputs_flag = os.environ.get("PLE_OFFLOAD_SAVED_INPUTS", "0")
    compact_inputs_flag = os.environ.get("PLE_COMPACT_SAVED_INPUTS", "0")
    audit_inputs_flag = os.environ.get("PLE_AUDIT_COMPACT_INPUTS", "0")
    try:
        update_threads = int(os.environ.get("PLE_CPU_UPDATE_THREADS", "0"))
    except ValueError as error:
        raise ValueError(
            "PLE_CPU_UPDATE_THREADS must be a nonnegative integer"
        ) from error
    if update_threads < 0:
        raise ValueError("PLE_CPU_UPDATE_THREADS must be a nonnegative integer")
    if update_threads and update_flag != "1":
        raise ValueError("PLE_CPU_UPDATE_THREADS requires PLE_FUSED_TABLE_UPDATE=1")
    for name, value in (
        ("PLE_ASYNC_CLIP", clip_flag),
        ("PLE_TENSOR_EXCHANGE", exchange_flag),
        ("PLE_DIRECT_GRADIENT", direct_flag),
        ("PLE_FUSED_TABLE_UPDATE", update_flag),
        ("PLE_RELEASE_FETCH_CACHE", release_flag),
        ("PLE_GPU_NORM", norm_flag),
        ("PLE_PREPARE_UPDATE", prepare_flag),
        ("PLE_PREPARE_ROUTING", routing_flag),
        ("PLE_EARLY_GRADIENT", allocation_flag),
        ("PLE_STREAMING_UPDATE", streaming_flag),
        ("PLE_DEFER_CPU_JOIN", defer_join_flag),
        ("PLE_REUSE_CONTEXT_NORM_INPUT", reuse_norm_flag),
        ("PLE_OFFLOAD_SAVED_INPUTS", offload_inputs_flag),
        ("PLE_COMPACT_SAVED_INPUTS", compact_inputs_flag),
        ("PLE_AUDIT_COMPACT_INPUTS", audit_inputs_flag),
    ):
        if value not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
        if value == "1" and not config.model.ple_async_training:
            raise ValueError(f"{name}=1 requires PLE_ASYNC_TRAINING=1")
    if direct_flag == "1" and defer_flag != "1":
        raise ValueError("PLE_DIRECT_GRADIENT=1 requires PLE_DEFER_TABLE_GRADIENT=1")
    if release_flag == "1" and early_flag != "1":
        raise ValueError("PLE_RELEASE_FETCH_CACHE=1 requires PLE_EARLY_FETCH=1")
    if norm_flag == "1" and direct_flag != "1":
        raise ValueError("PLE_GPU_NORM=1 requires PLE_DIRECT_GRADIENT=1")
    if update_flag == "1" and overlap_flag != "1":
        raise ValueError(
            "PLE_FUSED_TABLE_UPDATE=1 requires PLE_OVERLAP_CPU_OPTIMIZER=1"
        )
    if clip_flag == "1" and update_flag != "1":
        raise ValueError("PLE_ASYNC_CLIP=1 requires PLE_FUSED_TABLE_UPDATE=1")
    if prepare_flag == "1" and not all(
        flag == "1" for flag in (early_flag, defer_flag, update_flag)
    ):
        raise ValueError(
            "Prepared PLE updates require early fetch, deferred gradients and fused updates"
        )
    if routing_flag == "1" and route_flag != "1":
        raise ValueError("Prepared PLE routing requires owner routing")
    if allocation_flag == "1" and not (routing_flag == "1" and defer_flag == "1"):
        raise ValueError(
            "Early PLE gradients require prepared routing and deferred gradients"
        )
    if streaming_flag == "1" and prepare_flag != "1":
        raise ValueError("Streaming PLE updates require prepared updates")
    if defer_join_flag == "1" and not (early_flag == "1" and update_flag == "1"):
        raise ValueError("Deferred PLE join requires early fetch and fused CPU updates")
    if reuse_norm_flag == "1" and context_checkpoint_flag != "1":
        raise ValueError("Context norm input reuse requires context checkpointing")
    if offload_inputs_flag == "1" and not (save_flag == "1" and early_flag == "1"):
        raise ValueError(
            "Saved PLE input staging requires saved matmuls and early fetch"
        )
    if compact_inputs_flag == "1" and offload_inputs_flag != "1":
        raise ValueError("Compact PLE inputs require saved input staging")
    if audit_inputs_flag == "1" and compact_inputs_flag != "1":
        raise ValueError("PLE input audits require compact saved inputs")
    if config.model.ple is None:
        raise ValueError("Set PLE=ple for the row-sharded DRAM recipe")
    if os.environ.get("PLE_COMPILE_FP8", "0") == "1":
        config.compile.regions.append("float8_quantize")
    if not config.model.ple_async_training:
        return config
    config.model = _convert(config.model, RowShardedGlimmer.Config)
    config.model.ple = _convert(config.model.ple, RowShardedPLE.Config)
    config.model.ple.fused_row_gather = gather_flag == "1"
    config.model.ple.fused_embedding = embedding_flag == "1"
    config.model.ple.defer_table_gradient = defer_flag == "1"
    config.model.ple.direct_gradient_copy = direct_flag == "1"
    config.model.ple.gpu_norm = norm_flag == "1"
    config.model.ple.owner_routing = route_flag == "1"
    config.model.ple.tensor_exchange = exchange_flag == "1"
    config.model.ple.prepare_routing = routing_flag == "1"
    config.model.ple.early_gradient = allocation_flag == "1"
    config.model.ple.offload_saved_inputs = offload_inputs_flag == "1"
    config.model.ple.compact_saved_inputs = compact_inputs_flag == "1"
    config.model.ple.audit_compact_inputs = audit_inputs_flag == "1"
    if save_flag == "1":
        from torchtitan.distributed.activation_checkpoint import SelectiveAC

        from .checkpoint import SavedContext, SavedLayerPLE, SavePLEMatmuls

        if not isinstance(config.activation_checkpoint, SelectiveAC.Config):
            raise ValueError(
                "PLE_SAVE_MATMULS=1 requires selective activation checkpointing"
            )
        count = int(os.environ.get("PLE_SAVE_LAYERS", str(len(config.model.layers))))
        if not 0 <= count <= len(config.model.layers):
            raise ValueError("PLE_SAVE_LAYERS must be between 0 and the layer count")
        for layer in config.model.layers:
            if layer.layer_ple is not None:
                layer.layer_ple = _convert(layer.layer_ple, SavedLayerPLE.Config)
                layer.layer_ple.overlap_replicated_gradient = replica_flag == "1"
        if replica_flag == "1":
            config.model.ple.context = _convert(
                config.model.ple.context, SavedContext.Config
            )
            config.model.ple.context.activation_checkpoint = (
                context_checkpoint_flag == "1"
            )
            config.model.ple.context.reuse_norm_input = reuse_norm_flag == "1"
        config.activation_checkpoint = _convert(
            config.activation_checkpoint, SavePLEMatmuls.Config
        )
        config.activation_checkpoint.num_layers_to_save = count
    if early_flag == "1":
        from .prefetch import EarlyFetchGlimmer, EarlyFetchPLE, EarlyFetchTrainer

        config = _convert(config, EarlyFetchTrainer.Config)
        config.release_fetch_cache = release_flag == "1"
        config.model = _convert(config.model, EarlyFetchGlimmer.Config)
        config.model.ple = _convert(config.model.ple, EarlyFetchPLE.Config)
    config.optim = _convert(config.optim, RowShardedOptim.Config)
    config.optim.fused_cpu_update = update_flag == "1"
    config.optim.async_clip = clip_flag == "1"
    config.optim.cpu_update_threads = update_threads
    config.optim.prepare_zero_gradient = prepare_flag == "1"
    config.optim.streaming_update = streaming_flag == "1"
    if overlap_flag == "1":
        config.optim.optimizer = _convert(
            config.optim.optimizer, CPUOverlapOptimizersContainer.Config
        )
        config.optim.optimizer.defer_cpu_join = defer_join_flag == "1"
    if not config.optim.cpu_norm_chunk_size:
        config.optim.cpu_norm_chunk_size = 65536
    return config


def glimmer_dense_dp():
    """Use a replicated GPU backbone with either native or DRAM PLE."""
    config = glimmer_dense()
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("NGPU", "1")))
    config.parallelism.data_parallel_replicate_degree = world
    config.parallelism.data_parallel_shard_degree = 1
    if not config.model.ple_async_training:
        config.optim.optimizer.optimizers.insert(
            -1,
            replace(
                config.optim.optimizer.optimizers[-1],
                pattern=r"(?:^ple\.|\.layer_ple\.)",
                fused=True,
                foreach=False,
            ),
        )
    return config
