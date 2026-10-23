# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Engram CPU row ownership; ordinary GPU training remains the default."""

import os
from dataclasses import fields, replace

from torchtitan_recipes.models.glimmer_dense import glimmer_dense as base_recipe

from torchtitan.experiments.ple_dram_shard.optim import RowShardedOptim
from torchtitan.experiments.ple_dram_shard.overlap import CPUOverlapOptimizersContainer

from .model import DRAMEngram, DRAMEngramModel


def _convert(config, cls):
    return cls(**{field.name: getattr(config, field.name) for field in fields(config)})


def glimmer_dense():
    flags = {}
    for name in (
        "ENGRAM_ASYNC_TRAINING",
        "ENGRAM_DRAM_COMPACT_INDICES",
        "ENGRAM_DRAM_CACHE_STAGES",
        "ENGRAM_DRAM_EARLY_FETCH",
        "ENGRAM_DRAM_FUSED_GATHER",
        "ENGRAM_DRAM_OWNER_ROUTING",
        "ENGRAM_DRAM_DEFER_GRADIENT",
        "ENGRAM_DRAM_DIRECT_GRADIENT",
        "ENGRAM_DRAM_GPU_NORM",
        "ENGRAM_DRAM_CPU_OVERLAP",
        "ENGRAM_DRAM_FUSED_UPDATE",
        "ENGRAM_DRAM_ASYNC_CLIP",
        "ENGRAM_DRAM_STREAMING_UPDATE",
        "ENGRAM_DRAM_PREPARE_UPDATE",
        "ENGRAM_DRAM_PREPARE_ROUTING",
        "ENGRAM_DRAM_EARLY_GRADIENT",
        "ENGRAM_DRAM_BOUNDED_UNIQUE",
        "ENGRAM_DRAM_GPU_INDICES",
        "ENGRAM_DRAM_TENSOR_EXCHANGE",
        "ENGRAM_DRAM_GPU_TABLE_CACHE",
        "ENGRAM_DRAM_DOUBLE_BUFFER_CACHE",
        "ENGRAM_DRAM_SINGLE_BUFFER_CACHE",
        "ENGRAM_DRAM_TRANSIENT_CACHE_WORKSPACE",
        "ENGRAM_DRAM_DEFER_CPU_JOIN",
        "ENGRAM_DRAM_REUSE_CACHE_WORKSPACE",
        "ENGRAM_DRAM_GPU_PROJECTIONS",
        "ENGRAM_DRAM_OFFLOAD_VALUE",
        "ENGRAM_DRAM_CHECKPOINT_VALUE",
        "ENGRAM_DRAM_PREFETCH_VALUE",
        "ENGRAM_DRAM_ALIGN_VALUE",
        "ENGRAM_DRAM_ALIGN_VALUE_FORWARD",
        "ENGRAM_COMPILE_FP8",
    ):
        flag = os.environ.get(name, "0")
        if flag not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
        flags[name] = flag == "1"
    config = base_recipe()
    if config.model.engram is None:
        raise ValueError("Set PLE=engram_legacy for the Engram DRAM recipe")
    if flags["ENGRAM_COMPILE_FP8"]:
        config.compile.regions.append("float8_quantize")
    asynchronous = flags["ENGRAM_ASYNC_TRAINING"]
    for name, enabled in flags.items():
        if name.startswith("ENGRAM_DRAM_") and enabled and not asynchronous:
            raise ValueError(f"{name}=1 requires ENGRAM_ASYNC_TRAINING=1")
    if not asynchronous:
        return config
    if flags["ENGRAM_DRAM_DIRECT_GRADIENT"] and not flags["ENGRAM_DRAM_DEFER_GRADIENT"]:
        raise ValueError("Direct gradients require deferred gradients")
    if flags["ENGRAM_DRAM_GPU_NORM"] and not flags["ENGRAM_DRAM_DIRECT_GRADIENT"]:
        raise ValueError("GPU norms require direct gradients")
    if flags["ENGRAM_DRAM_ASYNC_CLIP"] and not flags["ENGRAM_DRAM_FUSED_UPDATE"]:
        raise ValueError("Asynchronous clipping requires fused CPU updates")
    if flags["ENGRAM_DRAM_FUSED_UPDATE"] and not flags["ENGRAM_DRAM_CPU_OVERLAP"]:
        raise ValueError("Fused updates require CPU optimizer overlap")
    if flags["ENGRAM_DRAM_DEFER_CPU_JOIN"] and not (
        flags["ENGRAM_DRAM_EARLY_FETCH"] and flags["ENGRAM_DRAM_FUSED_UPDATE"]
    ):
        raise ValueError(
            "Deferred Engram join requires early fetch and fused CPU updates"
        )
    if flags["ENGRAM_DRAM_PREPARE_UPDATE"] and not all(
        flags[name]
        for name in (
            "ENGRAM_DRAM_EARLY_FETCH",
            "ENGRAM_DRAM_COMPACT_INDICES",
            "ENGRAM_DRAM_DEFER_GRADIENT",
            "ENGRAM_DRAM_FUSED_UPDATE",
        )
    ):
        raise ValueError(
            "Prepared updates require early compact fetch, deferred gradients and fused updates"
        )
    if flags["ENGRAM_DRAM_PREPARE_ROUTING"] and not all(
        flags[name]
        for name in (
            "ENGRAM_DRAM_COMPACT_INDICES",
            "ENGRAM_DRAM_OWNER_ROUTING",
        )
    ):
        raise ValueError("Prepared routing requires compact indices and owner routing")
    if flags["ENGRAM_DRAM_EARLY_GRADIENT"] and not (
        flags["ENGRAM_DRAM_PREPARE_ROUTING"] and flags["ENGRAM_DRAM_DEFER_GRADIENT"]
    ):
        raise ValueError(
            "Early gradient allocation requires prepared routing and deferred gradients"
        )
    if (
        flags["ENGRAM_DRAM_BOUNDED_UNIQUE"]
        or flags["ENGRAM_DRAM_GPU_INDICES"]
        or flags["ENGRAM_DRAM_TENSOR_EXCHANGE"]
    ) and not flags["ENGRAM_DRAM_COMPACT_INDICES"]:
        raise ValueError("Bounded unique and GPU indices require compact indices")
    if flags["ENGRAM_DRAM_GPU_TABLE_CACHE"] and not (
        flags["ENGRAM_DRAM_COMPACT_INDICES"] and flags["ENGRAM_DRAM_FUSED_UPDATE"]
    ):
        raise ValueError(
            "GPU table cache requires compact indices and fused CPU updates"
        )
    if flags["ENGRAM_DRAM_DOUBLE_BUFFER_CACHE"] and not (
        flags["ENGRAM_DRAM_GPU_TABLE_CACHE"]
        and flags["ENGRAM_DRAM_PREPARE_UPDATE"]
        and config.training.mixed_precision_param == "bfloat16"
    ):
        raise ValueError(
            "Double buffering requires BF16 table cache and prepared updates"
        )
    if flags["ENGRAM_DRAM_OFFLOAD_VALUE"] and not flags["ENGRAM_DRAM_EARLY_FETCH"]:
        raise ValueError("Value offload requires early fetch for exception cleanup")
    if (
        flags["ENGRAM_DRAM_STREAMING_UPDATE"]
        and not flags["ENGRAM_DRAM_PREPARE_UPDATE"]
    ):
        raise ValueError("Streaming stores require prepared updates")
    config.model = _convert(config.model, DRAMEngramModel.Config)
    config.model.engram = _convert(config.model.engram, DRAMEngram.Config)
    config.model.engram.compact_indices = flags["ENGRAM_DRAM_COMPACT_INDICES"]
    config.model.engram.prepare_routing = flags["ENGRAM_DRAM_PREPARE_ROUTING"]
    config.model.engram.early_gradient = flags["ENGRAM_DRAM_EARLY_GRADIENT"]
    config.model.engram.bounded_unique = flags["ENGRAM_DRAM_BOUNDED_UNIQUE"]
    config.model.engram.gpu_indices = flags["ENGRAM_DRAM_GPU_INDICES"]
    config.model.engram.tensor_exchange = flags["ENGRAM_DRAM_TENSOR_EXCHANGE"]
    config.model.engram.gpu_table_cache = flags["ENGRAM_DRAM_GPU_TABLE_CACHE"]
    config.model.engram.double_buffer_cache = flags["ENGRAM_DRAM_DOUBLE_BUFFER_CACHE"]
    config.model.engram.single_buffer_cache = flags["ENGRAM_DRAM_SINGLE_BUFFER_CACHE"]
    config.model.engram.transient_cache_workspace = flags[
        "ENGRAM_DRAM_TRANSIENT_CACHE_WORKSPACE"
    ]
    if (
        flags["ENGRAM_DRAM_TRANSIENT_CACHE_WORKSPACE"]
        and not flags["ENGRAM_DRAM_SINGLE_BUFFER_CACHE"]
    ):
        raise ValueError("Transient cache workspace requires single buffering")
    config.model.engram.cache_patch_chunk_rows = int(
        os.environ.get("ENGRAM_DRAM_CACHE_PATCH_CHUNK_ROWS", "262144")
    )
    if config.model.engram.cache_patch_chunk_rows <= 0:
        raise ValueError("Cache patch chunk rows must be positive")
    if flags["ENGRAM_DRAM_SINGLE_BUFFER_CACHE"] and not (
        flags["ENGRAM_DRAM_DOUBLE_BUFFER_CACHE"]
        and flags["ENGRAM_DRAM_REUSE_CACHE_WORKSPACE"]
        and flags["ENGRAM_DRAM_EARLY_FETCH"]
    ):
        raise ValueError(
            "Single buffering requires double-buffer scheduling, workspace reuse and early fetch"
        )
    if (
        flags["ENGRAM_DRAM_REUSE_CACHE_WORKSPACE"]
        and not flags["ENGRAM_DRAM_DOUBLE_BUFFER_CACHE"]
    ):
        raise ValueError("Cache workspace reuse requires double buffering")
    config.model.engram.reuse_cache_workspace = flags[
        "ENGRAM_DRAM_REUSE_CACHE_WORKSPACE"
    ]
    config.model.engram.gpu_projections = flags["ENGRAM_DRAM_GPU_PROJECTIONS"]
    if flags["ENGRAM_DRAM_PREFETCH_VALUE"] and (
        flags["ENGRAM_DRAM_CHECKPOINT_VALUE"]
        or flags["ENGRAM_DRAM_OFFLOAD_VALUE"]
        or flags["ENGRAM_DRAM_GPU_PROJECTIONS"]
    ):
        raise ValueError(
            "Prefetched value recomputation requires CPU projections without value offload or checkpointing"
        )
    if flags["ENGRAM_DRAM_PREFETCH_VALUE"] and not flags["ENGRAM_DRAM_EARLY_FETCH"]:
        raise ValueError(
            "Prefetched value recomputation requires early fetch for exception cleanup"
        )
    if flags["ENGRAM_DRAM_CHECKPOINT_VALUE"] and (
        flags["ENGRAM_DRAM_GPU_PROJECTIONS"] or flags["ENGRAM_DRAM_OFFLOAD_VALUE"]
    ):
        raise ValueError(
            "Value checkpointing requires CPU projections without value offload"
        )
    if flags["ENGRAM_DRAM_ALIGN_VALUE"] and not (
        flags["ENGRAM_DRAM_PREFETCH_VALUE"]
        and config.training.mixed_precision_param == "bfloat16"
    ):
        raise ValueError("Aligned value recomputation requires BF16 prefetched values")
    if (
        flags["ENGRAM_DRAM_ALIGN_VALUE_FORWARD"]
        and not flags["ENGRAM_DRAM_ALIGN_VALUE"]
    ):
        raise ValueError("Aligned value forward requires aligned prefetched values")
    config.model.engram.align_value_forward = flags["ENGRAM_DRAM_ALIGN_VALUE_FORWARD"]
    config.model.engram.align_value = flags["ENGRAM_DRAM_ALIGN_VALUE"]
    config.model.engram.prefetch_value = flags["ENGRAM_DRAM_PREFETCH_VALUE"]
    config.model.engram.checkpoint_value = flags["ENGRAM_DRAM_CHECKPOINT_VALUE"]
    config.model.engram.offload_value = flags["ENGRAM_DRAM_OFFLOAD_VALUE"]
    config.model.engram.fetch_threads = int(
        os.environ.get("ENGRAM_DRAM_FETCH_THREADS", "0")
    )
    if config.model.engram.fetch_threads < 0:
        raise ValueError("ENGRAM_DRAM_FETCH_THREADS must be nonnegative")
    if config.model.engram.fetch_threads or config.model.engram.bounded_unique:
        from torchtitan.experiments.ple_dram_shard.cpu_update import load_kernel

        load_kernel()
    config.model.engram.cache_stages = flags["ENGRAM_DRAM_CACHE_STAGES"]
    config.model.engram.fused_gather = flags["ENGRAM_DRAM_FUSED_GATHER"]
    config.model.engram.owner_routing = flags["ENGRAM_DRAM_OWNER_ROUTING"]
    config.model.engram.defer_gradient = flags["ENGRAM_DRAM_DEFER_GRADIENT"]
    config.model.engram.direct_gradient = flags["ENGRAM_DRAM_DIRECT_GRADIENT"]
    config.model.engram.gpu_norm = flags["ENGRAM_DRAM_GPU_NORM"]
    if flags["ENGRAM_DRAM_EARLY_FETCH"]:
        from .prefetch import EarlyEngramModel, EarlyEngramTrainer

        config = _convert(config, EarlyEngramTrainer.Config)
        config.model = _convert(config.model, EarlyEngramModel.Config)
    config.optim.optimizer.optimizers.insert(
        -1,
        replace(
            config.optim.optimizer.optimizers[-1],
            pattern=(
                r"^engram\.table\."
                if flags["ENGRAM_DRAM_GPU_PROJECTIONS"]
                else r"^engram\."
            ),
            fused=True,
            foreach=False,
        ),
    )
    if flags["ENGRAM_DRAM_CPU_OVERLAP"]:
        config.optim.optimizer = _convert(
            config.optim.optimizer, CPUOverlapOptimizersContainer.Config
        )
        config.optim.optimizer.defer_cpu_join = flags["ENGRAM_DRAM_DEFER_CPU_JOIN"]
    config.optim = _convert(config.optim, RowShardedOptim.Config)
    config.optim.table_path = "engram.table"
    config.optim.async_clip = flags["ENGRAM_DRAM_ASYNC_CLIP"]
    config.optim.streaming_update = flags["ENGRAM_DRAM_STREAMING_UPDATE"]
    config.optim.prepare_zero_gradient = flags["ENGRAM_DRAM_PREPARE_UPDATE"]
    config.optim.fused_cpu_update = flags["ENGRAM_DRAM_FUSED_UPDATE"]
    config.optim.cpu_update_threads = int(
        os.environ.get("ENGRAM_DRAM_CPU_THREADS", "0")
    )
    if config.optim.cpu_update_threads < 0:
        raise ValueError("ENGRAM_DRAM_CPU_THREADS must be nonnegative")
    if config.optim.cpu_update_threads and not config.optim.fused_cpu_update:
        raise ValueError("CPU thread tuning requires fused CPU updates")
    config.optim.cpu_norm_chunk_size = int(
        os.environ.get("CPU_NORM_CHUNK_SIZE", "65536")
    )
    return config


def glimmer_dense_dp():
    config = glimmer_dense()
    config.parallelism.data_parallel_replicate_degree = int(
        os.environ.get("WORLD_SIZE", os.environ.get("NGPU", "1"))
    )
    config.parallelism.data_parallel_shard_degree = 1
    return config
