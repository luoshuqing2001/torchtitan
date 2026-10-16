# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Moonlight Muon for Muse Glimmer dense and MoE recipes.

Matrix parameters use DistMuon with ``adjust_lr_fn="match_rms_adamw"``
(``0.2 * sqrt(max(rows, columns))``), the update scaling from Moonlight /
Kimi K2. Embeddings, RMSNorms, biases, and the tied LM head stay on AdamW.
See https://arxiv.org/abs/2502.16982.
"""

from torch.distributed.tensor import Shard

from torchtitan.components.optim import (
    AdamW,
    DistMuon,
    LRSchedulersContainer,
    OptimizersContainer,
)
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.parallelism_context import MeshAxisName

# Warmup-Stable-Decay (Hu et al., 2024). Ten percent of the token budget in
# the decay stage closes most of the gap to a full cosine schedule, so the
# stable stage keeps the peak learning rate for the rest. Cosine decay starts
# with zero slope, so the constant stage does not kink into the decay.
_WARMUP_RATIO = 0.02
_DECAY_RATIO = 0.10
_MIN_LR_FACTOR = 0.1


def glimmer_wsd_scheduler(steps: int) -> LRSchedulersContainer.Config:
    """WSD: 2% linear warmup, 88% stable, 10% cosine decay to 1/10 of peak."""
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    return LRSchedulersContainer.Config(
        warmup_steps=max(1, round(steps * _WARMUP_RATIO)),
        decay_ratio=_DECAY_RATIO,
        decay_type="cosine",
        min_lr_factor=_MIN_LR_FACTOR,
    )


_MUON_PATTERN = (
    r"(?:"
    r"attention\.qkv_linear\.wqkv\.weight|"
    r"attention\.wo\.weight|"
    r"attention\.o_gate\.weight|"
    r"shared_experts\.\d+\.w(?:13|2)\.weight|"
    r"moe\.routed_experts\.w(?:13|2)\.weight|"
    r"moe\.router\.gate\.weight|"
    r"feed_forward\.w(?:13|2)\.weight"
    r")$"
)


def _owned(axis: str) -> ComputeLayout:
    return ComputeLayout(shardings_by_mesh_axis={axis: Owned()})


def _matrix_rows(axis: str) -> ComputeLayout:
    """Shard a stacked ``[N, rows, columns]`` tensor across its matrix batch."""
    return ComputeLayout(shardings_by_mesh_axis={axis: Shard(0)})


def _per_head(axis: str, head_dim: int) -> ComputeLayout:
    """Newton-Schulz on each ``head_dim`` row block of a fused projection."""
    return ComputeLayout(
        shardings_by_mesh_axis={
            axis: BlockShard(dim=0, block_sizes=(head_dim,)),
        }
    )


def _per_expert_compute_layout(parallelism: ParallelismConfig) -> ComputeLayout:
    ep_size = parallelism.expert_parallel_degree
    if ep_size <= 0:
        raise ValueError("expert_parallel_degree must be positive")
    if ep_size == 1:
        return _matrix_rows(MeshAxisName.DP_SHARD.value)

    # EP splits the expert dimension first. edp_shard then repartitions each
    # EP-local expert domain, which reverses the storage-mesh axis order.
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.EDP_SHARD.value: Shard(0),
            MeshAxisName.EP.value: Shard(0),
        },
        shard_order_by_tensor_dim={
            0: (MeshAxisName.EP.value, MeshAxisName.EDP_SHARD.value),
        },
    )


def glimmer_muon_optimizer(
    model_config,
    *,
    lr: float,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
    """Build the Moonlight Muon + AdamW split for one Glimmer model config."""
    if parallelism.tensor_parallel_degree != 1:
        raise ValueError(
            "Glimmer DistMuon requires tensor_parallel_degree=1. "
            "Tensor parallel can produce _StridedShard layouts DistMuon rejects."
        )

    dense_axis = MeshAxisName.DP_SHARD.value
    per_expert = _per_expert_compute_layout(parallelism)
    compute_sharding_by_fqn: dict[str, ComputeLayout] = {}
    layer_matrices: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    for layer_id, layer in enumerate(model_config.layers):
        attention = layer.attention
        head_dim = attention.head_dim
        if head_dim is None:
            raise ValueError(f"layer {layer_id} attention has no head_dim")
        prefix = f"layers.{layer_id}"
        matrices = {
            f"{prefix}.attention.qkv_linear.wqkv.weight": _per_head(
                dense_axis, head_dim
            ),
            f"{prefix}.attention.wo.weight": _owned(dense_axis),
        }
        if attention.o_gate is not None:
            matrices[f"{prefix}.attention.o_gate.weight"] = _per_head(
                dense_axis, head_dim
            )

        routed: list[str] = []
        feed_forward = getattr(layer, "feed_forward", None)
        if feed_forward is not None:
            matrices[f"{prefix}.feed_forward.w13.weight"] = _matrix_rows(dense_axis)
            matrices[f"{prefix}.feed_forward.w2.weight"] = _owned(dense_axis)
        for expert_id, _expert in enumerate(getattr(layer, "shared_experts", ())):
            matrices[f"{prefix}.shared_experts.{expert_id}.w13.weight"] = (
                _matrix_rows(dense_axis)
            )
            matrices[f"{prefix}.shared_experts.{expert_id}.w2.weight"] = _owned(
                dense_axis
            )
        moe = getattr(layer, "moe", None)
        if moe is not None:
            for projection in ("w13.weight", "w2.weight"):
                fqn = f"{prefix}.moe.routed_experts.{projection}"
                matrices[fqn] = per_expert
                routed.append(fqn)
            matrices[f"{prefix}.moe.router.gate.weight"] = _owned(dense_axis)

        compute_sharding_by_fqn.update(matrices)
        layer_matrices.append((tuple(matrices), tuple(routed)))

    bucket_configs: list[BucketConfig] = []
    for first_layer_id in range(0, len(layer_matrices), 2):
        group = layer_matrices[first_layer_id : first_layer_id + 2]
        name = "layers." + "-".join(
            str(first_layer_id + offset) for offset in range(len(group))
        )
        non_routed = tuple(
            fqn
            for matrices, routed in group
            for fqn in matrices
            if fqn not in routed
        )
        routed = tuple(fqn for _matrices, routed_fqns in group for fqn in routed_fqns)
        bucket_configs.append(BucketConfig(name=name, patterns=non_routed))
        if routed:
            bucket_configs.append(
                BucketConfig(name=f"{name}.routed-experts", patterns=routed)
            )

    return OptimizersContainer.Config(
        optimizers=[
            DistMuon.Config(
                pattern=_MUON_PATTERN,
                bucket_configs=tuple(bucket_configs),
                compute_sharding_by_fqn=compute_sharding_by_fqn,
                lr=lr,
                weight_decay=0.1,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                adjust_lr_fn="match_rms_adamw",
            ),
            AdamW.Config(
                pattern=r".*",
                lr=lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
                foreach=True,
                fused=False,
            ),
        ],
    )
