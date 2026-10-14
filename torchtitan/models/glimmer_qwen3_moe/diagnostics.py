# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-layer training diagnostics for the Glimmer + Qwen3-MoE model.

Each block records one observation per forward. Activation-checkpoint
recomputation is skipped, matching the MoE router's token counter. The
metrics processor reduces the observations across layers and ranks at log
time, then clears the buffers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.distributed import ParallelismContext
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.protocols.module import Module

logger = logging.getLogger(__name__)

STAT_NAMES: tuple[str, ...] = (
    "expert_load_cv",
    "expert_load_max_over_mean",
    "expert_dead_fraction",
    "router_entropy_norm",
    "router_max_prob",
    "hidden_outlier_fraction",
    "hidden_massive_fraction",
    "hidden_abs_max",
    "hidden_rms",
    "attn_out_rms",
    "moe_out_rms",
    "expert_bias_abs_max",
)
_MAX_STATS = {
    "expert_load_max_over_mean",
    "expert_dead_fraction",
    "hidden_outlier_fraction",
    "hidden_massive_fraction",
    "hidden_abs_max",
    "expert_bias_abs_max",
}
_OUTLIER_RMS_FACTOR = 6.0
_MASSIVE_ABS = 100.0


def _local(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


class LayerDiagnostics(Module):
    """Running sums of one layer's internal training statistics."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "stat_sum",
            torch.zeros(len(STAT_NAMES), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "stat_max",
            torch.full((len(STAT_NAMES),), -torch.inf),
            persistent=False,
        )
        self.register_buffer(
            "stat_count",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device or self.stat_sum.device
        with torch.device(device):
            self.stat_sum = torch.zeros(len(STAT_NAMES), dtype=torch.float32)
            self.stat_max = torch.full((len(STAT_NAMES),), -torch.inf)
            self.stat_count = torch.zeros((), dtype=torch.float32)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = _local(values).detach().to(
            dtype=torch.float32, device=self.stat_sum.device
        )
        self.stat_sum.add_(values)
        self.stat_max.copy_(torch.maximum(self.stat_max, values))
        self.stat_count.add_(1)


def routing_and_hidden_stats(
    *,
    scores_TE: torch.Tensor,
    top_k: int,
    num_experts: int,
    hidden_TD: torch.Tensor,
    attn_out_TD: torch.Tensor,
    moe_out_TD: torch.Tensor,
    expert_bias_E: torch.Tensor | None,
    padding_mask_T: torch.Tensor | None,
) -> torch.Tensor:
    """Return one observation in ``STAT_NAMES`` order."""
    scores_TE = _local(scores_TE).float()
    hidden_TD = _local(hidden_TD).float()
    attn_out_TD = _local(attn_out_TD).float()
    moe_out_TD = _local(moe_out_TD).float()
    bias_E = None if expert_bias_E is None else _local(expert_bias_E).float()
    if padding_mask_T is not None:
        valid_T = ~_local(padding_mask_T).bool()
        scores_TE = scores_TE[valid_T]
        hidden_TD = hidden_TD[valid_T]
        attn_out_TD = attn_out_TD[valid_T]
        moe_out_TD = moe_out_TD[valid_T]

    choice_TE = scores_TE if bias_E is None else scores_TE + bias_E
    if choice_TE.numel():
        expert_ids_TK = torch.topk(choice_TE, k=top_k, dim=-1, sorted=False).indices
    else:
        expert_ids_TK = scores_TE.new_empty((0, top_k), dtype=torch.long)

    counts_E = torch.zeros(num_experts, dtype=torch.float32, device=scores_TE.device)
    if expert_ids_TK.numel():
        counts_E.scatter_add_(
            0,
            expert_ids_TK.reshape(-1).to(torch.long),
            torch.ones(
                expert_ids_TK.numel(),
                dtype=torch.float32,
                device=scores_TE.device,
            ),
        )
    mean_count = counts_E.mean().clamp_min(1.0)
    load_cv = counts_E.std(unbiased=False) / mean_count
    load_max_over_mean = counts_E.max() / mean_count
    dead_fraction = (counts_E == 0).float().mean()

    if scores_TE.numel():
        log_probs = scores_TE.clamp_min(1e-9).log()
        entropy = -(scores_TE * log_probs).sum(dim=-1).mean()
        entropy_norm = entropy / torch.log(
            torch.tensor(float(num_experts), device=scores_TE.device)
        )
        max_prob = scores_TE.max(dim=-1).values.mean()
    else:
        entropy_norm = scores_TE.new_zeros(())
        max_prob = scores_TE.new_zeros(())

    hidden_rms = hidden_TD.square().mean().sqrt()
    outlier_threshold = torch.maximum(
        hidden_rms * _OUTLIER_RMS_FACTOR,
        hidden_rms.new_tensor(float(_MASSIVE_ABS)),
    )
    hidden_outlier = (hidden_TD.abs() > outlier_threshold).float().mean()
    hidden_massive = (hidden_TD.abs() > _MASSIVE_ABS).float().mean()
    hidden_abs_max = (
        hidden_TD.abs().amax() if hidden_TD.numel() else hidden_TD.new_zeros(())
    )
    attn_rms = attn_out_TD.square().mean().sqrt()
    moe_rms = moe_out_TD.square().mean().sqrt()
    if expert_bias_E is None:
        bias_abs_max = scores_TE.new_zeros(())
    else:
        bias_abs_max = bias_E.abs().amax()

    return torch.stack(
        [
            load_cv,
            load_max_over_mean,
            dead_fraction,
            entropy_norm,
            max_prob,
            hidden_outlier,
            hidden_massive,
            hidden_abs_max,
            hidden_rms,
            attn_rms,
            moe_rms,
            bias_abs_max,
        ]
    )


def collect_layer_diagnostics(
    model_parts: list[nn.Module] | None,
    parallelism_context: ParallelismContext | None,
) -> dict[str, float]:
    """Mean and selected max diagnostics since the previous log."""
    del parallelism_context
    if not model_parts:
        return {}
    sums = []
    maxes = []
    counts = []
    modules: list[LayerDiagnostics] = []
    for part in model_parts:
        for module in part.modules():
            if isinstance(module, LayerDiagnostics):
                modules.append(module)
                sums.append(_local(module.stat_sum))
                maxes.append(_local(module.stat_max))
                counts.append(_local(module.stat_count))
    if not modules:
        return {}

    device = sums[0].device
    total_sum = torch.stack([value.to(device) for value in sums]).sum(dim=0)
    total_max = torch.stack([value.to(device) for value in maxes]).amax(dim=0)
    total_count = torch.stack([value.to(device) for value in counts]).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
    count = float(total_count.item())
    for module in modules:
        module.stat_sum.zero_()
        module.stat_max.fill_(-torch.inf)
        module.stat_count.zero_()
    if count <= 0:
        return {}

    mean = total_sum / count
    metrics: dict[str, float] = {}
    for index, name in enumerate(STAT_NAMES):
        metrics[f"diagnostics/{name}"] = float(mean[index].item())
        if name in _MAX_STATS:
            metrics[f"diagnostics/{name}_max"] = float(total_max[index].item())
    return metrics


class DiagnosticMetricsProcessor(MetricsProcessor):
    """Logs throughput metrics plus MoE balance and activation outliers."""

    @dataclass(kw_only=True, slots=True)
    class Config(MetricsProcessor.Config):
        pass

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict | None = None,
    ) -> None:
        diagnostics = collect_layer_diagnostics(
            self.model_parts, self.parallelism_context
        )
        merged = dict(extra_metrics or {})
        merged.update(diagnostics)
        super().log(
            step,
            global_avg_loss,
            global_max_loss,
            grad_norm,
            extra_metrics=merged,
        )
        line = _format_diagnostics(diagnostics)
        if line:
            logger.info("diagnostics  %s", line)


def _format_diagnostics(metrics: dict[str, float]) -> str:
    wanted = (
        "diagnostics/expert_load_cv",
        "diagnostics/expert_load_max_over_mean_max",
        "diagnostics/expert_dead_fraction_max",
        "diagnostics/router_entropy_norm",
        "diagnostics/router_max_prob",
        "diagnostics/hidden_outlier_fraction_max",
        "diagnostics/hidden_massive_fraction_max",
        "diagnostics/hidden_abs_max_max",
        "diagnostics/hidden_rms",
        "diagnostics/expert_bias_abs_max_max",
    )
    parts = []
    for key in wanted:
        if key in metrics:
            parts.append(f"{key.removeprefix('diagnostics/')}: {metrics[key]:.4g}")
    return "  ".join(parts)
