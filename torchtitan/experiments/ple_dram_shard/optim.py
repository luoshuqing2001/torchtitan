# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Clip owned CPU table shards with the replicated global gradient norm."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from torchtitan.components.optim.optim import Optim
from torchtitan.distributed import utils


@torch.no_grad()
def clip_owned(parameters, owned, ownership, max_norm, chunk_size, table_update=None):
    if owned.grad is None:
        raise ValueError("The owned table gradient is missing")
    if chunk_size <= 0 or owned.numel() % chunk_size:
        raise ValueError("Each owned shard must contain whole CPU norm chunks")
    if ownership.vocab_size * ownership.width // chunk_size > chunk_size:
        raise ValueError("Global norm summary must fit in one CPU norm chunk")
    # Concatenating complete chunks preserves the original CPU reduction order.
    cached = getattr(owned.grad, "_ple_owned_norm_cache", None)
    if cached is not None and cached[:2] == (chunk_size, owned.grad._version):
        local_norms = cached[2]
    else:
        local_norms = torch.linalg.vector_norm(
            owned.grad.view(-1, chunk_size), ord=2, dim=1
        )
    all_norms = [
        torch.empty_like(local_norms)
        for _ in range(dist.get_world_size(ownership.group))
    ]
    dist.all_gather(all_norms, local_norms, group=ownership.group)
    norms = torch.cat(all_norms)
    # Replicated summaries compose with native mixed CPU/FSDP gradient clipping.
    proxy = nn.Parameter(torch.empty_like(norms), requires_grad=False)
    proxy.grad = norms
    parameters = [proxy if param is owned else param for param in parameters]
    if table_update is not None and table_update.async_clip:
        from .clipping import total_norm_async

        gradients = utils._cpu_norm_inputs(
            [p.grad for p in parameters if p.grad is not None], chunk_size
        )
        norm = total_norm_async(gradients)
        actual = [p for p in parameters if p is not proxy]
        gpu = [p for p in actual if p.device.type == "cuda"]
        cpu = [p for p in actual if p.device.type == "cpu"]
        torch.nn.utils.clip_grads_with_norm_(gpu, max_norm, norm, foreach=True)
        table_update.defer_async(norm, max_norm, cpu)
        return norm
    norm = utils.clip_grad_norm_(
        parameters, max_norm, foreach=True, cpu_norm_chunk_size=chunk_size
    )
    if table_update is None:
        torch.nn.utils.clip_grads_with_norm_([owned], max_norm, norm, foreach=True)
    else:
        table_update.defer(norm, max_norm)
    return norm


class RowShardedOptim(Optim):
    @dataclass(kw_only=True, slots=True)
    class Config(Optim.Config):
        fused_cpu_update: bool = False
        cpu_update_threads: int = 0
        table_path: str = "ple.table"
        prepare_zero_gradient: bool = False
        async_clip: bool = False
        streaming_update: bool = False

    def __init__(self, config, **kwargs):
        if type(config.cpu_update_threads) is not int or config.cpu_update_threads < 0:
            raise ValueError("PLE CPU update threads must be a nonnegative integer")
        if config.cpu_update_threads and not config.fused_cpu_update:
            raise ValueError("CPU update threads require fused CPU table updates")
        if config.ema is not None:
            raise ValueError("Row-sharded PLE does not yet support EMA checkpoints")
        super().__init__(config, **kwargs)
        self._pending_scheduler_step = False
        context = self.parallelism_context
        if (
            context.pp_enabled
            or context.ep_enabled
            or context.tp_enabled
            or context.cp_enabled
        ):
            raise ValueError("Row-sharded PLE currently supports pure data parallelism")
        if len(kwargs["model_parts"]) != 1:
            raise ValueError("Row-sharded PLE requires one model part")
        self.table = kwargs["model_parts"][0].get_submodule(config.table_path)
        size = config.cpu_norm_chunk_size
        transport = self.table.ownership.gradient_transport
        if transport is not None:
            transport.norm_chunk_size = size
        if size <= 0 or self.table.weight.numel() % size:
            raise ValueError("Set cpu_norm_chunk_size to divide the local PLE table")

        if config.streaming_update and not config.prepare_zero_gradient:
            raise ValueError("Streaming stores require prepared updates")
        if config.async_clip and not config.fused_cpu_update:
            raise ValueError("Asynchronous clipping requires fused CPU updates")
        if config.prepare_zero_gradient and not config.fused_cpu_update:
            raise ValueError("Prepared updates require fused CPU updates")
        self.table_update = None
        if config.fused_cpu_update:
            from .cpu_update import CPUTableUpdate
            from .overlap import CPUOverlapOptimizersContainer

            if not isinstance(self.optimizers, CPUOverlapOptimizersContainer):
                raise ValueError(
                    "Fused CPU table updates require CPU optimizer overlap"
                )
            optimizer = next(
                optimizer
                for optimizer in self.optimizers._cpu_optimizers
                if any(
                    parameter is self.table.weight
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                )
            )
            self.table_update = CPUTableUpdate(
                self.table.weight, optimizer, num_threads=config.cpu_update_threads
            )
            self.table_update.async_clip = config.async_clip
            self.table_update.streaming = config.streaming_update
            if config.async_clip:
                managed = {
                    p for group in optimizer.param_groups for p in group["params"]
                }
                if any(
                    p.device.type == "cpu" and p not in managed for p in self.parameters
                ):
                    raise ValueError(
                        "Asynchronous clipping requires one CPU optimizer owning all CPU parameters"
                    )
            optimizer._ple_table_update = self.table_update
            cache = getattr(self.table, "gpu_cache", None)
            if cache is not None:
                self.table_update.compute_cache = cache
            if config.prepare_zero_gradient:
                self.table.ownership.prepared_update = self.table_update

    def step(self, loss, *, current_step):
        if not getattr(self.optimizers, "defer_cpu_join", False):
            return super().step(loss, current_step=current_step)
        self.finish_step()
        norm = self._run_update(loss)
        # CPU AdamW still reads its parameter-group learning rate. Advance the
        # scheduler only after that worker completes, at the trainer boundary.
        self._pending_scheduler_step = True
        return norm

    def finish_step(self):
        advance_scheduler = self._pending_scheduler_step
        self._pending_scheduler_step = False
        if hasattr(self.optimizers, "finish_cpu_step"):
            self.optimizers.finish_cpu_step()
        if advance_scheduler:
            self.lr_schedulers.step()

    def _update(self, loss):
        table_update = self.table_update
        if table_update is not None and not table_update.can_fuse(self.optimizers):
            table_update = None
        norm = clip_owned(
            self.parameters,
            self.table.weight,
            self.table.ownership,
            self.config.max_norm,
            self.config.cpu_norm_chunk_size,
            **({"table_update": table_update} if table_update is not None else {}),
        )
        finite = torch.isfinite(loss).all().to(torch.int32)
        loss_mesh = self.parallelism_context.get_optional_mesh("loss")
        if loss_mesh is not None:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=loss_mesh.get_group())
        torch._assert_async(
            finite.logical_and(torch.isfinite(norm).all()),
            "Loss or gradient norm is not finite. Stopping before the update.",
        )
        self.optimizers.step()
        return norm
