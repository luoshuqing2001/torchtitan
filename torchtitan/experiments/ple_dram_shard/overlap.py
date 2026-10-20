# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overlap independent CPU AdamW and device optimizer updates within a step."""

import threading
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from torchtitan.components.optim import OptimizersContainer
from torchtitan.models.glimmer_dense import ple_dram


def _step_cpu_optimizers(optimizers, ready, device, communication_ready=None):
    context = torch.cuda.device(device) if device is not None else nullcontext()
    try:
        with context:
            if ready is not None:
                # Include clipping and finite checks, but not subsequent GPU updates.
                ready.synchronize()
            for optimizer in optimizers:
                table_update = getattr(optimizer, "_ple_table_update", None)
                if table_update is None:
                    optimizer.step()
                else:
                    table_update.step()
    finally:
        if communication_ready is not None:
            communication_ready.set()


class CPUOverlapOptimizersContainer(OptimizersContainer):
    """Join CPU updates at step return, or at the opt-in trainer boundary."""

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        defer_cpu_join: bool = False

    def __init__(self, config, *, model_parts, enable_cuda_graph=False):
        if enable_cuda_graph:
            raise ValueError("CPU optimizer overlap does not support CUDA graphs")
        super().__init__(config, model_parts=model_parts)
        self._cpu_optimizers = []
        self._device_optimizers = []
        self._device = None
        self._pending_cpu_step = None
        self.defer_cpu_join = config.defer_cpu_join
        for optimizer in self.optimizers:
            parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            devices = {parameter.device for parameter in parameters}
            if len(devices) != 1:
                raise ValueError(
                    "CPU overlap requires separate CPU and device optimizers; "
                    "set PLE_CPU_FUSED_ADAM=1"
                )
            device = next(iter(devices))
            if device.type == "cpu":
                if not isinstance(optimizer, torch.optim.AdamW):
                    raise ValueError("CPU overlap supports native CPU AdamW only")
                self._cpu_optimizers.append(optimizer)
            elif device.type == "cuda":
                if self._device is not None and self._device != device:
                    raise ValueError("CPU overlap supports one CUDA device per rank")
                self._device = device
                self._device_optimizers.append(optimizer)
            else:
                raise ValueError("CPU overlap supports CPU and CUDA parameters only")
        if not self._cpu_optimizers:
            raise ValueError("CPU overlap requires a separate CPU AdamW optimizer")

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("CPU optimizer overlap does not support closures")
        self.finish_cpu_step()
        ready = (
            torch.cuda.current_stream(self._device).record_event()
            if self._device is not None
            else None
        )
        # The fetch pool is idle after backward. Reuse it without allowing an
        # update to escape this step or race the next fetch/zero_grad/checkpoint.
        table_updates = [
            getattr(opt, "_ple_table_update", None) for opt in self._cpu_optimizers
        ]
        table_updates = [update for update in table_updates if update is not None]
        cached_updates = [
            update for update in table_updates if update.compute_cache is not None
        ]
        device_update = (threading.Event(), [None]) if cached_updates else None
        for update in cached_updates:
            update.device_update_ready = device_update
        communication_ready = threading.Event() if self.defer_cpu_join else None
        if communication_ready is not None:
            if len(table_updates) != 1:
                raise ValueError(
                    "Deferred CPU join requires exactly one fused table updater"
                )
            update = table_updates[0]
            if update.pending is None:
                communication_ready = None
            update.communication_ready = communication_ready
            if communication_ready is not None and update.compute_cache is None:
                # This worker will not enqueue any distributed device work.
                communication_ready.set()
        future = ple_dram._POOL.submit(
            _step_cpu_optimizers,
            self._cpu_optimizers,
            ready,
            self._device,
            communication_ready,
        )
        self._pending_cpu_step = future
        try:
            for optimizer in self._device_optimizers:
                optimizer.step()
        except BaseException:
            if device_update is not None:
                device_update[0].set()
            self.finish_cpu_step()
            raise
        if device_update is not None:
            if self._device is not None:
                device_update[1][0] = torch.cuda.current_stream(
                    self._device
                ).record_event()
            device_update[0].set()
        if communication_ready is None:
            self.finish_cpu_step()
        else:
            # All cache collectives must be enqueued before main-thread metric
            # collectives. Only the remaining CPU-only tail may overlap them.
            communication_ready.wait()
            if future.done():
                self.finish_cpu_step()
        if self._device is not None:
            for update in cached_updates:
                torch.cuda.current_stream(self._device).wait_event(
                    update.compute_cache.ready
                )

    def finish_cpu_step(self):
        future, self._pending_cpu_step = self._pending_cpu_step, None
        if future is not None:
            future.result()
