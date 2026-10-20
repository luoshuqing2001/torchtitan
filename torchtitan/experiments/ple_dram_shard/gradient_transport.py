# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Complete owned CPU gradients at the end of the same autograd task.

Shape legend: R = owned rows, U = requested owned rows, D = layer width.
"""

import torch

from torchtitan.models.glimmer_dense import ple_dram as dram


def _finish(allocation, values_UD, indices_U, copied, device, offset, norm_chunk_size):
    with torch.cuda.device(device):
        copied.synchronize()
    full_RD = allocation.result()
    full_RD.index_add_(0, indices_U - offset, values_UD)
    full_RD._ple_owned_rows = (full_RD._version, indices_U - offset)
    if norm_chunk_size:
        norms = torch.linalg.vector_norm(
            full_RD.view(-1, norm_chunk_size), ord=2, dim=1
        )
        full_RD._ple_owned_norm_cache = (norm_chunk_size, full_RD._version, norms)
    return full_RD


def _finish_direct(
    full_RD, copied, device, indices_U, norm_chunk_size, norm_summary=None
):
    with torch.cuda.device(device):
        copied.synchronize()
        if norm_summary is not None:
            norm_summary.ready.synchronize()
    torch.autograd.graph.increment_version(full_RD)
    full_RD._ple_owned_rows = (full_RD._version, indices_U)
    if norm_chunk_size:
        norms = (
            norm_summary.values
            if norm_summary is not None
            else torch.linalg.vector_norm(
                full_RD.view(-1, norm_chunk_size), ord=2, dim=1
            )
        )
        full_RD._ple_owned_norm_cache = (norm_chunk_size, full_RD._version, norms)
    return full_RD


class OwnedGradientTransport:
    """Keep backward()'s completion boundary and accumulate on its callback."""

    def __init__(self, parameter, offset, *, direct_copy=False, gpu_norm=False):
        if gpu_norm and not direct_copy:
            raise ValueError("GPU chunk norms require direct gradient copies")
        self.parameter = parameter
        self.offset = offset
        self.pending = []
        self.graph_id = None
        self.streams = {}
        self.norm_chunk_size = 0
        self.direct_copy = direct_copy
        self.gpu_norm = gpu_norm
        self._norm_computers = {}

    def can_defer(self):
        identifier = torch._C._current_graph_task_id()
        if self.pending and self.graph_id != identifier:
            # A failed graph does not execute final callbacks. These futures
            # produce data only; discard them rather than apply stale gradients.
            for future in self.pending:
                future.result()
            self.pending.clear()
            self.graph_id = None
        return (
            identifier >= 0
            and not torch.is_grad_enabled()
            and torch.autograd._is_checkpoint_valid()
            and not self.parameter._backward_hooks
            and not self.parameter._post_accumulate_grad_hooks
        )

    def schedule(self, allocation, merged_UD, indices_U):
        identifier = torch._C._current_graph_task_id()
        if self.graph_id != identifier:
            self.graph_id = identifier
            torch.autograd.Variable._execution_engine.queue_callback(
                lambda: self._complete(identifier)
            )
        device = merged_UD.device
        stream = self.streams.get(device.index)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self.streams[device.index] = stream
        stream.wait_stream(torch.cuda.current_stream(device))
        norm_chunk_size = (
            self.norm_chunk_size
            if self.parameter.grad is None and not self.pending
            else 0
        )
        if self.direct_copy:
            from .direct_copy import copy_rows

            full_RD = allocation.result()
            indices_U = indices_U - self.offset
            norm_summary = None
            if self.gpu_norm and norm_chunk_size:
                from .norm import DeviceChunkNorm

                key = (device.index, norm_chunk_size)
                computer = self._norm_computers.get(key)
                if computer is None:
                    computer = self._norm_computers[key] = DeviceChunkNorm(
                        device, norm_chunk_size, dram._POOL
                    )
                norm_summary = computer.compute(
                    merged_UD, indices_U, self.parameter.shape[0]
                )
            with torch.cuda.stream(stream):
                merged_UD.record_stream(stream)
                copy_rows(full_RD, merged_UD, indices_U, stream)
                copied = stream.record_event()
            self.pending.append(
                dram._POOL.submit(
                    _finish_direct,
                    full_RD,
                    copied,
                    device,
                    indices_U,
                    norm_chunk_size,
                    norm_summary,
                )
            )
            return
        values_UD = torch.empty(
            merged_UD.shape, dtype=torch.float32, device="cpu", pin_memory=True
        )
        with torch.cuda.stream(stream):
            values_UD.copy_(merged_UD, non_blocking=True)
            merged_UD.record_stream(stream)
            copied = stream.record_event()
        self.pending.append(
            dram._POOL.submit(
                _finish,
                allocation,
                values_UD,
                indices_U,
                copied,
                device,
                self.offset,
                norm_chunk_size,
            )
        )

    def _complete(self, identifier):
        if self.graph_id != identifier:
            return
        futures = self.pending
        self.pending = []
        self.graph_id = None
        for future in futures:
            value_RD = future.result()
            if self.parameter.grad is None:
                self.parameter.grad = value_RD
            else:
                self.parameter.grad.add_(value_RD)
