# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replicated compute-dtype lookup cache; master and Adam state stay in DRAM."""

import threading

import torch
import torch.distributed as dist

from torchtitan.experiments.ple_dram_shard.model import owned_gradient
from torchtitan.models.glimmer_dense import ple_dram as dram


class TableCache:
    def __init__(
        self,
        table,
        device,
        *,
        double_buffer=False,
        reuse_workspace=False,
        single_buffer=False,
        patch_chunk_rows=262144,
        transient_workspace=False,
    ):
        self.double_buffer = double_buffer
        self.single_buffer = single_buffer
        self.patch_chunk_rows = patch_chunk_rows
        self.transient_workspace = transient_workspace
        self.consumer_stream = None
        self.packed_ids_host = None
        self.lookup_gate = None
        self.lookup_version = None
        self.preload_enabled = True
        self.ordered_collectives = False
        self.forward_window = None
        self.lookup_request = None
        self.loss_chunks = self.loss_chunk = 0
        self.patch_workspace = None
        self.reuse_workspace = reuse_workspace
        self.next_full = self.next_ready = self.next_ids = None
        self.packed_host = None
        self.candidate_layout = self.patch_layout = None
        self.table = table
        self.device = device
        self.dtype = None
        self.group = dist.new_group(backend="nccl")
        self.stream = torch.cuda.Stream(device=device)
        self.full = self.local = self.host = None
        self.ready = None
        self.version = None
        self.host_version = None

    def _allocate(self, dtype):
        self.dtype = dtype
        self.full = torch.empty(
            self.table.ownership.vocab_size,
            self.table.ownership.width,
            device=self.device,
            dtype=dtype,
        )
        self.local = (
            None
            if self.reuse_workspace
            else torch.empty(self.table.weight.shape, device=self.device, dtype=dtype)
        )
        self.host = torch.empty(self.table.weight.shape, dtype=dtype, pin_memory=True)
        self.host_version = None
        if self.double_buffer:
            self.next_full = (
                self.full if self.single_buffer else torch.empty_like(self.full)
            )
            self.packed_host = torch.empty_like(self.host, pin_memory=True)
        if self.single_buffer and not self.transient_workspace:
            world_size = dist.get_world_size(self.table.ownership.group)
            self.patch_workspace = torch.empty(
                world_size,
                self.patch_chunk_rows,
                self.table.ownership.width,
                dtype=dtype,
                device=self.device,
            )
        self.ready = None

    def begin_lookup(self, *, training):
        self.preload_enabled = training
        if getattr(self, "ordered_collectives", False):
            self.forward_window = (threading.Event(), [None])
            self.lookup_request = None
            self.loss_chunk = 0
        if not self.single_buffer:
            return
        version = self.table.weight._version
        if training and self.lookup_version == version:
            raise ValueError(
                "Single-buffer cache requires one training lookup per optimizer step"
            )
        if training:
            self.lookup_version = version
        self.lookup_gate = (threading.Event(), [None])

    def finish_lookup(self, request=None):
        self.lookup_request = request
        if self.single_buffer:
            gate, result = self.lookup_gate
            result[0] = torch.cuda.current_stream(self.device).record_event()
            gate.set()

    def cancel_lookup(self):
        if (
            getattr(self, "ordered_collectives", False)
            and self.forward_window is not None
        ):
            self.forward_window[0].set()
            self.lookup_request = None
        if self.single_buffer and self.lookup_gate is not None:
            self.lookup_gate[0].set()
            self.lookup_version = None

    def open_forward_window(self):
        if self.preload_enabled and self.lookup_request is not None:
            gate, event = self.forward_window
            event[0] = torch.cuda.current_stream(self.device).record_event()
            gate.set()

    def before_loss_chunk(self):
        if not self.preload_enabled or self.lookup_request is None:
            return
        self.loss_chunk += 1
        if self.loss_chunk != self.loss_chunks:
            return
        # Earlier loss chunks do not communicate gradients. The last one does;
        # finish cache communication before its FSDP/DP reduce-scatter can start.
        self.lookup_request.future.result()
        preparation = self.table.ownership.prepared_update
        for future in preparation.preparations:
            result = future.result()
            if result[5] is not None:
                torch.cuda.current_stream(self.device).wait_event(self.next_ready)

    def wait_host(self):
        for event in (self.ready, self.next_ready):
            if event is not None:
                event.synchronize()

    def _local_view(self, target):
        if not self.reuse_workspace:
            return self.local
        rank = dist.get_rank(self.table.ownership.group)
        count = self.table.weight.shape[0]
        return target.narrow(0, rank * count, count)

    def preload(self, layout):
        window = None
        if self.ordered_collectives:
            gate, window = self.forward_window
            gate.wait()
            if window[0] is None:
                return False
        if self.single_buffer:
            gate, result = self.lookup_gate
            gate.wait()
            if result[0] is None:
                return False
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if window is not None:
                self.stream.wait_event(window[0])
            if self.single_buffer:
                self.stream.wait_event(result[0])
                # The live buffer now becomes scratch for the next version.
                # A failed step must refresh it from authoritative host state.
                self.version = None
            local = self._local_view(self.next_full)
            local.copy_(self.host, non_blocking=True)
            dist.all_gather_into_tensor(self.next_full, local, group=self.group)
            if self.transient_workspace:
                self.packed_ids_host = layout[0].pin_memory()
            else:
                self.next_ids = (
                    layout[0].pin_memory().to(self.device, non_blocking=True)
                )
            self.next_ready = self.stream.record_event()
        self.candidate_layout = layout
        return True

    def _patch_chunks(self, counts):
        from .kernels import scatter_rows

        rank = dist.get_rank(self.table.ownership.group)
        world_size = len(counts)
        chunk = self.patch_chunk_rows
        if self.transient_workspace:
            with torch.cuda.stream(self.consumer_stream):
                workspace = torch.empty(
                    world_size,
                    chunk,
                    self.table.ownership.width,
                    device=self.device,
                    dtype=self.dtype,
                )
            self.stream.wait_stream(self.consumer_stream)
            workspace.record_stream(self.stream)
            ids = self.packed_ids_host.to(self.device, non_blocking=True)
        else:
            workspace = self.patch_workspace
            ids = self.next_ids
        local = workspace[rank]
        starts = [sum(counts[:owner]) for owner in range(world_size)]
        for offset in range(0, max(counts), chunk):
            count = max(0, min(chunk, counts[rank] - offset))
            local[:count].copy_(
                self.packed_host[offset : offset + count], non_blocking=True
            )
            dist.all_gather_into_tensor(
                workspace.flatten(0, 1), local, group=self.group
            )
            for owner, total in enumerate(counts):
                count = max(0, min(chunk, total - offset))
                start = starts[owner] + offset
                scatter_rows(
                    self.full,
                    ids[start : start + count],
                    workspace[owner, :count],
                )

    def refresh_after_step(self):
        patch = False
        if self.double_buffer:
            eligible = torch.tensor(
                int(self.patch_layout is not None), dtype=torch.int32
            )
            dist.all_reduce(
                eligible, op=dist.ReduceOp.MIN, group=self.table.ownership.group
            )
            patch = bool(eligible)
        if not patch:
            self.refresh()
            self.patch_layout = None
            return
        _ids, counts = self.patch_layout
        count = counts[dist.get_rank(self.table.ownership.group)]
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if self.single_buffer:
                self._patch_chunks(counts)
                self.ready = self.stream.record_event()
                self.version = self.table.weight._version
                self.patch_layout = None
                return
            if self.reuse_workspace:
                # The consumed lookup has no backward readers. Reuse it for the
                # patch collective while next_full holds the next lookup state.
                values = self.full[: sum(counts)]
                rank = dist.get_rank(self.table.ownership.group)
                local = values.narrow(0, sum(counts[:rank]), count)
            else:
                local = self.local[:count]
                values = torch.empty(
                    (sum(counts), self.table.ownership.width),
                    device=self.device,
                    dtype=self.dtype,
                )
            local.copy_(self.packed_host[:count], non_blocking=True)
            dist.all_gather(list(values.split(counts)), local, group=self.group)
            from .kernels import scatter_rows

            scatter_rows(self.next_full, self.next_ids, values)
            self.full, self.next_full = self.next_full, self.full
            self.ready = self.stream.record_event()
        self.version = self.table.weight._version
        self.patch_layout = None

    @torch.no_grad()
    def refresh(self):
        if self.dtype is None:
            return
        self.wait_host()
        if self.host_version != self.table.weight._version:
            self.host.copy_(self.table.weight.detach())
            self.host_version = self.table.weight._version
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            local = self._local_view(self.full)
            local.copy_(self.host, non_blocking=True)
            dist.all_gather_into_tensor(self.full, local, group=self.group)
            self.ready = self.stream.record_event()
        self.version = self.table.weight._version

    def value(self, dtype):
        if self.dtype != dtype:
            self.wait_host()
            self._allocate(dtype)
            self.version = None
        if self.version != self.table.weight._version:
            self.refresh()
        stream = torch.cuda.current_stream(self.device)
        self.consumer_stream = stream
        stream.wait_event(self.ready)
        self.full.record_stream(stream)
        return self.full


class CachedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter, cached, ids, request):
        ctx.request = request
        result = cached.index_select(0, ids)
        cache = getattr(request.ownership, "compute_cache", None)
        if cache is not None:
            cache.finish_lookup(request)
        return result

    @staticmethod
    def backward(ctx, gradient):
        request = ctx.request
        (_, unique, inverse, _, rank_ids, _, _, mapping, event) = (
            request.future.result()
        )
        if event is not None:
            stream = torch.cuda.current_stream(gradient.device)
            stream.wait_event(event)
            for tensor in (unique, inverse, mapping):
                tensor.record_stream(stream)
        rows = dram._embedding_row_grad(
            gradient.reshape(-1, gradient.shape[-1]), inverse, unique.numel()
        )
        return (
            owned_gradient(unique, rows, rank_ids, request.ownership),
            None,
            None,
            None,
        )
