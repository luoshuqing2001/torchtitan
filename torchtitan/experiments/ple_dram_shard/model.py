# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experimental CPU row ownership for PLE, limited to pure data parallelism.

Shape legend: V = vocabulary rows, D = concatenated layer width, U = unique
requested rows, R = rows owned by one rank, T = tokens. Training parameters
are local CPU tensors; checkpoint views carry global DTensor placements.
"""

from concurrent.futures import Future
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from torchtitan.models.glimmer_dense import ple_dram as dram
from torchtitan.models.glimmer_dense.model import GlimmerDenseModel
from torchtitan.models.glimmer_dense.ple import PerLayerEmbedder

from .embedding import FusedOwnedEmbedding
from .gather import gather_packed
from .gradient_transport import OwnedGradientTransport


@dataclass
class RowOwnership:
    vocab_size: int
    width: int
    group: dist.ProcessGroup
    mesh: DeviceMesh
    gradient_transport: OwnedGradientTransport | None = None
    owner_routing: bool = False
    tensor_exchange: bool = False

    @property
    def num_rows(self):
        return self.vocab_size // dist.get_world_size(self.group)

    @property
    def offset(self):
        return self.num_rows * dist.get_rank(self.group)

    def checkpoint_view(self, tensor_RD):
        return DTensor.from_local(
            tensor_RD,
            self.mesh,
            (Shard(0),),
            run_check=False,
            shape=torch.Size((self.vocab_size, self.width)),
            stride=(self.width, 1),
        )


def shard_table(table, group):
    """Release replicated storage and expose a globally indexed DCP view."""
    weight_VD = table.weight
    if weight_VD.device.type != "cpu" or weight_VD.dtype != torch.float32:
        raise ValueError("Row-sharded PLE requires FP32 CPU master weights")
    vocab_size, width = weight_VD.shape
    if vocab_size % dist.get_world_size(group):
        raise ValueError("Row-sharded PLE currently requires equal vocabulary shards")
    mesh = DeviceMesh.from_group(group, "cpu", mesh_dim_names=("ple_rows",))
    ownership = RowOwnership(vocab_size, width, group, mesh)
    local_RD = weight_VD.detach()[
        ownership.offset : ownership.offset + ownership.num_rows
    ].clone()
    if weight_VD.is_pinned():
        local_RD = local_RD.pin_memory()
    table.weight = nn.Parameter(local_RD)
    table.ownership = ownership

    def save(module, state, prefix, metadata):
        state[prefix + "weight"] = ownership.checkpoint_view(state[prefix + "weight"])

    def load(module, state, prefix, metadata, strict, missing, unexpected, errors):
        key = prefix + "weight"
        if key not in state:
            return
        value = state[key]
        if isinstance(value, DTensor):
            value = value.to_local()
        elif value.shape == (ownership.vocab_size, ownership.width):
            value = value[ownership.offset : ownership.offset + ownership.num_rows]
        state[key] = value

    table.register_state_dict_post_hook(save)
    table.register_load_state_dict_pre_hook(load)
    return ownership


def optimizer_checkpoint_hooks(optimizer, parameter, ownership):
    """Store Adam moment shards as global tensors, keeping runtime state local."""

    if not any(
        parameter is param
        for group in optimizer.param_groups
        for param in group["params"]
    ):
        return

    def save(optimizer, state):
        for live, saved in zip(
            optimizer.param_groups, state["param_groups"], strict=True
        ):
            for param, index in zip(live["params"], saved["params"], strict=True):
                if param is parameter and index in state["state"]:
                    state["state"][index] = {
                        key: ownership.checkpoint_view(value)
                        if isinstance(value, torch.Tensor)
                        and value.shape == parameter.shape
                        else value
                        for key, value in state["state"][index].items()
                    }
        return state

    def load(optimizer, state):
        state = dict(state)
        state["state"] = dict(state["state"])
        for live, saved in zip(
            optimizer.param_groups, state["param_groups"], strict=True
        ):
            for param, index in zip(live["params"], saved["params"], strict=True):
                if param is parameter and index in state["state"]:
                    local = {}
                    for key, value in state["state"][index].items():
                        if isinstance(value, DTensor):
                            value = value.to_local()
                        elif isinstance(value, torch.Tensor) and value.shape == (
                            ownership.vocab_size,
                            ownership.width,
                        ):
                            value = value[
                                ownership.offset : ownership.offset + ownership.num_rows
                            ]
                        local[key] = value
                    state["state"][index] = local
        return state

    optimizer.register_state_dict_post_hook(save)
    optimizer.register_load_state_dict_pre_hook(load)


def _fetch_owned(
    weight_RD, ids_T, ready, device, stream, dtype, ownership, fused_gather=False
):
    if ready is not None:
        ready.synchronize()
    unique_U, inverse_T = torch.unique(ids_T, return_inverse=True)
    if ownership.tensor_exchange:
        from .request_exchange import exchange_indices

        rank_ids = [
            request[0] for request in exchange_indices((unique_U,), ownership.group)
        ]
    else:
        rank_ids = [None] * dist.get_world_size(ownership.group)
        dist.all_gather_object(rank_ids, unique_U, group=ownership.group)
    union_U = torch.unique(torch.cat(rank_ids))
    owner_counts = [
        int(
            (
                (union_U >= rank * ownership.num_rows)
                & (union_U < (rank + 1) * ownership.num_rows)
            ).sum()
        )
        for rank in range(len(rank_ids))
    ]
    local_U = union_U[
        (union_U >= ownership.offset)
        & (union_U < ownership.offset + ownership.num_rows)
    ]
    if fused_gather:
        rows_UD = gather_packed(
            weight_RD,
            local_U - ownership.offset,
            dtype,
            pin_memory=device.type == "cuda",
        )
    else:
        rows_UD = weight_RD.index_select(0, local_U - ownership.offset).to(dtype)
    mapping_U = torch.searchsorted(union_U, unique_U)
    local_counts = torch.bincount(
        unique_U // ownership.num_rows, minlength=len(rank_ids)
    ).tolist()
    if device.type == "cuda":
        rows_UD = rows_UD.pin_memory()
        with torch.cuda.device(device), torch.cuda.stream(stream):
            rows_UD = rows_UD.to(device, non_blocking=True)
            inverse_T = inverse_T.pin_memory().to(device, non_blocking=True)
            unique_U = unique_U.pin_memory().to(device, non_blocking=True)
            mapping_U = mapping_U.pin_memory().to(device, non_blocking=True)
            event = stream.record_event()
    else:
        event = None
    if getattr(ownership, "prepare_routing", False):
        from .routing import PreparedRequests, prepare_routing

        rank_ids = PreparedRequests(rank_ids)
        rank_ids.routing_plan = dram._POOL.submit(
            prepare_routing, rank_ids, ownership, local_U
        )
    if getattr(ownership, "early_gradient", False):
        rank_ids.gradient_allocation = dram._POOL.submit(
            _allocate_gradient, ownership, device
        )
    preparation = getattr(ownership, "prepared_update", None)
    if preparation is not None:
        preparation.prepare(local_U - ownership.offset)
    return (
        rows_UD,
        unique_U,
        inverse_T,
        union_U,
        rank_ids,
        owner_counts,
        local_counts,
        mapping_U,
        event,
    )


@dataclass
class OwnedPrefetch:
    future: Future | None
    ownership: RowOwnership
    device: torch.device
    cached: tuple | None = None
    cached_ready: torch.cuda.Event | None = None

    def rows(self, dtype):
        if self.cached is not None:
            rows_UD, inverse_T, unique_U, rank_ids = self.cached
            if self.cached_ready is not None:
                consumer = torch.cuda.current_stream(self.device)
                consumer.wait_event(self.cached_ready)
                for tensor in (rows_UD, inverse_T, unique_U):
                    tensor.record_stream(consumer)
            return rows_UD.to(dtype=dtype), inverse_T, unique_U, rank_ids
        assert self.future is not None
        (
            owned_UD,
            unique_U,
            inverse_T,
            _,
            rank_ids,
            counts,
            local_counts,
            mapping_U,
            event,
        ) = self.future.result()
        if event is not None:
            producer = torch.cuda.current_stream(self.device)
            producer.wait_event(event)
            for tensor in (owned_UD, unique_U, inverse_T, mapping_U):
                tensor.record_stream(producer)
        values_UD = torch.empty(
            (max(counts), self.ownership.width), device=self.device, dtype=dtype
        )
        rows_UD = torch.empty(
            (unique_U.numel(), self.ownership.width), device=self.device, dtype=dtype
        )
        cursor = 0
        local_cursor = 0
        for rank, count in enumerate(counts):
            if rank == dist.get_rank(self.ownership.group):
                values_UD[:count].copy_(owned_UD)
            dist.broadcast(
                values_UD[:count] if self.ownership.owner_routing else values_UD,
                src=rank,
                group=self.ownership.group if self.device.type == "cpu" else None,
            )
            local_count = local_counts[rank]
            end = local_cursor + local_count
            positions_U = mapping_U[local_cursor:end] - cursor
            torch.index_select(values_UD, 0, positions_U, out=rows_UD[local_cursor:end])
            local_cursor = end
            cursor += count
        self.cached = (
            rows_UD,
            inverse_T,
            unique_U,
            rank_ids,
        )
        if self.device.type == "cuda":
            self.cached_ready = torch.cuda.current_stream(self.device).record_event()
        self.future = None
        return self.cached


def prefetch_owned(weight_RD, tokens_T, ownership, dtype, *, fused_gather=False):
    device = tokens_T.device
    stream = None
    ready = None
    if device.type == "cuda":
        stream = dram._copy_stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            ids_T = torch.empty(tokens_T.shape, dtype=torch.int64, pin_memory=True)
            ids_T.copy_(tokens_T, non_blocking=True)
            tokens_T.record_stream(stream)
            ready = stream.record_event()
    else:
        ids_T = tokens_T.detach().to(torch.int64)
    return OwnedPrefetch(
        dram._POOL.submit(
            _fetch_owned,
            weight_RD.detach(),
            ids_T,
            ready,
            device,
            stream,
            dtype,
            ownership,
            fused_gather,
        ),
        ownership,
        device,
    )


def _allocate_gradient(ownership, device):
    if device.type == "cuda":
        with torch.cuda.device(device):
            return torch.zeros(
                ownership.num_rows,
                ownership.width,
                dtype=torch.float32,
                pin_memory=True,
            )
    return torch.zeros(ownership.num_rows, ownership.width, dtype=torch.float32)


def owned_gradient(unique_U, gradient_UD, rank_ids, ownership):
    if (
        ownership.owner_routing
        and gradient_UD.device.type == "cuda"
        and not torch.is_grad_enabled()
    ):
        from .routing import owned_gradient as routed_gradient

        return routed_gradient(unique_U, gradient_UD, rank_ids, ownership)
    future = dram._POOL.submit(_allocate_gradient, ownership, gradient_UD.device)
    local_U = torch.unique(torch.cat(rank_ids))
    local_U = local_U[
        (local_U >= ownership.offset)
        & (local_U < ownership.offset + ownership.num_rows)
    ]
    local_cpu_U = local_U
    if gradient_UD.device.type == "cuda":
        local_U = local_U.pin_memory().to(gradient_UD.device, non_blocking=True)
    merged_UD = gradient_UD.new_zeros((local_U.numel(), ownership.width))
    max_rows = max(ids.numel() for ids in rank_ids)
    values_UD = gradient_UD.new_zeros((max_rows + 1, ownership.width))
    for rank, ids_U in enumerate(rank_ids):
        if rank == dist.get_rank(ownership.group):
            values_UD[: unique_U.numel()].copy_(gradient_UD)
        dist.broadcast(
            values_UD,
            src=rank,
            group=ownership.group if gradient_UD.device.type == "cpu" else None,
        )
        padded_U = torch.full(
            (max_rows + 1,),
            ownership.vocab_size,
            dtype=torch.int64,
            device=gradient_UD.device,
        )
        source_U = ids_U.pin_memory() if gradient_UD.device.type == "cuda" else ids_U
        padded_U[: ids_U.numel()].copy_(source_U, non_blocking=True)
        positions_U = torch.searchsorted(padded_U, local_U)
        matched_U = padded_U.index_select(0, positions_U) == local_U
        positions_U = torch.where(matched_U, positions_U, max_rows)
        merged_UD.add_(values_UD.index_select(0, positions_U))
    transport = ownership.gradient_transport
    if (
        gradient_UD.device.type == "cuda"
        and transport is not None
        and transport.can_defer()
    ):
        transport.schedule(future, merged_UD, local_cpu_U)
        return None
    values_cpu_UD = dram._cpu_grad(merged_UD, torch.float32)
    full_RD = future.result()
    del future
    full_RD.index_add_(0, local_U.cpu() - ownership.offset, values_cpu_UD)
    return full_RD


class _OwnedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight_RD, rows_UD, inverse_T, unique_U, rank_ids, ownership):
        ctx.save_for_backward(inverse_T, unique_U)
        ctx.rank_ids = rank_ids
        ctx.ownership = ownership
        return rows_UD.index_select(0, inverse_T)

    @staticmethod
    def backward(ctx, grad_TD):
        inverse_T, unique_U = ctx.saved_tensors
        gradient_UD = dram._embedding_row_grad(
            grad_TD.reshape(-1, grad_TD.shape[-1]), inverse_T, unique_U.numel()
        )
        return (
            owned_gradient(unique_U, gradient_UD, ctx.rank_ids, ctx.ownership),
            None,
            None,
            None,
            None,
            None,
        )


class RowShardedPLE(PerLayerEmbedder):
    @dataclass(kw_only=True, slots=True)
    class Config(PerLayerEmbedder.Config):
        async_training: bool = True
        fused_row_gather: bool = False
        fused_embedding: bool = False
        defer_table_gradient: bool = False
        direct_gradient_copy: bool = False
        gpu_norm: bool = False
        owner_routing: bool = False
        tensor_exchange: bool = False
        prepare_routing: bool = False
        early_gradient: bool = False
        offload_saved_inputs: bool = False
        compact_saved_inputs: bool = False
        audit_compact_inputs: bool = False

    def __init__(self, config):
        super().__init__(config)
        self.saved_input_staging = None
        if config.offload_saved_inputs:
            from .saved_inputs import SavedInputStaging

            self.saved_input_staging = SavedInputStaging(
                self.num_layers,
                compact=config.compact_saved_inputs,
                audit=config.audit_compact_inputs,
            )

    def prefetch_rows(self, tokens, *, dtype=None):
        return prefetch_owned(
            self.table.weight,
            tokens,
            self.table.ownership,
            dtype or self.table.weight.dtype,
            fused_gather=self.config.fused_row_gather,
        )

    def _combine(self, hidden_TD, row_prefetch):
        context_TD = self.context(hidden_TD)
        rows_UD, inverse_T, unique_U, rank_ids = row_prefetch.rows(hidden_TD.dtype)
        if self.saved_input_staging is not None and torch.is_grad_enabled():
            self.saved_input_staging.set_indices(inverse_T, unique_U.numel())
        if (
            self.config.fused_embedding
            and hidden_TD.device.type == "cuda"
            and hidden_TD.dtype == torch.bfloat16
        ):
            tokens_TD = FusedOwnedEmbedding.apply(
                self.table.weight,
                rows_UD,
                inverse_T,
                unique_U,
                rank_ids,
                self.table.ownership,
                context_TD,
                self.ple_dim**0.5,
            )
            return tokens_TD.unflatten(-1, (self.num_layers, self.ple_dim))
        tokens_TD = _OwnedEmbedding.apply(
            self.table.weight,
            rows_UD,
            inverse_T,
            unique_U,
            rank_ids,
            self.table.ownership,
        )
        tokens_TD.mul_(self.ple_dim**0.5)
        tokens_TD.add_(context_TD)
        tokens_TD.mul_(0.5**0.5)
        return tokens_TD.unflatten(-1, (self.num_layers, self.ple_dim))


class RowShardedGlimmer(GlimmerDenseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(GlimmerDenseModel.Config):
        ple_async_training: bool = True

    def __init__(self, config):
        super().__init__(config)
        manager = self.ple.saved_input_staging
        if manager is not None:
            for layer in self.layers.values():
                layer._ple_saved_input_staging = manager

    def init_states(self, *, buffer_device=None):
        if not self.config.ple_async_training:
            raise ValueError("Row-sharded PLE requires ple_async_training=True")
        super().init_states(buffer_device=buffer_device)
        if self.ple is None:
            raise ValueError("Row-sharded experiment requires PLE")
        group = dist.new_group(backend="gloo")
        ownership = shard_table(self.ple.table, group)
        ownership.owner_routing = self.ple.config.owner_routing
        ownership.tensor_exchange = self.ple.config.tensor_exchange
        ownership.prepare_routing = self.ple.config.prepare_routing
        ownership.early_gradient = self.ple.config.early_gradient
        if self.ple.config.defer_table_gradient:
            ownership.gradient_transport = OwnedGradientTransport(
                self.ple.table.weight,
                ownership.offset,
                direct_copy=self.ple.config.direct_gradient_copy,
                gpu_norm=self.ple.config.gpu_norm,
            )

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallelism_context):
        super()._register_optimizer_hooks(optimizers, model_parts, parallelism_context)
        for model in model_parts:
            for optimizer in optimizers:
                optimizer_checkpoint_hooks(
                    optimizer, model.ple.table.weight, model.ple.table.ownership
                )
