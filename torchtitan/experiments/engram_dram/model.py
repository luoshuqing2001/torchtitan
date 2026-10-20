# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep Engram master parameters and dense Adam state in CPU DRAM.

Shape legend: T = tokens, D = hidden width, H = hash heads.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torchtitan.experiments.ple_dram_shard.gradient_transport import (
    OwnedGradientTransport,
)
from torchtitan.experiments.ple_dram_shard.model import (
    OwnedPrefetch,
    _fetch_owned,
    _OwnedEmbedding,
    optimizer_checkpoint_hooks,
    shard_table,
)
from torchtitan.experiments.ple_dram_shard.replicated_gradient import (
    ReplicatedGradientTransport,
)
from torchtitan.models.glimmer_dense import ple_dram as dram
from torchtitan.models.glimmer_dense.engram import (
    _CONV_DILATION,
    ENGRAM_LAYER_INDEX,
    NGramEngram,
    _segment_ids,
    _segmented_short_conv,
)
from torchtitan.models.glimmer_dense.model import GlimmerDenseModel

from .indexing import hash_ids


def _fetch(
    weight,
    tokens,
    positions,
    ready,
    device,
    stream,
    dtype,
    ownership,
    rows,
    eos,
    fused,
    compact,
    fetch_threads,
    bounded,
    gpu_indices,
    tensor_exchange,
):
    previous = None
    if fetch_threads:
        previous = torch.ops.torchtitan_ple_cpu.set_worker_threads(fetch_threads)
    try:
        return _fetch_impl(
            weight,
            tokens,
            positions,
            ready,
            device,
            stream,
            dtype,
            ownership,
            rows,
            eos,
            fused,
            compact,
            bounded,
            gpu_indices,
            tensor_exchange,
        )
    finally:
        if previous is not None:
            torch.ops.torchtitan_ple_cpu.set_worker_threads(previous)


def _fetch_impl(
    weight,
    tokens,
    positions,
    ready,
    device,
    stream,
    dtype,
    ownership,
    rows,
    eos,
    fused,
    compact,
    bounded,
    gpu_indices,
    tensor_exchange,
):
    if ready is not None:
        ready.synchronize()
    if compact:
        from .compact import fetch

        return fetch(
            weight,
            tokens,
            positions,
            device,
            stream,
            dtype,
            ownership,
            rows,
            eos,
            fused,
            bounded,
            gpu_indices,
            tensor_exchange,
        )
    ids = hash_ids(tokens, positions, rows, eos)
    return _fetch_owned(weight, ids, None, device, stream, dtype, ownership, fused)


class _StagedParameter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter, value, transport):
        ctx.parameter, ctx.transport = parameter, transport
        return value

    @staticmethod
    def backward(ctx, gradient):
        return ctx.transport.gradient(ctx.parameter, gradient), None, None


class DRAMEngram(NGramEngram):
    @dataclass(kw_only=True, slots=True)
    class Config(NGramEngram.Config):
        fused_gather: bool = False
        compact_indices: bool = False
        prepare_routing: bool = False
        early_gradient: bool = False
        fetch_threads: int = 0
        bounded_unique: bool = False
        gpu_indices: bool = False
        tensor_exchange: bool = False
        gpu_table_cache: bool = False
        double_buffer_cache: bool = False
        single_buffer_cache: bool = False
        transient_cache_workspace: bool = False
        cache_patch_chunk_rows: int = 262144
        reuse_cache_workspace: bool = False
        gpu_projections: bool = False
        offload_value: bool = False
        checkpoint_value: bool = False
        prefetch_value: bool = False
        align_value: bool = False
        align_value_forward: bool = False
        cache_stages: bool = False
        owner_routing: bool = False
        defer_gradient: bool = False
        direct_gradient: bool = False
        gpu_norm: bool = False

    def __init__(self, config):
        super().__init__(config)
        self.transport = ReplicatedGradientTransport()
        self.stages = {}
        self.value_recomputer = None
        if config.prefetch_value:
            from .recompute import ValueRecomputer

            self.value_recomputer = ValueRecomputer(
                align=config.align_value, align_forward=config.align_value_forward
            )
        self.value_offloader = None
        if config.offload_value:
            from .activation import ValueOffloader

            self.value_offloader = ValueOffloader()

    def prefetch(self, tokens, positions, *, device=None, dtype=torch.float32):
        device = tokens.device if device is None else device
        cache = getattr(self.table, "gpu_cache", None)
        if cache is not None:
            cache.begin_lookup(training=self.training and torch.is_grad_enabled())
        if self.config.cache_stages:
            # Fused CPU AdamW does not bump parameter versions. Refresh at
            # every consumed batch; reuse only within this request.
            self.stages.clear()
            if not self.config.gpu_projections:
                for parameter in self.update.parameters():
                    self._stage(parameter, device, dtype)
            if self.table.remainder_weight is not None:
                self._stage(self.table.remainder_weight, device, dtype)
        ready = stream = None
        if tokens.device.type == "cuda":
            stream = dram._copy_stream(device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                host_tokens = torch.empty(
                    tokens.shape, dtype=torch.int64, pin_memory=True
                )
                host_positions = torch.empty(
                    positions.shape, dtype=torch.int64, pin_memory=True
                )
                host_tokens.copy_(tokens, non_blocking=True)
                host_positions.copy_(positions, non_blocking=True)
                tokens.record_stream(stream)
                positions.record_stream(stream)
                ready = stream.record_event()
            tokens, positions = host_tokens, host_positions
        elif device.type == "cuda":
            stream = dram._copy_stream(device)
        return OwnedPrefetch(
            dram._POOL.submit(
                _fetch,
                self.table.weight.detach(),
                tokens,
                positions,
                ready,
                device,
                stream,
                dtype,
                self.table.ownership,
                self.rows_per_head,
                self.eos_token_id,
                self.config.fused_gather,
                self.config.compact_indices,
                self.config.fetch_threads,
                self.config.bounded_unique,
                self.config.gpu_indices,
                self.config.tensor_exchange,
            ),
            self.table.ownership,
            device,
        )

    def _stage(self, parameter, device, dtype):
        key = (parameter._version, device, dtype)
        cached = self.stages.get(parameter) if self.config.cache_stages else None
        if cached is None or cached[0] != key:
            staged = dram.stage(parameter, device, dtype)
            if self.config.cache_stages:
                self.stages[parameter] = (key, staged)
        else:
            staged = cached[1]
        return staged

    def _parameter(self, parameter, hidden):
        staged = self._stage(parameter, hidden.device, hidden.dtype).value()
        return _StagedParameter.apply(parameter, staged, self.transport)

    def forward(self, hidden, tokens, positions):
        request = getattr(tokens, "_engram_prefetch", None)
        if request is None:
            request = self.prefetch(tokens, positions, dtype=hidden.dtype)
        else:
            del tokens._engram_prefetch
        cache = getattr(self.table, "gpu_cache", None)
        if cache is None:
            rows, inverse, unique, rank_ids = request.rows(hidden.dtype)
            embeddings = _OwnedEmbedding.apply(
                self.table.weight, rows, inverse, unique, rank_ids, self.table.ownership
            ).reshape(tokens.numel(), -1)
        else:
            from .cache import CachedEmbedding

            ids = hash_ids(tokens, positions, self.rows_per_head, self.eos_token_id)
            embeddings = CachedEmbedding.apply(
                self.table.weight, cache.value(hidden.dtype), ids, request
            ).reshape(tokens.numel(), -1)
            if not torch.is_grad_enabled():
                # CachedEmbedding.backward normally joins routing. Evaluation
                # has no backward, so finish its collectives before returning.
                request.future.result()
        if self.table.remainder_weight is not None:
            remainder = self._parameter(self.table.remainder_weight, hidden)
            width = remainder.numel()
            embeddings = torch.cat(
                (embeddings[:, :width] + remainder, embeddings[:, width:]), -1
            )
        if self.value_offloader is not None:
            with self.value_offloader.capture(
                (tokens.numel(), self.update.value_width)
            ):
                return self._apply_update(hidden, embeddings, tokens, positions)
        return self._apply_update(hidden, embeddings, tokens, positions)

    def _apply_update(self, hidden, embeddings, tokens, positions):
        update = self.update
        if self.config.gpu_projections:
            return update(hidden, embeddings, tokens, positions)
        segments = _segment_ids(tokens, positions, fill_id=self.eos_token_id)
        keys = F.linear(embeddings, self._parameter(update.key_weight, hidden))
        up = self._parameter(update.value_up, hidden)
        down = self._parameter(update.value_down, hidden)
        if self.value_recomputer is not None and torch.is_grad_enabled():
            values = self.value_recomputer.project(embeddings, up, down)
        elif self.config.checkpoint_value and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            values = checkpoint(
                self._value_projection,
                embeddings,
                up,
                down,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            values = self._value_projection(embeddings, up, down)
        values = _segmented_short_conv(
            values,
            self._parameter(update.conv_weight, hidden),
            segments,
            dilation=_CONV_DILATION,
        )
        scale_weight = self._parameter(update.value_scale, hidden)
        width = scale_weight.numel()
        scaled = values[:, :width] * scale_weight
        values = torch.cat((scaled, values[:, width:]), -1)
        gate = torch.sigmoid(
            (hidden * keys).sum(-1, keepdim=True) * hidden.shape[-1] ** -0.5 - 4
        )
        return hidden + gate * values

    @staticmethod
    def _value_projection(embeddings, up, down):
        return F.linear(F.linear(embeddings, up), down)


class DRAMEngramModel(GlimmerDenseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(GlimmerDenseModel.Config):
        def set_sharding_(self, parallelism):
            super(DRAMEngramModel.Config, self).set_sharding_(parallelism)
            self.engram.table.sharding_config = None
            if not self.engram.gpu_projections:
                self.engram.update.sharding_config = None

    def __init__(self, config):
        super().__init__(config)
        if self.engram.value_offloader is not None:
            layer = self.layers[str(min(5, len(self.layers) - 1))]
            layer.register_forward_hook(
                self.engram.value_offloader.decoder_forward_hook
            )
        if self.engram.value_recomputer is not None:
            # Start after upper-layer activations have died, leaving several
            # decoder backward blocks to overlap the value GEMM.
            layer = self.layers[str(min(ENGRAM_LAYER_INDEX + 2, len(self.layers) - 1))]
            layer.register_forward_hook(
                self.engram.value_recomputer.decoder_forward_hook
            )
        offloaded = self.engram.table if config.engram.gpu_projections else self.engram
        for parameter in offloaded.parameters():
            parameter._torchtitan_skip_fsdp = True

    def init_states(self, *, buffer_device=None):
        super().init_states(buffer_device=buffer_device)
        offloaded = (
            self.engram.table if self.engram.config.gpu_projections else self.engram
        )
        for parameter in offloaded.parameters():
            cpu = parameter.detach().to("cpu", copy=True).contiguous()
            parameter.data = cpu.pin_memory() if torch.cuda.is_available() else cpu
        ownership = shard_table(self.engram.table, dist.new_group(backend="gloo"))
        ownership.owner_routing = self.engram.config.owner_routing
        ownership.tensor_exchange = self.engram.config.tensor_exchange
        ownership.prepare_routing = self.engram.config.prepare_routing
        ownership.early_gradient = self.engram.config.early_gradient
        if self.engram.config.gpu_table_cache:
            from .cache import TableCache

            self.engram.table.gpu_cache = TableCache(
                self.engram.table,
                self.tok_embeddings.weight.device,
                double_buffer=self.engram.config.double_buffer_cache,
                reuse_workspace=self.engram.config.reuse_cache_workspace,
                single_buffer=self.engram.config.single_buffer_cache,
                patch_chunk_rows=self.engram.config.cache_patch_chunk_rows,
                transient_workspace=self.engram.config.transient_cache_workspace,
            )
            ownership.compute_cache = self.engram.table.gpu_cache
            ownership.metadata_only = True
        if self.engram.config.defer_gradient:
            ownership.gradient_transport = OwnedGradientTransport(
                self.engram.table.weight,
                ownership.offset,
                direct_copy=self.engram.config.direct_gradient,
                gpu_norm=self.engram.config.gpu_norm,
            )

    def forward(
        self, tokens, positions=None, attention_masks=None, *, padding_mask=None
    ):
        if positions is not None and not hasattr(tokens, "_engram_prefetch"):
            tokens._engram_prefetch = self.engram.prefetch(
                tokens, positions, dtype=self.tok_embeddings.weight.dtype
            )
        return super().forward(
            tokens, positions, attention_masks, padding_mask=padding_mask
        )

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallelism_context):
        super()._register_optimizer_hooks(optimizers, model_parts, parallelism_context)
        for model in model_parts:
            for optimizer in optimizers:
                optimizer_checkpoint_hooks(
                    optimizer, model.engram.table.weight, model.engram.table.ownership
                )
