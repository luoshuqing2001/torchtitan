# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Start current-step Engram fetches before vocabulary statistics."""

from dataclasses import dataclass

import torch

from torchtitan.experiments.ple_dram_shard.prefetch import _PrefetchBatch
from torchtitan.trainer import Trainer

from .model import DRAMEngramModel


class EarlyEngramModel(DRAMEngramModel):
    @dataclass(kw_only=True, slots=True)
    class Config(DRAMEngramModel.Config):
        pass

    def preprocess_inputs(self, input_dict, **kwargs):
        request = getattr(input_dict["input"], "_engram_prefetch", None)
        inputs, labels, model_kwargs = super().preprocess_inputs(input_dict, **kwargs)
        if request is not None:
            inputs._engram_prefetch = request
        return inputs, labels, model_kwargs


class EarlyEngramTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def __init__(self, config):
        super().__init__(config)
        model = self.engine.model_parts[0]
        cache = getattr(model.engram.table, "gpu_cache", None)
        self._cache_window_hooks = []
        if cache is None or not cache.double_buffer:
            return
        if self.gradient_accumulation_steps != 1:
            raise ValueError(
                "Prepared GPU cache requires one microbatch per optimizer step"
            )
        from torchtitan.components.loss import ChunkedLossWrapper

        loss = self.engine.loss_fn
        if not isinstance(loss, ChunkedLossWrapper) or loss.num_chunks < 2:
            raise ValueError(
                "Prepared GPU cache requires chunked loss with at least two chunks"
            )
        if (
            self.engine.parallelism_context.dp_shard_enabled
            and config.parallelism.fsdp_reshard_after_forward != "never"
        ):
            raise ValueError(
                "Prepared GPU cache requires fsdp_reshard_after_forward=never"
            )
        cache.ordered_collectives = True
        cache.loss_chunks = loss.num_chunks

        def forward_boundary(module, args, output):
            if module.training and torch.is_grad_enabled():
                cache.open_forward_window()

        def loss_chunk_boundary(module, args):
            if module.training and torch.is_grad_enabled():
                cache.before_loss_chunk()

        self._cache_window_hooks = [
            model.register_forward_hook(forward_boundary),
            loss.lm_head.register_forward_pre_hook(loss_chunk_boundary),
        ]

    def close(self):
        # Final validation runs outside train_step. Drain its fetch children
        # and side-stream cache transfers before process groups are destroyed.
        from torchtitan.models.glimmer_dense.ple_dram import _POOL

        if hasattr(self, "engine") and self.engine.model_parts:
            self.engine.optim.finish_step()
            cache = getattr(self.engine.model_parts[0].engram.table, "gpu_cache", None)
            if cache is not None:
                cache.cancel_lookup()
        _POOL.submit(lambda: None).result()
        if hasattr(self, "engine") and self.engine.model_parts:
            model = self.engine.model_parts[0]
            preparation = getattr(model.engram.table.ownership, "prepared_update", None)
            if preparation is not None:
                preparation.join_preparations()
            cache = getattr(model.engram.table, "gpu_cache", None)
            if cache is not None:
                cache.wait_host()
        super().close()
        for handle in getattr(self, "_cache_window_hooks", ()):
            handle.remove()

    def train_step(self, data_iterator):
        try:
            return super().train_step(
                _PrefetchBatch(
                    batch, release_cache=True, request_attribute="_engram_prefetch"
                )
                for batch in data_iterator
            )
        finally:
            model = self.engine.model_parts[0]
            self.engine.optim.finish_step()
            cache = getattr(model.engram.table, "gpu_cache", None)
            if cache is not None:
                cache.cancel_lookup()
            preparation = getattr(model.engram.table.ownership, "prepared_update", None)
            if preparation is not None:
                # A failed step must not leave background reads of live state.
                from torchtitan.models.glimmer_dense.ple_dram import _POOL

                _POOL.submit(lambda: None).result()
                preparation.join_preparations()
            if model.engram.value_recomputer is not None:
                model.engram.value_recomputer.clear()
            offloader = model.engram.value_offloader
            if offloader is not None:
                offloader.clear()

    def _record_trained_vocab(self, microbatch):
        model = self.engine.model_parts[0]
        if isinstance(microbatch, _PrefetchBatch) and isinstance(
            model, EarlyEngramModel
        ):
            data = microbatch.as_input_dict()
            microbatch.request = model.engram.prefetch(
                data["input"],
                data["positions"],
                device=self.engine.device,
                dtype=getattr(torch, self.engine.config.training.mixed_precision_param),
            )
        return super()._record_trained_vocab(microbatch)
