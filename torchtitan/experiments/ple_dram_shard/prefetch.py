# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fetch the consumed current batch while CPU vocabulary statistics run."""

from dataclasses import dataclass

import torch

from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.models.glimmer_dense import ple_dram as dram
from torchtitan.trainer import Trainer

from .model import OwnedPrefetch, RowShardedGlimmer, RowShardedPLE, _fetch_owned


class _PrefetchBatch(TrainingMicrobatch):
    def __init__(
        self, original, *, release_cache=False, request_attribute="_ple_prefetch"
    ):
        self.original = original
        self.release_cache = release_cache
        self.request_attribute = request_attribute
        self.request = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def as_input_dict(self):
        return self.original.as_input_dict()

    def loss_kwargs(self):
        return self.original.loss_kwargs()

    def to_input_dict(self, device, *, non_blocking=False):
        result = self.original.to_input_dict(device, non_blocking=non_blocking)
        if self.request is not None:
            setattr(result["input"], self.request_attribute, self.request)
            if self.release_cache:
                self.request = None
        return result


class EarlyFetchPLE(RowShardedPLE):
    @dataclass(kw_only=True, slots=True)
    class Config(RowShardedPLE.Config):
        pass

    def prefetch_rows(self, tokens, *, dtype=None):
        request = getattr(tokens, "_ple_prefetch", None)
        if request is None:
            return super().prefetch_rows(tokens, dtype=dtype)
        del tokens._ple_prefetch
        return request


class EarlyFetchGlimmer(RowShardedGlimmer):
    @dataclass(kw_only=True, slots=True)
    class Config(RowShardedGlimmer.Config):
        pass

    def preprocess_inputs(self, input_dict, **kwargs):
        request = getattr(input_dict["input"], "_ple_prefetch", None)
        inputs, labels, model_kwargs = super().preprocess_inputs(input_dict, **kwargs)
        if request is not None:
            inputs._ple_prefetch = request
        return inputs, labels, model_kwargs


class EarlyFetchTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        release_fetch_cache: bool = False

    def train_step(self, data_iterator):
        staging = self.engine.model_parts[0].ple.saved_input_staging
        if staging is not None and self.gradient_accumulation_steps != 1:
            raise ValueError("Saved PLE input staging requires one microbatch per step")
        # Wrap only batches consumed by this step; never advance the data cursor.
        try:
            return super().train_step(
                _PrefetchBatch(batch, release_cache=self.config.release_fetch_cache)
                for batch in data_iterator
            )
        finally:
            self.engine.optim.finish_step()
            if staging is not None:
                staging.finish_step()
            preparation = getattr(
                self.engine.model_parts[0].ple.table.ownership, "prepared_update", None
            )
            if preparation is not None:
                dram._POOL.submit(lambda: None).result()
                preparation.join_preparations()

    def close(self):
        if hasattr(self, "engine") and self.engine.model_parts:
            self.engine.optim.finish_step()
            staging = self.engine.model_parts[0].ple.saved_input_staging
            if staging is not None:
                staging.finish_step(wait=True)
            preparation = getattr(
                self.engine.model_parts[0].ple.table.ownership, "prepared_update", None
            )
            if preparation is not None:
                dram._POOL.submit(lambda: None).result()
                preparation.join_preparations()
        super().close()

    def _record_trained_vocab(self, microbatch):
        root = self.engine.model_parts[0]
        if isinstance(microbatch, _PrefetchBatch) and isinstance(
            root, EarlyFetchGlimmer
        ):
            tokens = microbatch.as_input_dict()["input"]
            device = self.engine.device
            owner = root.ple.table.ownership
            microbatch.request = OwnedPrefetch(
                dram._POOL.submit(
                    _fetch_owned,
                    root.ple.table.weight.detach(),
                    tokens.detach().to(torch.int64),
                    None,
                    device,
                    dram._copy_stream(device),
                    getattr(torch, self.engine.config.training.mixed_precision_param),
                    owner,
                    root.ple.config.fused_row_gather,
                ),
                owner,
                device,
            )
        return super()._record_trained_vocab(microbatch)
