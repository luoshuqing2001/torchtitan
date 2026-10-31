# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grain-backed TorchTitan dataloader."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import grain.python as grain
import torch
from grain import experimental as grain_experimental
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.data.collators import Collator, TextCollator
from torchtitan.components.data.dataset import DatasetConfig
from torchtitan.components.data.types import (
    DatasetBuildContext,
    DatasetIterationPolicy,
    TokenizedTrainingMicrobatch,
    TrainingMicrobatch,
)
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import Configurable


# NOTE: This class deliberately inherits from `Exception` and not `StopIteration`.
# According to PEP 479, raising a `StopIteration` or its subclass from within a
# generator will wrap it in a `RuntimeError`. Since this exception is designed
# to be raised from a generator-based dataloader and caught by the training loop,
# inheriting from `StopIteration` would make it uncatchable and would crash the
# program.
# See: https://peps.python.org/pep-0479/
class DataloaderExhaustedError(Exception):
    """An exception that indicates dataloader exhaustion."""

    pass


class BaseDataLoader(Stateful, ABC, Configurable):
    """Enforces the `Stateful`, `state_dict()`, and `load_state_dict()` contract."""

    max_num_documents: int | None = None

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        max_num_documents: int | None = None
        """Maximum non-padding document segments in one local token microbatch."""

        def __post_init__(self) -> None:
            if self.max_num_documents is not None and self.max_num_documents <= 0:
                raise ValueError("max_num_documents must be positive")

    @abstractmethod
    def __iter__(self) -> Iterator[TrainingMicrobatch]:
        ...

    def close(self) -> None:
        pass


class GrainDataLoader(BaseDataLoader):
    """Batches and checkpoints one composed Grain dataset graph."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: DatasetConfig
        collator: Collator.Config = field(default_factory=TextCollator.Config)
        seed: int = 42
        shuffle: bool = True
        repeat: bool = True
        streaming_shuffle_buffer_size: int = 1_000
        """Streaming rows retained per rank for approximate shuffling."""
        read_options: grain.ReadOptions = field(default_factory=grain.ReadOptions)
        """Concurrent indexed reads used when a `MapDataset` becomes an `IterDataset`."""
        num_prefetch_microbatches: int = 2
        """Collated microbatches queued per rank for trainer consumption."""
        num_data_shards: int | None = None
        """Shard the data as if there were this many data-parallel ranks.

        Each rank reads ``num_data_shards / dp_world_size`` virtual shards
        (ranks ``dp_rank + j * dp_world_size``) and concatenates their
        microbatches along the token dimension. Packing starts every row at
        position 0, so the concatenation adds a document boundary and the step
        sees exactly the tokens a ``num_data_shards``-rank run sees. The
        checkpoint state uses that run's format.
        """

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        max_context_length: int,
        num_tokens_per_microbatch: int,
        **kwargs: Any,
    ) -> None:
        del kwargs
        # Validate the run policy.
        # TODO(data-finite-dp): Support finite distributed datasets with a global
        # remainder policy. Simple map datasets can truncate or pad before DP
        # sharding; filtered, mixed, packed, and streaming datasets need coordinated
        # exhaustion so every rank runs the same number of steps.
        if dp_world_size > 1 and not config.repeat:
            raise ValueError(
                "repeat=False with data parallelism can exhaust ranks at different "
                "steps and hang collectives; use repeat=True with a trainer-"
                "controlled step count"
            )
        self._dp_world_size = dp_world_size
        self._rank_id = f"dp_rank_{dp_rank}"
        self.max_num_documents = config.max_num_documents
        self._children: list[GrainDataLoader] = []
        shards = config.num_data_shards
        if shards is not None and shards != dp_world_size:
            if shards % dp_world_size != 0:
                raise ValueError(
                    f"num_data_shards={shards} must be a multiple of "
                    f"dp_world_size={dp_world_size}"
                )
            per_rank = shards // dp_world_size
            if num_tokens_per_microbatch % per_rank != 0:
                raise ValueError(
                    f"num_tokens_per_microbatch={num_tokens_per_microbatch} must "
                    f"split evenly over {per_rank} virtual shards"
                )
            if config.max_num_documents is not None:
                raise ValueError("num_data_shards does not support max_num_documents")
            child_config = replace(config, num_data_shards=None)
            self._dp_world_size = shards
            self._children = [
                GrainDataLoader(
                    child_config,
                    dp_world_size=shards,
                    dp_rank=dp_rank + j * dp_world_size,
                    tokenizer=tokenizer,
                    max_context_length=max_context_length,
                    num_tokens_per_microbatch=num_tokens_per_microbatch // per_rank,
                )
                for j in range(per_rank)
            ]
            self._iterator = self._concat_children()
            return

        # Build the dataset graph and collator.
        read_options = config.read_options
        context = DatasetBuildContext(
            tokenizer=tokenizer,
            max_context_length=max_context_length,
            num_tokens_per_microbatch=num_tokens_per_microbatch,
            read_options=read_options,
            max_num_documents=config.max_num_documents,
        )
        dataset_iteration_policy = DatasetIterationPolicy(
            seed=config.seed,
            shuffle=config.shuffle,
            repeat=config.repeat,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            streaming_shuffle_buffer_size=config.streaming_shuffle_buffer_size,
        )

        dataset = config.dataset.build(
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )
        collator = config.collator.build(context=context)

        # TODO(data-multiprocessing): CPU-heavy processing should use multiple
        # processes rather than only threads. Grain can divide map-style data among
        # workers, but packing and mixing map data with a stream produce an iterable
        # before the loader sees it. Investigate an earlier boundary where one
        # shared worker pool processes samples, instead of creating a pool per
        # dataset or packing per worker.
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=read_options)

        # Group and collate samples into training microbatches. ``batch`` and
        # ``batch_fn`` are names from Grain's API.
        dataset = dataset.batch(
            collator.num_rows_per_microbatch(),
            drop_remainder=config.repeat,
            batch_fn=collator,
        )

        # Queue completed microbatches while the trainer consumes the previous one.
        dataset = grain_experimental.ThreadPrefetchIterDataset(
            dataset, prefetch_buffer_size=config.num_prefetch_microbatches
        )
        self._iterator = iter(dataset)

    def _concat_children(self) -> Iterator[TrainingMicrobatch]:
        iterators = [iter(child) for child in self._children]
        while True:
            parts = [next(iterator) for iterator in iterators]
            for part in parts:
                if not isinstance(part, TokenizedTrainingMicrobatch) or part.model_kwargs:
                    raise TypeError("num_data_shards supports plain token microbatches")
            yield TokenizedTrainingMicrobatch(
                input=_pinned_cat([part.input for part in parts]),
                labels=_pinned_cat([part.labels for part in parts]),
                positions=_pinned_cat([part.positions for part in parts]),
                padding_mask=_pinned_cat([part.padding_mask for part in parts]),
                num_valid_tokens=sum(part.num_valid_tokens for part in parts),
            )

    def __iter__(self) -> Iterator[TrainingMicrobatch]:
        return self._iterator

    def state_dict(self) -> dict[str, Any]:
        if self._children:
            state: dict[str, Any] = {"version": 1, "dp_world_size": self._dp_world_size}
            for child in self._children:
                state[child._rank_id] = child._iterator.get_state()
            return state
        return {
            "version": 1,
            "dp_world_size": self._dp_world_size,
            self._rank_id: self._iterator.get_state(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if state_dict["version"] != 1:
            raise ValueError(
                f"unsupported GrainDataLoader state version {state_dict['version']}"
            )
        if state_dict["dp_world_size"] != self._dp_world_size:
            raise ValueError(
                "cannot resume after changing the effective data-parallel degree"
            )
        if self._children:
            for child in self._children:
                child.load_state_dict(state_dict)
            return
        if self._rank_id not in state_dict:
            raise ValueError(
                f"checkpoint is missing dataloader state for {self._rank_id}"
            )
        try:
            self._iterator.set_state(state_dict[self._rank_id])
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._children:
            for child in self._children:
                child.close()
            return
        self._iterator.close()


def _pinned_cat(tensors: list[torch.Tensor]) -> torch.Tensor:
    pin = torch.cuda.is_available() and all(t.is_pinned() for t in tensors)
    out = torch.empty(
        sum(t.shape[0] for t in tensors),
        dtype=tensors[0].dtype,
        pin_memory=pin,
    )
    return torch.cat(tensors, out=out)
