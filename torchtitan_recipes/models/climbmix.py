# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ClimbMix train/validation split shared by the Glimmer recipes.

The last 16 parquet shards (about 1B tokens) are held out. Training never
reads them. Validation runs about every 250M training tokens and scores
about 8M held-out tokens, which is enough for a stable loss and about 1%
of step time.
"""

import glob
import os

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.components.validate import Validator
from torchtitan.hf_datasets.text_datasets import TextProcessor

_VAL_SHARDS = 16
_VAL_EVERY_TOKENS = 250_000_000
_VAL_TOKENS = 8_388_608


def climbmix_train_and_val_files(data_dir: str) -> tuple[list[str], list[str]]:
    """Return sorted train shards and the held-out validation shards."""
    paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.parquet")))
    if len(paths) <= _VAL_SHARDS:
        raise ValueError(
            f"{data_dir} has {len(paths)} shards; need more than {_VAL_SHARDS} "
            "so validation can hold some out."
        )
    return paths[:-_VAL_SHARDS], paths[-_VAL_SHARDS:]


def climbmix_dataset(files: list[str]) -> SingleDatasetConfig:
    return SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="parquet",
            split="train",
            load_dataset_kwargs={"data_files": {"train": files}},
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    )


def climbmix_validator(
    val_files: list[str],
    *,
    global_tokens_per_step: int,
    microbatch_tokens: int | None = None,
    val_every_tokens: int = _VAL_EVERY_TOKENS,
) -> Validator.Config:
    """Validation cadence scaled to the training step's token count.

    Data-parallel degree is greater than 1, so every rank must run the same
    positive number of validation steps. A validation step scores one
    microbatch per rank, so ``microbatch_tokens`` (global tokens in one
    microbatch) sets the step count when training accumulates gradients.
    """
    if global_tokens_per_step <= 0:
        raise ValueError(
            f"global_tokens_per_step must be positive, got {global_tokens_per_step}"
        )
    freq_env = os.environ.get("VAL_FREQ")
    freq = (
        int(freq_env)
        if freq_env is not None
        else max(1, round(val_every_tokens / global_tokens_per_step))
    )
    if freq <= 0:
        raise ValueError(f"VAL_FREQ must be positive, got {freq}")
    return Validator.Config(
        freq=freq,
        validate_on_step_one=False,
        steps=max(
            1, round(_VAL_TOKENS / (microbatch_tokens or global_tokens_per_step))
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=climbmix_dataset(val_files)),
        ),
    )
