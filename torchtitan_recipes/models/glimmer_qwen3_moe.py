# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Train Glimmer attention + Qwen3 MoE on the local ClimbMix subset."""

import os

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    Float8GroupedLinearConverter,
    Float8LinearConverter,
)
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.hf_datasets.text_datasets import TextProcessor
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.glimmer_qwen3_moe import (
    build_model_config,
    DiagnosticMetricsProcessor,
)
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer


def glimmer_qwen3_moe(seq_len: int | None = None) -> Trainer.Config:
    """1.9B-total / ~180M-active model on the downloaded ClimbMix parquet shards.

    ``CLIMBMIX_DIR`` selects the shard directory. ``TRAIN_STEPS`` and
    ``LOG_FREQ`` override the run length and log interval. ``ATTN_BACKEND``
    selects the attention kernel (``flex`` or ``flex_flash``).
    """
    attn_backend = os.environ.get("ATTN_BACKEND", "flex")
    # Dense linears (attention, shared experts) use rowwise FP8.
    # Routed experts use FP8 scaled grouped GEMM. lm_head and the router stay BF16.
    use_fp8 = os.environ.get("FP8", "1") != "0"
    converters = (
        [
            Float8LinearConverter.Config(
                filter_fqns=["lm_head", "router.gate"],
                model_compile_enabled=True,
            ),
            Float8GroupedLinearConverter.Config(model_compile_enabled=True),
        ]
        if use_fp8
        else None
    )
    model_config = build_model_config(
        "a200m",
        seq_len=seq_len,
        attn_backend=attn_backend,
        converters=converters,
    )
    data_dir = os.environ.get("CLIMBMIX_DIR", "/home/sky/data/climbmix-100b")
    steps = int(os.environ.get("TRAIN_STEPS", "100000"))
    log_freq = int(os.environ.get("LOG_FREQ", "10"))
    # Sequences of ``max_context_length`` packed into one microbatch on each DP rank.
    local_batch = int(os.environ.get("LOCAL_BATCH_SIZE", "64"))
    dataset = SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="parquet",
            split="train",
            load_dataset_kwargs={
                "data_files": {"train": f"{data_dir}/shard_*.parquet"},
            },
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path=os.environ.get(
            "TOKENIZER_DIR", "./assets/hf/gpt2"
        ),
        metrics=DiagnosticMetricsProcessor.Config(
            log_freq=log_freq,
            enable_tensorboard=True,
        ),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=dataset),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=4e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=1000),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=(
                local_batch * model_config.max_context_length
            ),
            max_context_length=model_config.max_context_length,
            steps=steps,
            # Token-choice expert parallel synchronizes during dispatch, so the
            # default CUDA-graph capture cannot replay these steps.
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=int(os.environ.get("EP_DEGREE", "1")),
            tensor_parallel_degree=1,
            enable_sequence_parallel=False,
        ),
        checkpointer=CheckpointManager.Config(interval=200),
        activation_checkpoint=SelectiveAC.Config(),
        compile=LocalCompileConfig(),
        profiler=Profiler.Config(
            enable_profiling=os.environ.get("PROFILE", "0") == "1",
            profile_freq=5,
            profiler_warmup=1,
            profiler_active=1,
            profiler_repeat=1,
        ),
    )
