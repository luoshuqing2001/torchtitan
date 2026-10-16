# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Train a dense Muse Glimmer model on the local ClimbMix subset.

``FLAVOR`` selects the scaling step (``d200m``, ``d400m``, ``d700m``,
``d1b``). Checkpoints go to ``./outputs/glimmer_dense/<flavor>`` so they
do not share a folder with the MoE run.
"""

import os

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import Optim
from torchtitan_recipes.models.glimmer_muon import (
    glimmer_muon_optimizer,
    glimmer_wsd_scheduler,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import Float8LinearConverter
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.hf_datasets.text_datasets import TextProcessor
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.glimmer_dense import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

# Per-GPU sequences. Sized so the dense FFN activations fit on one H100
# next to the MoE run's memory headroom. Override with LOCAL_BATCH_SIZE.
_DEFAULT_LOCAL_BATCH = {
    "d200m": 64,
    "d400m": 32,
    "d700m": 16,
    "d1b": 8,
}
# Same 4e-4 as the ~180M-active MoE run at d200m, then lower for width.
_DEFAULT_LR = {
    "d200m": 4e-4,
    "d400m": 3e-4,
    "d700m": 2.5e-4,
    "d1b": 2e-4,
}


def glimmer_dense(seq_len: int | None = None) -> Trainer.Config:
    """Dense Glimmer baseline on the downloaded ClimbMix parquet shards.

    ``FLAVOR`` selects the ladder step. ``TOKEN_BUDGET`` (default 10B) sets
    the run length; ``TRAIN_STEPS`` overrides that when set. Attention uses
    FlashAttention-3 on Hopper when ``ATTN_BACKEND=varlen``.

    Matrix parameters use Moonlight Muon. Embeddings, norms, and the LM head
    stay on AdamW at the same base learning rate. The schedule is WSD: 2%
    warmup, 88% at the peak, then 10% cosine decay to one tenth of the peak.
    """
    flavor = os.environ.get("FLAVOR", "d200m")
    if flavor not in _DEFAULT_LOCAL_BATCH:
        raise ValueError(
            f"Unknown FLAVOR {flavor!r}. Choose from {sorted(_DEFAULT_LOCAL_BATCH)}."
        )
    attn_backend = os.environ.get("ATTN_BACKEND", "varlen")
    use_fp8 = os.environ.get("FP8", "1") != "0"
    converters = []
    if use_fp8:
        converters.append(
            Float8LinearConverter.Config(
                filter_fqns=["lm_head"],
                model_compile_enabled=True,
            )
        )
    model_config = build_model_config(
        flavor,
        seq_len=seq_len,
        attn_backend=attn_backend,
        converters=converters or None,
    )
    data_dir = os.environ.get(
        "CLIMBMIX_DIR", "/home/sky/torchtitan/data/climbmix-100b"
    )
    log_freq = int(os.environ.get("LOG_FREQ", "10"))
    local_batch = int(
        os.environ.get("LOCAL_BATCH_SIZE", str(_DEFAULT_LOCAL_BATCH[flavor]))
    )
    lr = float(os.environ.get("LR", str(_DEFAULT_LR[flavor])))
    world_size = int(os.environ.get("NGPU", "4"))
    global_tokens_per_step = local_batch * model_config.max_context_length * world_size
    steps_env = os.environ.get("TRAIN_STEPS")
    if steps_env is not None:
        steps = int(steps_env)
    else:
        token_budget = int(os.environ.get("TOKEN_BUDGET", "10000000000"))
        steps = (token_budget + global_tokens_per_step - 1) // global_tokens_per_step
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
    parallelism = ParallelismConfig(
        expert_parallel_degree=1,
        tensor_parallel_degree=1,
        enable_sequence_parallel=False,
        fsdp_reshard_after_forward="never",
    )
    return Trainer.Config(
        dump_folder=os.environ.get(
            "DUMP_FOLDER", f"./outputs/glimmer_dense/{flavor}"
        ),
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path=os.environ.get("TOKENIZER_DIR", "./assets/hf/gpt2"),
        metrics=MetricsProcessor.Config(
            log_freq=log_freq,
            enable_tensorboard=True,
        ),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=dataset),
        ),
        parallelism=parallelism,
        optim=Optim.Config(
            optimizer=glimmer_muon_optimizer(
                model_config,
                lr=lr,
                parallelism=parallelism,
            ),
            lr_scheduler=glimmer_wsd_scheduler(steps),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=(
                local_batch * model_config.max_context_length
            ),
            max_context_length=model_config.max_context_length,
            steps=steps,
            # Varlen metadata is data-dependent unless max_num_documents is
            # fixed, so CUDA graphs stay off.
            disable_cuda_graphs=True,
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
