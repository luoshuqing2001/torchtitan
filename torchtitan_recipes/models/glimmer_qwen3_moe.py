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
from torchtitan.components.optim import Optim
from torchtitan_recipes.models.glimmer_muon import (
    glimmer_muon_optimizer,
    glimmer_wsd_scheduler,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    apply_transforms,
    Float8GroupedLinearConverter,
    Float8LinearConverter,
    TokenDispatcherTransform,
)
from torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher
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

    ``CLIMBMIX_DIR`` selects the shard directory. ``TOKEN_BUDGET`` (default
    10B) sets the run length from the global batch; ``TRAIN_STEPS`` overrides
    that when set. ``ATTN_BACKEND`` selects the attention kernel. The default
    ``varlen`` path activates FlashAttention-3 on Hopper.

    ``DEEPEP=1`` (default) dispatches routed experts with DeepEP v2 and
    ``EP_DEGREE`` (default 4). DeepEP v2 does not pad per-expert token groups,
    and the FP8 grouped GEMM requires that padding, so routed experts stay on
    BF16 grouped GEMM while DeepEP is on. Dense linears still use rowwise FP8.

    Matrix parameters use Moonlight Muon (DistMuon, ``match_rms_adamw``).
    Embeddings, norms, and the LM head stay on AdamW at the same base lr.
    The learning rate is WSD: 2% warmup, 88% at the peak, then 10% cosine
    decay to one tenth of the peak. Checkpoints go to
    ``./outputs/glimmer_qwen3_moe_muon_wsd``.
    """
    attn_backend = os.environ.get("ATTN_BACKEND", "varlen")
    use_deepep = os.environ.get("DEEPEP", "1") != "0"
    # Dense linears (attention, shared experts) use rowwise FP8.
    # Routed experts use FP8 scaled grouped GEMM only without DeepEP.
    # lm_head and the router stay BF16.
    use_fp8 = os.environ.get("FP8", "1") != "0"
    converters = []
    if use_fp8:
        converters.append(
            Float8LinearConverter.Config(
                filter_fqns=["lm_head", "router.gate"],
                model_compile_enabled=True,
            )
        )
    if use_fp8 and not use_deepep:
        converters.append(
            Float8GroupedLinearConverter.Config(model_compile_enabled=True)
        )
    model_config = build_model_config(
        "a200m",
        seq_len=seq_len,
        attn_backend=attn_backend,
        converters=converters or None,
    )
    data_dir = os.environ.get(
        "CLIMBMIX_DIR", "/home/sky/torchtitan/data/climbmix-100b"
    )
    log_freq = int(os.environ.get("LOG_FREQ", "10"))
    # Sequences of ``max_context_length`` packed into one microbatch on each DP rank.
    local_batch = int(os.environ.get("LOCAL_BATCH_SIZE", "96"))
    # tp = cp = pp = 1, so every GPU holds a distinct data-parallel microbatch.
    world_size = int(os.environ.get("NGPU", "4"))
    global_tokens_per_step = local_batch * model_config.max_context_length * world_size
    steps_env = os.environ.get("TRAIN_STEPS")
    if steps_env is not None:
        steps = int(steps_env)
    else:
        token_budget = int(os.environ.get("TOKEN_BUDGET", "10000000000"))
        steps = (token_budget + global_tokens_per_step - 1) // global_tokens_per_step
    ep_degree = int(os.environ.get("EP_DEGREE", "4" if use_deepep else "1"))
    lr = float(os.environ.get("LR", "4e-4"))
    parallelism = ParallelismConfig(
        expert_parallel_degree=ep_degree,
        tensor_parallel_degree=1,
        enable_sequence_parallel=False,
        # The 1.9B model fits unsharded on each H100. Skipping reshard
        # removes an all-gather from every layer.
        fsdp_reshard_after_forward="never",
    )
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
    config = Trainer.Config(
        dump_folder=os.environ.get(
            "DUMP_FOLDER", "./outputs/glimmer_qwen3_moe_muon_wsd"
        ),
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
            optimizer=glimmer_muon_optimizer(
                model_config, lr=lr, parallelism=parallelism
            ),
            lr_scheduler=glimmer_wsd_scheduler(steps),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=(
                local_batch * model_config.max_context_length
            ),
            max_context_length=model_config.max_context_length,
            steps=steps,
            # Token-choice expert parallel synchronizes during dispatch, so the
            # default CUDA-graph capture cannot replay these steps. DeepEP's
            # training dispatch also host-syncs, so graphs stay off.
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
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
    if use_deepep:
        if ep_degree <= 1:
            raise ValueError("DeepEP needs expert_parallel_degree > 1")
        return apply_transforms(
            config,
            [TokenDispatcherTransform(dispatcher=DeepEPTokenDispatcher)],
        )
    return config
