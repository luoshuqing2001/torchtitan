# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Train a dense Muse Glimmer model on the local ClimbMix subset.

``FLAVOR`` selects the scaling step (``d200m``, ``d400m``, ``d700m``,
``d1b``). Checkpoints go to ``./outputs/glimmer_dense_olmo3/<flavor>``,
or ``./outputs/glimmer_dense_olmo3_ple/<flavor>`` when ``PLE=1``.
"""

import os
from dataclasses import replace

import torch

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import Optim
from torchtitan.config import Function, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import Float8LinearConverter
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.glimmer_dense import build_model_config
from torchtitan.models.glimmer_dense.ple import _PLE_DIM, PLE_DIM_TIERS
from torchtitan.models.utils import get_nparams_and_active_nparams
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

from torchtitan_recipes.models.climbmix import (
    climbmix_dataset,
    climbmix_train_and_val_files,
    climbmix_validator,
)
from torchtitan_recipes.models.glimmer_muon import (
    glimmer_muon_optimizer,
    glimmer_wsd_scheduler,
)

# Per-GPU sequences. Sized so the dense FFN activations fit on one H100
# next to the MoE run's memory headroom. Override with LOCAL_BATCH_SIZE.
_DEFAULT_LOCAL_BATCH = {
    "d200m": 64,
    "d400m": 32,
    "d700m": 16,
    "d1b": 8,
}
# Global tokens per optimizer step, fixed per flavor near the compute-optimal
# batch B = 0.292 * C^0.3271 (DeepSeek LLM, C = 6 * N * 20N). Dense, PLE and
# Engram at one flavor share it; LOCAL_BATCH_SIZE only sets the microbatch and
# gradient accumulation makes up the rest. Override with GLOBAL_BATCH_TOKENS.
_DEFAULT_GLOBAL_BATCH_TOKENS = {
    "d200m": 2**19,
    "d400m": 2**19,
    "d700m": 2**20,
    "d1b": 2**20,
}
# Validation every 2^28 tokens, a multiple of every global batch above, so
# val points of all runs land on the same token counts.
_VAL_EVERY_TOKENS = 2**28
# Checkpoints are placed by tokens, not steps, so every run saves at the
# same token counts. Resumable state every _VAL_EVERY_TOKENS (one per val
# point, CKPT_KEEP most recent kept); snapshots at doubling token counts from
# 2^28 (268M, 537M, 1.07B, ...), evenly spaced on a log axis, are exempt from
# that purge and kept for evaluation and decay branches.
_FIRST_SNAPSHOT_TOKENS = 2**28


def _is_snapshot_step(step: int, *, tokens_per_step: int) -> bool:
    tokens = step * tokens_per_step
    return tokens >= _FIRST_SNAPSHOT_TOKENS and tokens & (tokens - 1) == 0


# 20 tokens per dense parameter. PLE and Engram at the same flavor
# use this budget too, so the three runs see the same data.
_TOKENS_PER_PARAM = 20
# Same 4e-4 as the ~180M-active MoE run at d200m, then lower for width.
_DEFAULT_LR = {
    "d200m": 4e-4,
    "d400m": 3e-4,
    "d700m": 2.5e-4,
    "d1b": 2e-4,
}


def _memory_variant(raw: str) -> str:
    """Map ``PLE`` to ``none``, ``ple``, ``engram``, or ``engram_legacy``."""
    value = raw.strip().lower()
    if value in ("0", "none", "off", ""):
        return "none"
    if value in ("1", "ple", "gemma"):
        return "ple"
    if value in ("engram", "engram_legacy"):
        return value
    raise ValueError(f"PLE must be 0, ple, engram, or engram_legacy, got {raw!r}")


def _ple_dim(raw: str) -> int:
    """Map ``PLE_DIM`` to one of the per-layer intermediate widths."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"PLE_DIM must be one of {PLE_DIM_TIERS}, got {raw!r}"
        ) from exc
    if value not in PLE_DIM_TIERS:
        raise ValueError(f"PLE_DIM must be one of {PLE_DIM_TIERS}, got {value}")
    return value


def _default_token_budget(
    flavor: str,
    *,
    seq_len: int | None,
    attn_backend: str,
) -> int:
    """20 tokens per dense parameter at this flavor.

    PLE and Engram keep the dense budget. Their extra parameters are lookups,
    and a longer run would change the data as well as the architecture.
    """
    dense = build_model_config(
        flavor,
        seq_len=seq_len,
        attn_backend=attn_backend,
        memory="none",
    )
    with torch.device("meta"):
        model = dense.build()
    total, _active = get_nparams_and_active_nparams(model)
    return _TOKENS_PER_PARAM * total


def _dump_folder(flavor: str, memory: str, ple_dim: int) -> str:
    # OLMo3 checkpoints stay out of the Qwen3.8 dump folders.
    name = flavor if ple_dim == _PLE_DIM else f"{flavor}_ple{ple_dim}"
    if memory == "ple":
        return f"./outputs/glimmer_dense_olmo3_ple/{name}"
    if memory == "engram":
        return f"./outputs/glimmer_dense_olmo3_deepseek_engram/{name}"
    if memory == "engram_legacy":
        return f"./outputs/glimmer_dense_olmo3_engram/{name}"
    return f"./outputs/glimmer_dense_olmo3/{flavor}"


def glimmer_dense(seq_len: int | None = None) -> Trainer.Config:
    """Dense Glimmer baseline on the downloaded ClimbMix parquet shards.

    ``FLAVOR`` selects the ladder step. ``TOKEN_BUDGET`` sets the run length
    and defaults to 20 tokens per dense parameter at that flavor. PLE and
    Engram use the same budget. ``TRAIN_STEPS`` overrides that when set.
    Attention uses
    FlashAttention-3 on Hopper when ``ATTN_BACKEND=varlen``. The tokenizer is
    OLMo3. ``PLE=ple`` (or ``1``) adds Gemma 4 per-layer embeddings.
    ``PLE_ASYNC_TRAINING=1`` opts into DRAM staging; default 0 uses ordinary
    GPU PLE parameters, native autograd and FSDP.
    ``PLE_UNBIND_INPUTS=1`` opts into a single PLE layer-gradient assembly;
    it defaults to 0 and is independent of DRAM staging.
    ``PLE_DIM`` selects the per-layer width: 256, 512, or 1024.
    ``PLE_CHECKPOINT=1`` recomputes PLE entrance activations in backward.
    ``PLE_CPU_FUSED_ADAM=1`` opts into native fused CPU AdamW for PLE only;
    it can change floating-point rounding relative to foreach AdamW.
    ``CPU_NORM_CHUNK_SIZE=65536`` enables parallel CPU FP32 gradient L2 norms;
    it changes reduction rounding and defaults to the native path (0).
    ``PLE_MIXED_PRECISION_BACKWARD=1`` computes PLE linear weight gradients in
    the activation dtype, as native mixed-precision GPU linear layers do.
    ``PLE_RECOMPUTE_LINEAR=1`` prevents selective checkpointing from retaining
    ordinary linear outputs. FP8 matmuls keep their existing checkpoint policy.
    ``PLE=engram`` adds the n-gram table sized to that same total and the
    same active parameter count. Checkpoints for width 256 go to
    ``./outputs/glimmer_dense_olmo3_ple/<flavor>`` or
    ``./outputs/glimmer_dense_olmo3_engram/<flavor>``. Wider tiers use
    ``<flavor>_ple<PLE_DIM>``.

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
    memory = _memory_variant(os.environ.get("PLE", "0"))
    async_flag = os.environ.get("PLE_ASYNC_TRAINING", "0")
    if async_flag not in ("0", "1"):
        raise ValueError("PLE_ASYNC_TRAINING must be 0 or 1")
    ple_async_training = async_flag == "1"
    if ple_async_training and memory != "ple":
        raise ValueError("PLE_ASYNC_TRAINING=1 requires PLE=ple")
    unbind_flag = os.environ.get("PLE_UNBIND_INPUTS", "0")
    if unbind_flag not in ("0", "1"):
        raise ValueError("PLE_UNBIND_INPUTS must be 0 or 1")
    if unbind_flag == "1" and memory != "ple":
        raise ValueError("PLE_UNBIND_INPUTS=1 requires PLE=ple")
    ple_dim = _ple_dim(os.environ.get("PLE_DIM", str(_PLE_DIM)))
    use_fp8 = os.environ.get("FP8", "1") != "0"
    converters = []
    if use_fp8:
        converters.append(
            Float8LinearConverter.Config(
                filter_fqns=["lm_head"],
                model_compile_enabled=True,
            )
        )
    tokenizer_dir = os.environ.get("TOKENIZER_DIR", "./assets/hf/OLMo3")
    model_config = build_model_config(
        flavor,
        seq_len=seq_len,
        attn_backend=attn_backend,
        memory=memory,
        ple_dim=ple_dim,
        converters=converters or None,
        tokenizer_path=os.path.join(tokenizer_dir, "tokenizer.json"),
    )
    model_config.ple_async_training = ple_async_training
    model_config.ple_unbind_inputs = unbind_flag == "1"
    if model_config.ple is not None:
        model_config.ple.activation_checkpoint = (
            os.environ.get("PLE_CHECKPOINT", "0") == "1"
        )
        mixed_backward = os.environ.get("PLE_MIXED_PRECISION_BACKWARD", "0") == "1"
        assert model_config.ple.context is not None
        model_config.ple.context.mixed_precision_backward = mixed_backward
        for layer in model_config.layers:
            if layer.layer_ple is not None:
                layer.layer_ple.mixed_precision_backward = mixed_backward
    data_dir = os.environ.get("CLIMBMIX_DIR", "/home/sky/torchtitan/data/climbmix-100b")
    log_freq = int(os.environ.get("LOG_FREQ", "10"))
    local_batch = int(
        os.environ.get("LOCAL_BATCH_SIZE", str(_DEFAULT_LOCAL_BATCH[flavor]))
    )
    lr = float(os.environ.get("LR", str(_DEFAULT_LR[flavor])))
    world_size = int(os.environ.get("NGPU", "4"))
    microbatch_tokens = local_batch * model_config.max_context_length * world_size
    global_tokens_per_step = int(
        os.environ.get("GLOBAL_BATCH_TOKENS", str(_DEFAULT_GLOBAL_BATCH_TOKENS[flavor]))
    )
    if global_tokens_per_step % microbatch_tokens != 0:
        raise ValueError(
            f"GLOBAL_BATCH_TOKENS={global_tokens_per_step} is not a multiple of "
            f"LOCAL_BATCH_SIZE * seq_len * NGPU = {microbatch_tokens}"
        )
    if _VAL_EVERY_TOKENS % global_tokens_per_step != 0:
        raise ValueError(
            f"GLOBAL_BATCH_TOKENS={global_tokens_per_step} must divide "
            f"the validation interval {_VAL_EVERY_TOKENS}"
        )
    steps_env = os.environ.get("TRAIN_STEPS")
    if steps_env is not None:
        steps = int(steps_env)
    else:
        token_budget = int(
            os.environ.get(
                "TOKEN_BUDGET",
                str(
                    _default_token_budget(
                        flavor,
                        seq_len=seq_len,
                        attn_backend=attn_backend,
                    )
                ),
            )
        )
        # Round the budget up to whole validation intervals so the last step
        # is also a validation and checkpoint step (the trainer validates only
        # at multiples of the interval, never specially at the last step).
        intervals = -(-token_budget // _VAL_EVERY_TOKENS)
        steps = intervals * (_VAL_EVERY_TOKENS // global_tokens_per_step)
    train_files, val_files = climbmix_train_and_val_files(data_dir)
    dataset = climbmix_dataset(train_files)
    parallelism = ParallelismConfig(
        expert_parallel_degree=1,
        tensor_parallel_degree=1,
        enable_sequence_parallel=False,
        fsdp_reshard_after_forward="never",
    )
    optimizer = glimmer_muon_optimizer(model_config, lr=lr, parallelism=parallelism)
    if model_config.deepseek_engram is not None:
        # The Engram paper trains the n-gram tables with Adam at 5x the
        # learning rate and no weight decay.
        optimizer.optimizers.insert(
            -1,
            replace(
                optimizer.optimizers[-1],
                pattern=r"^deepseek_engram\.layers\.\d+\.table\.weight$",
                lr=5 * lr,
                weight_decay=0.0,
            ),
        )
    if ple_async_training and os.environ.get("PLE_CPU_FUSED_ADAM", "0") == "1":
        optimizer.optimizers.insert(
            -1,
            replace(
                optimizer.optimizers[-1],
                pattern=r"(?:^ple\.|\.layer_ple\.)",
                foreach=False,
                fused=True,
            ),
        )
    return Trainer.Config(
        dump_folder=os.environ.get(
            "DUMP_FOLDER",
            _dump_folder(flavor, memory, ple_dim),
        ),
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path=tokenizer_dir,
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
            optimizer=optimizer,
            cpu_norm_chunk_size=int(os.environ.get("CPU_NORM_CHUNK_SIZE", "0")),
            lr_scheduler=glimmer_wsd_scheduler(steps),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=(
                local_batch * model_config.max_context_length
            ),
            max_context_length=model_config.max_context_length,
            num_tokens_per_train_step=global_tokens_per_step,
            steps=steps,
            # Varlen metadata is data-dependent unless max_num_documents is
            # fixed, so CUDA graphs stay off.
            disable_cuda_graphs=True,
        ),
        validator=climbmix_validator(
            val_files,
            global_tokens_per_step=global_tokens_per_step,
            microbatch_tokens=microbatch_tokens,
            val_every_tokens=_VAL_EVERY_TOKENS,
        ),
        checkpointer=CheckpointManager.Config(
            interval=_VAL_EVERY_TOKENS // global_tokens_per_step,
            keep_latest_k=int(os.environ.get("CKPT_KEEP", "10")),
            purge_exempt=Function.Config(
                # pyrefly: ignore [bad-argument-type]
                fn=lambda step: _is_snapshot_step(
                    step, tokens_per_step=global_tokens_per_step
                )
            ),
        ),
        activation_checkpoint=SelectiveAC.Config(
            recompute_linear=(
                memory == "ple" and os.environ.get("PLE_RECOMPUTE_LINEAR", "0") == "1"
            ),
        ),
        compile=LocalCompileConfig(),
        profiler=Profiler.Config(
            enable_profiling=os.environ.get("PROFILE", "0") == "1",
            profile_freq=5,
            profiler_warmup=1,
            profiler_active=1,
            profiler_repeat=1,
        ),
    )
