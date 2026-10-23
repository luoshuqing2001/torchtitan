# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dense Muse Glimmer scaling ladder, capped at about 1B parameters.

Every flavor uses the same block as the MoE run's attention side:

- head dim 64, GQA ratio 4
- SwiGLU hidden size ``3 * dim`` (gate and up fused in ``w13``, down in ``w2``)
- q/k RMSNorm, tuned query scale, sigmoid output gate
- iRoPE with a 512/512/512/global sliding-window cycle
- tied OLMo3 vocabulary, padded from 100278 tokenizer ids to 100352

Depth is a multiple of 4 so the iRoPE / window cycle closes. Counts below
include the tied embedding once. The step names stay ``d200m`` through
``d1b``; the smaller OLMo3 table is why the totals sit below the old
Qwen3.8 counts. PLE and Engram at the same ``ple_dim`` match in total
parameters. ``engram`` is DeepSeek Engram (``deepseek_engram.py``);
``engram_legacy`` is the earlier single-layer variant kept for the
engram_dram and engram_optim experiments.

- ``d200m``: dim 1024, 12 layers, 16/4 heads, FFN 3072. 260,097,024.
- ``d400m``: dim 1280, 16 layers, 20/5 heads, FFN 3840. 456,213,760.
- ``d700m``: dim 1536, 20 layers, 24/6 heads, FFN 4608. 744,089,088.
- ``d1b``: dim 2048, 24 layers, 32/8 heads, FFN 6144. 1,464,010,752.
"""

from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn

from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.models.common import Embedding, Linear
from torchtitan.models.common.config_utils import make_ffn_config
from torchtitan.models.muse_glimmer.flavors import (
    _build_muse_glimmer_attention,
    _depth_init,
    _gain_norm,
    _LINEAR_INIT,
)
from torchtitan.models.muse_glimmer.model import RMSGainCenterNorm

from .deepseek_engram import (
    active_params_per_layer,
    compression_lookup,
    DeepSeekEngram,
    engram_layer_indices,
    EngramLayer,
    HashedTable,
    head_table_sizes,
    layer_multipliers,
    NUM_HEADS,
)
from .engram import EngramTable, EngramUpdate, NGramEngram, rows_per_head_matching_ple
from .model import GlimmerDenseBlock, GlimmerDenseModel
from .ple import _PLE_DIM, LayerPLE, PerLayerContext, PerLayerEmbedder, PLE_DIM_TIERS

__all__ = [
    "MODEL_FLAVORS",
    "GlimmerDenseModel",
    "build_model_config",
]


_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}
# allenai/Olmo-3-1025-7B tokenizer has 100278 ids. The embedding and tied
# LM head are padded to the next multiple of 256.
_VOCAB_SIZE = 100352
_EOS_TOKEN_ID = 100257
_HEAD_DIM = 64
_FFN_MULTIPLIER = 3
_WINDOW_PATTERN = [512, 512, 512, 0]
_NORM_EPS = 1e-5
_POST_NORM_EPS = 1e-8
_MAX_CONTEXT_LENGTH = 2048

# (dim, n_layers). Heads are dim / 64; KV heads are heads / 4.
_FLAVOR_SHAPES: dict[str, tuple[int, int]] = {
    "d200m": (1024, 12),
    "d400m": (1280, 16),
    "d700m": (1536, 20),
    "d1b": (2048, 24),
}


def _residual_output_weight_init(n_layers: int) -> Callable:
    std = 0.02 / (2 * n_layers) ** 0.5
    return partial(nn.init.normal_, std=std)


def _output_norm(dim: int) -> RMSGainCenterNorm.Config:
    return RMSGainCenterNorm.Config(
        normalized_shape=dim,
        eps=_NORM_EPS,
        gain_center=0.0,
        param_init={"weight": nn.init.ones_},
    )


def _layer_ple(dim: int, ple_dim: int) -> LayerPLE.Config:
    return LayerPLE.Config(
        dim=dim,
        ple_dim=ple_dim,
        param_init={
            "gate_weight": partial(nn.init.normal_, std=0.02),
            "proj_weight": partial(nn.init.normal_, std=0.02),
            "norm_weight": nn.init.ones_,
        },
    )


def _ple_config(dim: int, n_layers: int, ple_dim: int) -> PerLayerEmbedder.Config:
    width = n_layers * ple_dim
    return PerLayerEmbedder.Config(
        vocab_size=_VOCAB_SIZE,
        dim=dim,
        num_layers=n_layers,
        ple_dim=ple_dim,
        table=Embedding.Config(
            num_embeddings=_VOCAB_SIZE,
            embedding_dim=width,
            param_init=_EMBEDDING_INIT,
        ),
        context=PerLayerContext.Config(
            dim=dim,
            num_layers=n_layers,
            ple_dim=ple_dim,
            param_init={
                "weight": partial(nn.init.normal_, std=0.02),
                "norm_weight": nn.init.ones_,
            },
        ),
    )


def _init_engram_conv(tensor):
    nn.init.zeros_(tensor)
    with torch.no_grad():
        tensor[0].fill_(1)
    return tensor


def _engram_config(dim: int, n_layers: int, ple_dim: int) -> NGramEngram.Config:
    rows, remainder = rows_per_head_matching_ple(
        dim,
        n_layers,
        vocab_size=_VOCAB_SIZE,
        ple_dim=ple_dim,
    )
    table_init = {"weight": partial(nn.init.normal_, std=0.02)}
    if remainder:
        table_init["remainder_weight"] = nn.init.zeros_
    return NGramEngram.Config(
        dim=dim,
        n_layers=n_layers,
        ple_dim=ple_dim,
        eos_token_id=_EOS_TOKEN_ID,
        rows_per_head=rows,
        remainder=remainder,
        table=EngramTable.Config(
            dim=dim,
            rows_per_head=rows,
            eos_token_id=_EOS_TOKEN_ID,
            remainder=remainder,
            param_init=table_init,
        ),
        update=EngramUpdate.Config(
            dim=dim,
            n_layers=n_layers,
            ple_dim=ple_dim,
            eos_token_id=_EOS_TOKEN_ID,
            param_init={
                "key_weight": nn.init.zeros_,
                "value_up": partial(nn.init.normal_, std=0.02),
                "value_down": nn.init.zeros_,
                "value_scale": nn.init.ones_,
                "conv_weight": _init_engram_conv,
            },
        ),
    )


def _ple_extra_params(dim: int, n_layers: int, ple_dim: int) -> int:
    """Parameters PLE adds to the dense model at this ``ple_dim``."""
    with torch.device("meta"):
        modules = [_ple_config(dim, n_layers, ple_dim).build()]
        modules += [_layer_ple(dim, ple_dim).build() for _ in range(n_layers)]
    return sum(p.numel() for module in modules for p in module.parameters())


def _deepseek_engram_config(
    dim: int,
    n_layers: int,
    ple_dim: int,
    tokenizer_path: str | None,
) -> DeepSeekEngram.Config:
    """Two Engram layers whose total parameters match PLE at ``ple_dim``.

    The concatenated memory width is ``4 * ple_dim`` (1024 at the default
    256, the reference's two orders x 512), split over 16 heads. The tables
    take what is left of the PLE budget after the projections; each head gets
    a distinct prime row count, chosen as large as the budget allows.
    """
    memory_dim = 4 * ple_dim
    head_dim = memory_dim // NUM_HEADS
    indices = engram_layer_indices(n_layers)
    budget = _ple_extra_params(dim, n_layers, ple_dim) - len(indices) * (
        active_params_per_layer(dim, memory_dim)
    )
    rows = budget // (head_dim * NUM_HEADS * len(indices))
    sizes = head_table_sizes(rows, len(indices))
    # Distinct primes overshoot the budget a little; step the start down.
    while (excess := sum(sum(layer) for layer in sizes) * head_dim - budget) > 0:
        rows -= max(1, excess // (head_dim * 64))
        sizes = head_table_sizes(rows, len(indices))
    lookup = compression_lookup(tokenizer_path, _VOCAB_SIZE)
    compressed_vocab = int(lookup.max()) + 1
    layer_init = {
        "key_weight": partial(nn.init.normal_, std=0.02),
        "value_weight": _residual_output_weight_init(n_layers),
        "query_norm_weight": nn.init.ones_,
        "key_norm_weight": nn.init.ones_,
        "conv_norm_weight": nn.init.ones_,
        "conv_weight": nn.init.zeros_,
    }
    return DeepSeekEngram.Config(
        layers={
            index: EngramLayer.Config(
                dim=dim,
                memory_dim=memory_dim,
                layer_id=index,
                table_sizes=sizes[position],
                multipliers=layer_multipliers(index, compressed_vocab),
                pad_id=int(lookup[_EOS_TOKEN_ID]),
                eos_token_id=_EOS_TOKEN_ID,
                vocab_size=_VOCAB_SIZE,
                tokenizer_path=tokenizer_path,
                table=HashedTable.Config(
                    num_rows=sum(sizes[position]),
                    head_dim=head_dim,
                    param_init={"weight": partial(nn.init.normal_, std=0.02)},
                ),
                param_init=layer_init,
            )
            for position, index in enumerate(indices)
        }
    )


def _flavor_config(
    dim: int,
    n_layers: int,
    attn_backend: str,
    *,
    seq_len: int,
    memory: str = "none",
    ple_dim: int = _PLE_DIM,
    tokenizer_path: str | None = None,
) -> GlimmerDenseModel.Config:
    n_heads = dim // _HEAD_DIM
    n_kv_heads = n_heads // 4
    hidden_dim = _FFN_MULTIPLIER * dim
    output_init = {
        "weight": _residual_output_weight_init(n_layers),
        "bias": nn.init.zeros_,
    }
    layers = []
    for layer_id in range(n_layers):
        layers.append(
            GlimmerDenseBlock.Config(
                attention_norm=_gain_norm(dim, _NORM_EPS, gain_center=1.0),
                ffn_norm=_gain_norm(dim, _NORM_EPS, gain_center=1.0),
                post_attention_norm=_gain_norm(dim, _POST_NORM_EPS, gain_center=1.0),
                post_ffn_norm=_gain_norm(dim, _POST_NORM_EPS, gain_center=1.0),
                attention=_build_muse_glimmer_attention(
                    layer_id=layer_id,
                    n_layers=n_layers,
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=_HEAD_DIM,
                    max_context_length=seq_len,
                    window_pattern=_WINDOW_PATTERN,
                    attn_backend=attn_backend,
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
                layer_ple=_layer_ple(dim, ple_dim) if memory == "ple" else None,
            )
        )
    return GlimmerDenseModel.Config(
        max_context_length=seq_len,
        vocab_size=_VOCAB_SIZE,
        dim=dim,
        norm=_output_norm(dim),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=_VOCAB_SIZE,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=_VOCAB_SIZE,
            bias=False,
            param_init=output_init,
        ),
        layers=layers,
        ple=_ple_config(dim, n_layers, ple_dim) if memory == "ple" else None,
        engram=(
            _engram_config(dim, n_layers, ple_dim)
            if memory == "engram_legacy"
            else None
        ),
        deepseek_engram=(
            _deepseek_engram_config(dim, n_layers, ple_dim, tokenizer_path)
            if memory == "engram"
            else None
        ),
    )


MODEL_FLAVORS = {
    name: (dim, n_layers, _MAX_CONTEXT_LENGTH)
    for name, (dim, n_layers) in _FLAVOR_SHAPES.items()
}


def build_model_config(
    flavor: str = "d200m",
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    memory: str = "none",
    ple_dim: int = _PLE_DIM,
    converters: list[ModelConfigConverter.Config] | None = None,
    tokenizer_path: str | None = None,
) -> GlimmerDenseModel.Config:
    dim, n_layers, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    if memory not in ("none", "ple", "engram", "engram_legacy"):
        raise ValueError(
            "memory must be 'none', 'ple', 'engram', or 'engram_legacy', "
            f"got {memory!r}"
        )
    if ple_dim not in PLE_DIM_TIERS:
        raise ValueError(f"ple_dim must be one of {PLE_DIM_TIERS}, got {ple_dim}")
    config = _flavor_config(
        dim,
        n_layers,
        attn_backend,
        seq_len=context_len,
        memory=memory,
        ple_dim=ple_dim,
        tokenizer_path=tokenizer_path,
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for converter in converters:
            config = converter.build().convert(config)
    return config
