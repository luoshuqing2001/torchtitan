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
- tied GPT-2 vocabulary (50304)

Depth is a multiple of 4 so the iRoPE / window cycle closes. Counts below
include the tied embedding once.

- ``d200m``: dim 1024, 12 layers, 16/4 heads, FFN 3072. 209M.
  Closest dense match to the MoE run's ~180M active parameters.
- ``d400m``: dim 1280, 16 layers, 20/5 heads, FFN 3840. 392M.
- ``d700m``: dim 1536, 20 layers, 24/6 heads, FFN 4608. 667M.
- ``d1b``: dim 2048, 16 layers, 32/8 heads, FFN 6144. 942M.
  20 layers at this width would be 1.15B, past the 1B cap, so this step
  gets wider instead of deeper.
"""

from collections.abc import Callable
from functools import partial

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

from .model import GlimmerDenseBlock, GlimmerDenseModel

__all__ = [
    "MODEL_FLAVORS",
    "GlimmerDenseModel",
    "build_model_config",
]


_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}
_VOCAB_SIZE = 50304
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
    "d1b": (2048, 16),
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


def _flavor_config(
    dim: int,
    n_layers: int,
    attn_backend: str,
    *,
    seq_len: int,
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
    converters: list[ModelConfigConverter.Config] | None = None,
) -> GlimmerDenseModel.Config:
    dim, n_layers, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = _flavor_config(
        dim, n_layers, attn_backend, seq_len=context_len
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for converter in converters:
            config = converter.build().convert(config)
    return config
