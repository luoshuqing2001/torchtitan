# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Glimmer attention + Qwen3 MoE flavors.

``a200m`` is sized for roughly 1.9B total parameters and ~180M active
parameters with a tied GPT-2 vocabulary:

- hidden size 768, 12 layers, 12 query heads, 4 KV heads, head dim 64
- 128 routed SwiGLU experts, top-6, expert hidden size 512
- 2 always-on shared SwiGLU experts of the same hidden size
- Muse Glimmer attention: q/k RMSNorm, tuned query scale, sigmoid output
  gate, iRoPE, and a 512/512/512/global sliding-window cycle
"""

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.models.common import Embedding, Linear, Softmax
from torchtitan.models.common.config_utils import (
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.muse_glimmer.flavors import (
    _build_muse_glimmer_attention,
    _gain_norm,
)
from torchtitan.models.muse_glimmer.model import RMSGainCenterNorm

from .model import GlimmerQwen3MoEBlock, GlimmerQwen3MoEModel

__all__ = [
    "MODEL_FLAVORS",
    "GlimmerQwen3MoEModel",
    "build_model_config",
]


_LINEAR_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}

# GPT-2's 50257 tokens, padded so the tied embedding divides common
# tensor-parallel widths.
_VOCAB_SIZE = 50304
_DIM = 768
_N_LAYERS = 12
_N_HEADS = 12
_N_KV_HEADS = 4
_HEAD_DIM = 64
_MOE_HIDDEN = 512
_NUM_EXPERTS = 128
_TOP_K = 6
_NUM_SHARED_EXPERTS = 2
_WINDOW_PATTERN = [512, 512, 512, 0]
_NORM_EPS = 1e-5
_POST_NORM_EPS = 1e-8
_MAX_CONTEXT_LENGTH = 2048


def _residual_output_weight_init(n_layers: int) -> Callable:
    std = 0.02 / (2 * n_layers) ** 0.5
    return partial(nn.init.normal_, std=std)


def _experts_init(n_layers: int) -> dict[str, Callable]:
    return {
        "w1_EFD": _LINEAR_INIT["weight"],
        "w3_EFD": _LINEAR_INIT["weight"],
        "w2_EDF": _residual_output_weight_init(n_layers),
    }


def _output_norm(dim: int) -> RMSGainCenterNorm.Config:
    # Final norm starts at unit scale with a learnable weight initialized to 1.
    return RMSGainCenterNorm.Config(
        normalized_shape=dim,
        eps=_NORM_EPS,
        gain_center=0.0,
        param_init={"weight": nn.init.ones_},
    )


def _shared_expert(dim: int, hidden_dim: int) -> "object":
    return make_ffn_config(
        dim=dim,
        hidden_dim=hidden_dim,
        w1_param_init=_LINEAR_INIT,
        w2w3_param_init=_LINEAR_INIT,
    )


def _a200m(attn_backend: str, *, seq_len: int) -> GlimmerQwen3MoEModel.Config:
    output_init = {
        "weight": _residual_output_weight_init(_N_LAYERS),
        "bias": nn.init.zeros_,
    }
    layers = []
    for layer_id in range(_N_LAYERS):
        layers.append(
            GlimmerQwen3MoEBlock.Config(
                attention_norm=_gain_norm(_DIM, _NORM_EPS, gain_center=1.0),
                ffn_norm=_gain_norm(_DIM, _NORM_EPS, gain_center=1.0),
                post_attention_norm=_gain_norm(_DIM, _POST_NORM_EPS, gain_center=1.0),
                post_ffn_norm=_gain_norm(_DIM, _POST_NORM_EPS, gain_center=1.0),
                attention=_build_muse_glimmer_attention(
                    layer_id=layer_id,
                    n_layers=_N_LAYERS,
                    dim=_DIM,
                    n_heads=_N_HEADS,
                    n_kv_heads=_N_KV_HEADS,
                    head_dim=_HEAD_DIM,
                    max_context_length=seq_len,
                    window_pattern=_WINDOW_PATTERN,
                    attn_backend=attn_backend,
                ),
                moe=make_moe_config(
                    num_experts=_NUM_EXPERTS,
                    load_balance_coeff=1e-3,
                    router=make_router_config(
                        dim=_DIM,
                        num_experts=_NUM_EXPERTS,
                        gate_param_init=_LINEAR_INIT,
                        top_k=_TOP_K,
                        score_func=Softmax.Config(),
                        route_norm=True,
                    ),
                    routed_experts=make_routed_experts_config(
                        dim=_DIM,
                        hidden_dim=_MOE_HIDDEN,
                        num_experts=_NUM_EXPERTS,
                        top_k=_TOP_K,
                        param_init=_experts_init(_N_LAYERS),
                    ),
                ),
                shared_experts=[
                    _shared_expert(_DIM, _MOE_HIDDEN)
                    for _ in range(_NUM_SHARED_EXPERTS)
                ],
            )
        )
    return GlimmerQwen3MoEModel.Config(
        max_context_length=seq_len,
        vocab_size=_VOCAB_SIZE,
        dim=_DIM,
        norm=_output_norm(_DIM),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=_VOCAB_SIZE,
            embedding_dim=_DIM,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=_DIM,
            out_features=_VOCAB_SIZE,
            bias=False,
            param_init=output_init,
        ),
        layers=layers,
    )


MODEL_FLAVORS = {
    "a200m": (_a200m, _MAX_CONTEXT_LENGTH),
}


def build_model_config(
    flavor: str = "a200m",
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> GlimmerQwen3MoEModel.Config:
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(attn_backend, seq_len=context_len)
    if converters is not None:
        validate_converter_compatibility(converters)
        for converter in converters:
            config = converter.build().convert(config)
    return config
