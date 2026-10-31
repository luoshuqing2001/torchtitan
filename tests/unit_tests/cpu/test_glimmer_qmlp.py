# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import re

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.glimmer_dense import build_model_config
from torchtitan.models.glimmer_dense.qmlp import GELU_GAIN
from torchtitan_recipes.models.glimmer_muon import (
    _MUON_PATTERN,
    glimmer_muon_optimizer,
)


def _meta_model(flavor: str, arch: str):
    with torch.device("meta"):
        return build_model_config(flavor, arch=arch).build()


def _expected_qmlp_params(dim: int, n_layers: int, ratio: float = 1.375) -> int:
    vocab = 100352
    swa, full = 3 * n_layers // 4, n_layers // 4
    kv, wo_gate, norms = dim * dim // 2, 2 * dim * dim, 2 * dim
    return (
        vocab * dim
        + n_layers * (kv + wo_gate + norms)
        + full * dim * dim
        + swa * 2 * dim * round(ratio * dim)
        + dim
    )


@pytest.mark.parametrize(
    "flavor,dim,n_layers", [("d200m", 1024, 12), ("d400m", 1280, 16)]
)
def test_qmlp_parameter_count(flavor, dim, n_layers):
    model = _meta_model(flavor, "qmlp")
    count = sum(p.numel() for p in model.parameters())
    assert count == _expected_qmlp_params(dim, n_layers)


def test_ffn_arch_unchanged():
    model = _meta_model("d400m", "ffn")
    assert sum(p.numel() for p in model.parameters()) == 456_213_760


def test_qmlp_layout_keeps_attention_and_drops_ffn():
    model = _meta_model("d400m", "qmlp")
    assert len(model.layers) == 16
    for index, layer in enumerate(model.layers.values()):
        assert not hasattr(layer, "feed_forward")
        assert not hasattr(layer, "ffn_norm")
        attention = layer.attention
        assert attention.o_gate is not None
        swa = attention.window_size is not None
        assert swa == (index % 4 != 3)
        assert (attention.rope is not None) == swa
        assert hasattr(attention.qkv_linear, "q_up") == swa
        if swa:
            assert attention.qkv_linear.q_up.weight.shape == (1760, 1280)
            assert attention.qkv_linear.q_down.weight.shape == (1280, 1760)
            assert attention.qkv_linear.wkv.weight.shape == (2 * 5 * 64, 1280)


def test_qmlp_init_std():
    torch.manual_seed(0)
    model = build_model_config("d200m", arch="qmlp").build()
    model.init_states()
    qkv = model.layers["0"].attention.qkv_linear
    assert qkv.q_up.weight.std().item() == pytest.approx(
        GELU_GAIN / math.sqrt(1024), rel=0.02
    )
    assert qkv.q_down.weight.std().item() == pytest.approx(
        1 / math.sqrt(1408), rel=0.02
    )
    assert qkv.wkv.weight.std().item() == pytest.approx(0.02, rel=0.05)


def test_qmlp_matrices_use_muon():
    config = build_model_config("d400m", arch="qmlp")
    optimizer = glimmer_muon_optimizer(
        config, lr=3e-4, parallelism=ParallelismConfig()
    )
    layouts = optimizer.optimizers[0].compute_sharding_by_fqn
    with torch.device("meta"):
        model = config.build()
    for name, parameter in model.named_parameters():
        if not name.startswith("layers.") or parameter.ndim != 2:
            continue
        assert re.search(_MUON_PATTERN, name), name
        assert name in layouts, name
