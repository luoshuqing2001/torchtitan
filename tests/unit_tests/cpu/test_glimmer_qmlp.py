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


@pytest.mark.parametrize(
    "flavor,expected", [("d200m", 216_044_544), ("d400m", 364_442_880)]
)
def test_qmlpdeep_parameter_count(flavor, expected):
    model = _meta_model(flavor, "qmlpdeep")
    assert sum(p.numel() for p in model.parameters()) == expected


def test_qmlpdeep_layout_keeps_global_layers_at_dense_depths():
    model = _meta_model("d200m", "qmlpdeep")
    assert len(model.layers) == 18
    full_layers = []
    for index, layer in enumerate(model.layers.values()):
        attention = layer.attention
        assert not hasattr(layer, "feed_forward")
        swa = attention.window_size is not None
        assert (attention.rope is not None) == swa
        assert hasattr(attention.qkv_linear, "q_up") == swa
        if swa:
            assert attention.window_size == 512
            assert attention.qkv_linear.q_up.weight.shape == (2048, 1024)
        else:
            full_layers.append(index)
    # Dense d200m has its 3 global layers at 4/12, 8/12, 12/12 of the depth.
    assert full_layers == [5, 11, 17]


def test_qmlpdeep_init_std():
    torch.manual_seed(0)
    model = build_model_config("d200m", arch="qmlpdeep").build()
    model.init_states()
    qkv = model.layers["0"].attention.qkv_linear
    assert qkv.q_up.weight.std().item() == pytest.approx(
        GELU_GAIN / math.sqrt(1024), rel=0.02
    )
    assert qkv.q_down.weight.std().item() == pytest.approx(
        1 / math.sqrt(2048), rel=0.02
    )


def test_qmlpdeep_matrices_use_muon():
    config = build_model_config("d200m", arch="qmlpdeep")
    optimizer = glimmer_muon_optimizer(
        config, lr=4e-4, parallelism=ParallelismConfig()
    )
    layouts = optimizer.optimizers[0].compute_sharding_by_fqn
    with torch.device("meta"):
        model = config.build()
    for name, parameter in model.named_parameters():
        if not name.startswith("layers.") or parameter.ndim != 2:
            continue
        assert re.search(_MUON_PATTERN, name), name
        assert name in layouts, name


@pytest.mark.parametrize(
    "flavor,heads", [("d200m", 8), ("d400m", 12), ("d700m", 16), ("d1b", 20)]
)
def test_head_dim_128_heads_and_gqa(flavor, heads):
    with torch.device("meta"):
        model = build_model_config(flavor, head_dim=128).build()
    attention = model.layers["0"].attention
    assert attention.head_dim == 128
    assert attention.n_heads == heads
    assert attention.n_kv_heads == heads // 4
    assert attention.o_gate.weight.shape[0] == heads * 128
    assert attention.wo.weight.shape[1] == heads * 128


def test_head_dim_128_d200m_matches_head_dim_64_size():
    with torch.device("meta"):
        a = build_model_config("d200m", head_dim=128).build()
        b = build_model_config("d200m").build()
    assert sum(p.numel() for p in a.parameters()) == sum(
        p.numel() for p in b.parameters()
    )


def test_head_dim_128_qmlpdeep_queries_match_heads():
    with torch.device("meta"):
        model = build_model_config(
            "d400m",
            arch="qmlpdeep",
            head_dim=128,
            qmlp_swa_per_full=7,
            qmlp_intermediate=1760,
        ).build()
    assert len(model.layers) == 32
    qkv = model.layers["0"].attention.qkv_linear
    assert qkv.q_up.weight.shape == (1760, 1280)
    assert qkv.q_down.weight.shape == (12 * 128, 1760)
    assert qkv.wkv.weight.shape == (2 * 3 * 128, 1280)


def test_qmlp_linear_residual_starts_as_linear_query():
    torch.manual_seed(0)
    model = build_model_config(
        "d400m",
        arch="qmlpdeep",
        head_dim=128,
        qmlp_swa_per_full=7,
        qmlp_intermediate=1760,
        qmlp_residual=True,
    ).build()
    model.init_states()
    qkv = model.layers["0"].attention.qkv_linear
    assert qkv.wq.weight.shape == (12 * 128, 1280)
    assert torch.count_nonzero(qkv.q_down.weight) == 0
    assert qkv.wq.weight.std().item() == pytest.approx(0.02, rel=0.05)
    x = torch.randn(16, 1280)
    xq, _, _ = qkv(x)
    torch.testing.assert_close(xq.reshape(16, -1), qkv.wq(x))
    with torch.no_grad():
        qkv.q_down.weight.normal_()
    xq, _, _ = qkv(x)
    mlp = qkv.q_down(torch.nn.functional.gelu(qkv.q_up(x)))
    torch.testing.assert_close(xq.reshape(16, -1), qkv.wq(x) + mlp)


def test_qmlp_linear_residual_uses_muon():
    config = build_model_config(
        "d200m", arch="qmlpdeep", head_dim=128, qmlp_residual=True
    )
    optimizer = glimmer_muon_optimizer(
        config, lr=1.4e-3, parallelism=ParallelismConfig()
    )
    layouts = optimizer.optimizers[0].compute_sharding_by_fqn
    with torch.device("meta"):
        model = config.build()
    names = [n for n, p in model.named_parameters() if n.startswith("layers.") and p.ndim == 2]
    assert any(n.endswith("qkv_linear.wq.weight") for n in names)
    for name in names:
        assert re.search(_MUON_PATTERN, name), name
        assert name in layouts, name
