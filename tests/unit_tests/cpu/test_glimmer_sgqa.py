# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import QKVLinear
from torchtitan.models.glimmer_dense import build_model_config
from torchtitan.models.glimmer_dense.sgqa import PartialRoPE, SGQAProjection
from torchtitan_recipes.models.glimmer_muon import _MUON_PATTERN, glimmer_muon_optimizer

DENSE = dict(head_dim=128, swa_rotary_dim=32)
QMLP = dict(
    arch="qmlpdeep",
    head_dim=128,
    swa_rotary_dim=32,
    qmlp_swa_per_full=5,
    qmlp_intermediate=2752,
    qmlp_residual=True,
    qmlp_zero_init="linear",
)


def test_partial_rope_rotates_only_last_dims():
    rope = PartialRoPE.Config(dim=32, max_context_length=64, theta=500_000.0).build()
    rope.init_states()
    q, k = torch.randn(8, 4, 128), torch.randn(8, 2, 128)
    rq, rk = rope(q, k, torch.arange(8))
    torch.testing.assert_close(rq[..., :96], q[..., :96])
    torch.testing.assert_close(rk[..., :96], k[..., :96])
    assert not torch.allclose(rq[1:, :, 96:], q[1:, :, 96:])
    torch.testing.assert_close(rq.norm(dim=-1), q.norm(dim=-1))


@pytest.mark.parametrize("kwargs", [DENSE, QMLP], ids=["dense", "qmlp"])
def test_swa_layers_use_sgqa_and_full_layers_stay_fused(kwargs):
    with torch.device("meta"):
        model = build_model_config("d200m", **kwargs).build()
    for layer in model.layers.values():
        attention = layer.attention
        if attention.window_size is not None:
            assert isinstance(attention.qkv_linear, SGQAProjection)
            assert isinstance(attention.rope, PartialRoPE)
            assert attention.rope.config.dim == 32
            assert attention.qkv_linear.wk.weight.shape == (2 * 96, 1024)
            assert attention.qkv_linear.wk_rope.weight.shape == (32, 1024)
            assert attention.qkv_linear.wv.weight.shape == (2 * 128, 1024)
        else:
            assert isinstance(attention.qkv_linear, QKVLinear)
            assert attention.rope is None


def test_shared_rotary_key_across_kv_groups():
    torch.manual_seed(0)
    model = build_model_config("d200m", **DENSE).build()
    model.init_states()
    proj = model.layers["0"].attention.qkv_linear
    xq, xk, xv = proj(torch.randn(16, 1024))
    assert xq.shape == (16, 8, 128) and xk.shape == (16, 2, 128) and xv.shape == (16, 2, 128)
    torch.testing.assert_close(xk[:, 0, 96:], xk[:, 1, 96:])
    assert not torch.allclose(xk[:, 0, :96], xk[:, 1, :96])


def test_qmlp_sgqa_init():
    torch.manual_seed(0)
    model = build_model_config("d200m", **QMLP).build()
    model.init_states()
    proj = model.layers["0"].attention.qkv_linear
    assert torch.count_nonzero(proj.wq.weight) == 0
    assert proj.q_up.weight.shape == (2752, 1024)
    assert proj.q_down.weight.std().item() == pytest.approx(2752**-0.5, rel=0.02)


@pytest.mark.parametrize("kwargs", [DENSE, QMLP], ids=["dense", "qmlp"])
def test_sgqa_matrices_use_muon(kwargs):
    config = build_model_config("d200m", **kwargs)
    optimizer = glimmer_muon_optimizer(config, lr=1.4e-3, parallelism=ParallelismConfig())
    layouts = optimizer.optimizers[0].compute_sharding_by_fqn
    with torch.device("meta"):
        model = config.build()
    names = [n for n, p in model.named_parameters() if n.startswith("layers.") and p.ndim == 2]
    assert any(n.endswith("wk_rope.weight") for n in names)
    for name in names:
        assert re.search(_MUON_PATTERN, name), name
        assert name in layouts, name


@pytest.mark.parametrize("kwargs", [DENSE, QMLP], ids=["dense", "qmlp"])
def test_forward_runs(kwargs):
    torch.manual_seed(0)
    model = build_model_config("d200m", seq_len=256, **kwargs).build()
    model.init_states()
    tokens = torch.randint(0, 1000, (256,))
    positions = torch.arange(256)
    masks = model.get_attention_masks(positions)
    with torch.no_grad():
        logits = model(tokens, positions, masks)
    assert logits.shape[-1] == 100352 and torch.isfinite(logits).all()
