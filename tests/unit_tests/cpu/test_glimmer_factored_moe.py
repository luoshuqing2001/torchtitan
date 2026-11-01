# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.glimmer_dense import build_model_config
from torchtitan.models.glimmer_dense.factored_moe import (
    FactoredMoE,
    register_factored_moe_balancing_hook,
)
from torchtitan_recipes.models.glimmer_muon import (
    _MUON_PATTERN,
    glimmer_muon_optimizer,
)


def _layer(split: bool, **kwargs) -> FactoredMoE:
    torch.manual_seed(0)
    kwargs = {"dim": 16, "hidden_dim": 8, "num_experts": 6, "top_k": 2, **kwargs}
    layer = FactoredMoE(FactoredMoE.Config(split=split, **kwargs))
    for parameter in layer.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    # Nonzero biases check that they change only the choice, not the weights.
    torch.nn.init.normal_(layer.expert_bias, std=0.1)
    if split:
        torch.nn.init.normal_(layer.gate_expert_bias, std=0.1)
    return layer


def _route(x, weight, bias, top_k):
    scores = (x @ weight.T).softmax(-1)
    ids = torch.topk(scores + bias, top_k, dim=-1).indices
    weights = scores.gather(-1, ids)
    return weights / weights.sum(-1, keepdim=True), ids


def test_standard_matches_per_expert_sum():
    layer = _layer(split=False)
    x = torch.randn(2, 5, 16)
    flat = x.reshape(-1, 16)
    weights, ids = _route(flat, layer.router_weight, layer.expert_bias, 2)
    expected = torch.zeros_like(flat)
    for t in range(flat.shape[0]):
        for k in range(2):
            e = ids[t, k]
            hidden = F.silu(layer.w_gate[e] @ flat[t]) * (layer.w_up[e] @ flat[t])
            expected[t] += weights[t, k] * (layer.w_down[e] @ hidden)
    torch.testing.assert_close(layer(x), expected.reshape(x.shape), rtol=1e-5, atol=1e-5)


def test_split_matches_sum_over_composite_experts():
    layer = _layer(split=True, num_gate_experts=5, gate_top_k=3)
    x = torch.randn(2, 5, 16)
    flat = x.reshape(-1, 16)
    s, ud = _route(flat, layer.router_weight, layer.expert_bias, 2)
    t, gate = _route(flat, layer.gate_router_weight, layer.gate_expert_bias, 3)
    expected = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for a in range(2):
            for b in range(3):
                i, j = ud[token, a], gate[token, b]
                hidden = F.silu(layer.w_gate[j] @ flat[token]) * (
                    layer.w_up[i] @ flat[token]
                )
                expected[token] += s[token, a] * t[token, b] * (layer.w_down[i] @ hidden)
    torch.testing.assert_close(layer(x), expected.reshape(x.shape), rtol=1e-5, atol=1e-5)


def _swiglu(x, gate, up, down):
    return down @ (F.silu(gate @ x) * (up @ x))


def test_rank_pairing_matches_rank_by_rank_sum():
    layer = _layer(split=True, pairing="rank", num_gate_experts=5, top_k=3, gate_top_k=3)
    x = torch.randn(9, 16)
    s, ud = _route(x, layer.router_weight, layer.expert_bias, 3)
    t, gate = _route(x, layer.gate_router_weight, layer.gate_expert_bias, 3)
    expected = torch.zeros_like(x)
    for token in range(x.shape[0]):
        s_rank = s[token].argsort(descending=True)
        t_rank = t[token].argsort(descending=True)
        weights = s[token, s_rank] * t[token, t_rank]
        weights = weights / weights.sum()
        for r in range(3):
            i, j = ud[token, s_rank[r]], gate[token, t_rank[r]]
            expected[token] += weights[r] * _swiglu(
                x[token], layer.w_gate[j], layer.w_up[i], layer.w_down[i]
            )
    torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-5)


def test_shared_experts_add_unweighted_swiglus():
    routed = _layer(split=False)
    shared = _layer(split=False, num_shared_experts=2)
    with torch.no_grad():
        for name in ("router_weight", "w_gate", "w_up", "w_down", "expert_bias"):
            getattr(shared, name).copy_(getattr(routed, name))
    x = torch.randn(7, 16)
    expected = routed(x) + sum(
        torch.stack(
            [
                _swiglu(row, shared.shared_w_gate[e], shared.shared_w_up[e], shared.shared_w_down[e])
                for row in x
            ]
        )
        for e in range(2)
    )
    torch.testing.assert_close(shared(x), expected, rtol=1e-5, atol=1e-5)


def test_rank_pairing_needs_equal_top_k():
    with pytest.raises(ValueError):
        FactoredMoE.Config(
            dim=16, hidden_dim=8, num_experts=6, top_k=2, split=True,
            gate_top_k=3, pairing="rank",
        )


def test_split_gradients_reach_both_routers():
    layer = _layer(split=True)
    layer(torch.randn(7, 16)).square().sum().backward()
    for name in ("router_weight", "gate_router_weight", "w_gate", "w_up", "w_down"):
        assert getattr(layer, name).grad.abs().sum() > 0, name


def test_balancing_hook_moves_bias_toward_cold_experts():
    layer = _layer(split=True)
    layer.expert_bias.zero_()
    layer.gate_expert_bias.zero_()
    layer.train()
    layer(torch.randn(64, 16))
    assert layer.tokens_per_expert.sum() == 64 * 2
    assert layer.tokens_per_gate_expert.sum() == 64 * 2
    hooks = []
    optimizers = SimpleNamespace(register_step_pre_hook=hooks.append)
    context = SimpleNamespace(get_optional_mesh=lambda name: None)
    register_factored_moe_balancing_hook(optimizers, [layer], context)
    load = layer.tokens_per_expert.clone()
    hooks[0]()
    hot, cold = load.argmax(), load.argmin()
    assert layer.expert_bias[hot] < layer.expert_bias[cold]
    assert layer.tokens_per_expert.sum() == 0
    assert layer.tokens_per_gate_expert.sum() == 0


def _meta_model(arch: str, **kwargs):
    with torch.device("meta"):
        return build_model_config(
            "d200m", arch=arch, head_dim=128, swa_rotary_dim=32, **kwargs
        ).build()


def test_split_and_standard_match_in_parameters_and_active_flops():
    dense, moe, split = (_meta_model(arch) for arch in ("ffn", "moe", "moe_split"))
    count = lambda model: sum(p.numel() for p in model.parameters())
    ffn = 3 * 1024 * 3072 * 12
    routers_moe = 12 * 32 * 1024
    assert count(moe) == count(dense) + 3 * ffn + routers_moe
    assert count(split) == count(moe) + routers_moe
    seq = 2048
    _, dense_flops = dense.config.get_nparams_and_flops(dense, seq)
    _, moe_flops = moe.config.get_nparams_and_flops(moe, seq)
    _, split_flops = split.config.get_nparams_and_flops(split, seq)
    assert moe_flops == dense_flops + 6 * routers_moe
    assert split_flops == moe_flops + 6 * routers_moe


def test_qwen3_pair_parameters_and_flops():
    """128 experts: top 6 + 2 shared vs rank-paired top 4 / top 4 + 2 shared."""
    moe = _meta_model("moe", moe_experts=128, moe_top_k=6, moe_shared=2)
    split = _meta_model(
        "moe_split", moe_experts=128, moe_top_k=4, moe_gate_experts=128,
        moe_gate_top_k=4, moe_pairing="rank", moe_shared=2,
    )
    count = lambda model: sum(p.numel() for p in model.parameters())
    router = 12 * 128 * 1024
    assert count(split) == count(moe) + router
    unit = 12 * 384 * 1024  # one expert matrix in every layer
    _, moe_flops = moe.config.get_nparams_and_flops(moe, 2048)
    _, split_flops = split.config.get_nparams_and_flops(split, 2048)
    # moe: 3 * (6 + 2) expert matrices; split: 4 + 2 * 4 routed + 3 * 2 shared.
    assert moe_flops - split_flops == 6 * ((24 - 18) * unit - router)


@pytest.mark.parametrize("arch", ["moe", "moe_split"])
def test_muon_covers_every_routed_matrix(arch):
    config = build_model_config(
        "d200m", arch=arch, head_dim=128, swa_rotary_dim=32, moe_shared=2
    )
    optim = glimmer_muon_optimizer(
        config, lr=1e-3, parallelism=ParallelismConfig(tensor_parallel_degree=1)
    )
    muon = optim.optimizers[0]
    with torch.device("meta"):
        model = config.build()
    pattern = re.compile(_MUON_PATTERN)
    for name, parameter in model.named_parameters():
        if ".feed_forward." in name:
            assert pattern.search(name), name
            assert name in muon.compute_sharding_by_fqn, name
    routed = [
        fqn
        for bucket in muon.bucket_configs
        if bucket.name.endswith("routed-experts")
        for fqn in bucket.patterns
    ]
    assert len(routed) == 12 * 3
