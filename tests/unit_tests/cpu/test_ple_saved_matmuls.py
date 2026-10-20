# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
import torch.nn.functional as F

from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.ple_dram_shard.checkpoint import (
    SavedLayerPLE,
    SavePLEMatmuls,
)
from torchtitan.models.glimmer_dense.ple import LayerPLE


@pytest.mark.parametrize("saved_layers", [0, 1])
def test_saved_linear_outputs_preserve_updates(monkeypatch, saved_layers):
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    reference = LayerPLE.Config(dim=16, ple_dim=16, async_training=True).build()
    candidate = SavedLayerPLE.Config(dim=16, ple_dim=16, async_training=True).build()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.02)
    candidate.load_state_dict(reference.state_dict())
    expected = torch.optim.AdamW(reference.parameters(), lr=0.003, fused=True)
    actual = torch.optim.AdamW(candidate.parameters(), lr=0.003, fused=True)
    reference = (
        SelectiveAC.Config(recompute_linear=True, force_recompute_mm_shapes_by_fqns=[])
        .build()
        ._wrap_block(reference, base_fqn="layers.0")
    )
    candidate = (
        SavePLEMatmuls.Config(
            recompute_linear=True,
            force_recompute_mm_shapes_by_fqns=[],
            num_layers_to_save=saved_layers,
        )
        .build()
        ._wrap_block(candidate, base_fqn="layers.0")
    )
    native_linear = F.linear
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return native_linear(*args, **kwargs)

    monkeypatch.setattr(F, "linear", counted)
    for _ in range(8):
        hidden = torch.randn(37, 16)
        ple = torch.randn(37, 16)
        incoming = torch.randn(37, 16)
        gradients = []
        outputs = []
        counts = []
        for module in (reference, candidate):
            inputs = hidden.clone().requires_grad_()
            embeddings = ple.clone().requires_grad_()
            calls.clear()
            result = module(inputs, embeddings)
            result.backward(incoming)
            outputs.append(result.detach())
            gradients.append((inputs.grad, embeddings.grad))
            counts.append(len(calls))
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        torch.testing.assert_close(*gradients, rtol=0, atol=0)
        assert counts[0] == 4
        assert counts[1] == (2 if saved_layers else 4)
        for first, second in zip(
            reference.parameters(), candidate.parameters(), strict=True
        ):
            torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)
        expected.step()
        actual.step()
        torch.testing.assert_close(
            expected.state_dict(), actual.state_dict(), rtol=0, atol=0
        )
        for first, second in zip(
            reference.parameters(), candidate.parameters(), strict=True
        ):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        actual.load_state_dict(copy.deepcopy(expected.state_dict()))
        expected.zero_grad(set_to_none=True)
        actual.zero_grad(set_to_none=True)


@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_saved_matmul_switch_defaults_off(monkeypatch, enabled):
    from torchtitan.experiments.ple_dram_shard.recipe import glimmer_dense

    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    monkeypatch.delenv("PLE_SAVE_MATMULS", raising=False)
    if enabled is not None:
        monkeypatch.setenv("PLE_SAVE_MATMULS", enabled)
    config = glimmer_dense()
    assert isinstance(config.activation_checkpoint, SavePLEMatmuls.Config) == (
        enabled == "1"
    )
    assert isinstance(config.model.layers[0].layer_ple, SavedLayerPLE.Config) == (
        enabled == "1"
    )


def test_saved_matmul_switch_rejects_invalid_configuration(monkeypatch):
    from torchtitan.experiments.ple_dram_shard.recipe import glimmer_dense

    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_SAVE_MATMULS", "yes")
    with pytest.raises(ValueError, match="PLE_SAVE_MATMULS must"):
        glimmer_dense()
    monkeypatch.setenv("PLE_SAVE_MATMULS", "1")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "0")
    with pytest.raises(ValueError, match="requires PLE_ASYNC_TRAINING"):
        glimmer_dense()
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    for count in ("-1", "13"):
        monkeypatch.setenv("PLE_SAVE_LAYERS", count)
        with pytest.raises(ValueError, match="PLE_SAVE_LAYERS must"):
            glimmer_dense()
