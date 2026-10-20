# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch

from torchtitan.experiments.ple_dram_shard.checkpoint import SavedContext
from torchtitan.experiments.ple_dram_shard.recipe import glimmer_dense


@pytest.mark.parametrize("mixed_precision_backward", [False, True])
@pytest.mark.parametrize("input_requires_grad", [False, True])
def test_context_checkpoint_preserves_updates(
    mixed_precision_backward, input_requires_grad
):
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    reference = SavedContext.Config(
        dim=16,
        num_layers=3,
        ple_dim=16,
        async_training=True,
        mixed_precision_backward=mixed_precision_backward,
    ).build()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.02)
    candidate = copy.deepcopy(reference)
    candidate.config.activation_checkpoint = True
    expected = torch.optim.AdamW(reference.parameters(), lr=0.003, fused=True)
    actual = torch.optim.AdamW(candidate.parameters(), lr=0.003, fused=True)
    for step in range(8):
        inputs = torch.randn(37, 16)
        incoming = torch.randn(37, 48)
        outputs, gradients, shapes = [], [], []
        for model in (reference, candidate):
            saved_shapes = []

            def pack(tensor, saved_shapes=saved_shapes):
                saved_shapes.append(tuple(tensor.shape))
                return tensor

            hidden = inputs.clone().requires_grad_(input_requires_grad)
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                output = model(hidden)
            output.backward(incoming)
            outputs.append(output.detach())
            gradients.append(hidden.grad)
            shapes.append(saved_shapes)
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        if input_requires_grad:
            torch.testing.assert_close(*gradients, rtol=0, atol=0)
        assert (37, 3, 16) in shapes[0]
        assert (37, 3, 16) not in shapes[1]
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
        expected.zero_grad(set_to_none=True)
        actual.zero_grad(set_to_none=True)
        if step == 3:
            candidate.load_state_dict(copy.deepcopy(reference.state_dict()))
            actual.load_state_dict(copy.deepcopy(expected.state_dict()))


@pytest.mark.parametrize("enabled", [False, True])
def test_context_checkpoint_detects_aliased_input_mutation(enabled):
    torch.manual_seed(42)
    model = SavedContext.Config(
        dim=16,
        num_layers=3,
        ple_dim=16,
        async_training=True,
        activation_checkpoint=enabled,
    ).build()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.02)
    hidden = torch.randn(37, 16, requires_grad=True)
    output = model(hidden)
    with torch.no_grad():
        hidden.detach().add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()


@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_context_checkpoint_configuration(monkeypatch, enabled):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    monkeypatch.setenv("PLE_SAVE_MATMULS", "1")
    monkeypatch.setenv("PLE_OVERLAP_REPLICATED_GRADIENT", "1")
    monkeypatch.delenv("PLE_CONTEXT_CHECKPOINT", raising=False)
    if enabled is not None:
        monkeypatch.setenv("PLE_CONTEXT_CHECKPOINT", enabled)
    assert glimmer_dense().model.ple.context.activation_checkpoint is (enabled == "1")


def test_context_checkpoint_requires_saved_context(monkeypatch):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_CONTEXT_CHECKPOINT", "1")
    monkeypatch.setenv("PLE_OVERLAP_REPLICATED_GRADIENT", "0")
    with pytest.raises(ValueError, match="requires PLE_OVERLAP_REPLICATED_GRADIENT"):
        glimmer_dense()


def test_context_checkpoint_rejects_invalid_flag(monkeypatch):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_CONTEXT_CHECKPOINT", "yes")
    with pytest.raises(ValueError, match="PLE_CONTEXT_CHECKPOINT must be"):
        glimmer_dense()
