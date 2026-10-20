# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchtitan.models.glimmer_dense.model import GlimmerDenseModel
from torchtitan.models.glimmer_dense.ple import LayerPLE, PerLayerEmbedder


@pytest.fixture(autouse=True)
def _deterministic_math():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


class _RoutingBlock(nn.Module):
    def __init__(self, asynchronous):
        super().__init__()
        self.linear = nn.Linear(16, 16)
        self.ple = LayerPLE.Config(
            dim=16, ple_dim=8, async_training=asynchronous
        ).build()

    def forward(self, hidden, masks, positions, *, padding_mask, per_layer_input):
        return self.ple(self.linear(hidden).tanh(), per_layer_input)


class _RoutingModel(nn.Module):
    """Exercise the real model routing with small recurrent layer consumers."""

    def __init__(self, asynchronous, checkpointed):
        super().__init__()
        self.config = SimpleNamespace(
            ple_async_training=asynchronous, ple_unbind_inputs=False
        )
        self.tok_embeddings = nn.Embedding(31, 16)
        self.ple = PerLayerEmbedder.Config(
            vocab_size=31,
            dim=16,
            num_layers=4,
            ple_dim=8,
            activation_checkpoint=checkpointed,
            async_training=asynchronous,
        ).build()
        self.layers = nn.ModuleDict(
            {str(i): _RoutingBlock(asynchronous) for i in range(4)}
        )
        self.engram = None
        self.norm = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 31)
        self._skip_lm_head = False
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.1)

    forward = GlimmerDenseModel.forward

    def forward_with_independent_slices(self, tokens):
        hidden = self.tok_embeddings(tokens)
        inputs = self.ple(tokens, hidden)
        for index, layer in enumerate(self.layers.values()):
            hidden = layer(
                hidden,
                None,
                None,
                padding_mask=None,
                per_layer_input=inputs[:, index],
            )
        return self.lm_head(self.norm(hidden))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ple_layer_routing_preserves_updates(asynchronous, checkpointed, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(42)
    reference = _RoutingModel(asynchronous, checkpointed).to(device)
    if asynchronous and device == "cuda":
        reference.ple.cpu()
        for layer in reference.layers.values():
            layer.ple.cpu()
    candidate = copy.deepcopy(reference)
    candidate.config.ple_unbind_inputs = True
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=0.003, fused=True)
        for model in (reference, candidate)
    ]
    for _ in range(4):
        tokens = torch.randint(31, (17,), device=device)
        expected = reference.forward_with_independent_slices(tokens)
        actual = candidate(tokens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        upstream = torch.randn_like(expected)
        expected.backward(upstream)
        actual.backward(upstream)
        pairs = list(zip(reference.parameters(), candidate.parameters(), strict=True))
        for expected_param, actual_param in pairs:
            torch.testing.assert_close(
                actual_param.grad, expected_param.grad, rtol=0, atol=0
            )
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for expected_param, actual_param in pairs:
            torch.testing.assert_close(actual_param, expected_param, rtol=0, atol=0)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    optimizers[1].state[actual_param][key],
                    optimizers[0].state[expected_param][key],
                    rtol=0,
                    atol=0,
                )
