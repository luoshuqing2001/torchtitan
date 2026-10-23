# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch

from torchtitan.experiments.engram_optim.model import (
    FusedEngramTable,
    OptimizedEngramUpdate,
)
from torchtitan.experiments.engram_optim.recipe import glimmer_dense
from torchtitan.models.glimmer_dense.engram import NGramEngram

FLAGS = (
    "ENGRAM_FUSED_LOOKUP",
    "ENGRAM_FUSED_CONV",
    "ENGRAM_UPDATE_CHECKPOINT",
    "ENGRAM_COMPILE_FP8",
)


def modules(dtype, checkpoint):
    common = {"dim": 16, "n_layers": 2, "ple_dim": 8, "eos_token_id": 2}
    reference = NGramEngram.Config(**common, rows_per_head=31, remainder=3).build()
    candidate = NGramEngram.Config(
        **common,
        rows_per_head=31,
        remainder=3,
        table=FusedEngramTable.Config(
            dim=16, rows_per_head=31, eos_token_id=2, remainder=3, fused_lookup=True
        ),
        update=OptimizedEngramUpdate.Config(
            **common, fused_conv=True, activation_checkpoint=checkpoint
        ),
    ).build()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.02)
    reference.to(dtype=dtype)
    candidate.to(dtype=dtype)
    candidate.load_state_dict(reference.state_dict())
    return reference, candidate


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_cpu_fallback_preserves_training_and_checkpoint(dtype, checkpoint):
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    reference, candidate = modules(dtype, checkpoint)
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=0.003, fused=True)
        for model in (reference, candidate)
    ]
    for step in range(8):
        tokens = torch.randint(0, 31, (37,))
        tokens[5] = 2
        positions = torch.arange(37) % 7
        inputs = torch.randn(37, 16, dtype=dtype)
        incoming = torch.randn_like(inputs)
        outputs, gradients = [], []
        for model in (reference, candidate):
            hidden = inputs.clone().requires_grad_(step % 2 == 0)
            output = model(hidden, tokens, positions)
            output.backward(incoming)
            outputs.append(output.detach())
            gradients.append(hidden.grad)
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        torch.testing.assert_close(*gradients, rtol=0, atol=0)
        for first, second in zip(
            reference.parameters(), candidate.parameters(), strict=True
        ):
            torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.testing.assert_close(
            optimizers[0].state_dict(), optimizers[1].state_dict(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            reference.state_dict(), candidate.state_dict(), rtol=0, atol=0
        )
        if step == 3:
            candidate.load_state_dict(copy.deepcopy(reference.state_dict()))
            optimizers[1].load_state_dict(copy.deepcopy(optimizers[0].state_dict()))


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("value", [None, "0", "1"])
def test_engram_flags_default_off(monkeypatch, flag, value):
    monkeypatch.setenv("PLE", "engram_legacy")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "0")
    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv(flag, value)
    config = glimmer_dense()
    engram = config.model.engram
    enabled = value == "1"
    assert isinstance(engram.table, FusedEngramTable.Config) is (
        flag == "ENGRAM_FUSED_LOOKUP" and enabled
    )
    assert isinstance(engram.update, OptimizedEngramUpdate.Config) is (
        flag in ("ENGRAM_FUSED_CONV", "ENGRAM_UPDATE_CHECKPOINT") and enabled
    )
    assert ("float8_quantize" in config.compile.regions) is (
        flag == "ENGRAM_COMPILE_FP8" and enabled
    )


@pytest.mark.parametrize("flag", FLAGS)
def test_engram_flags_reject_invalid_values(monkeypatch, flag):
    monkeypatch.setenv("PLE", "engram_legacy")
    monkeypatch.setenv(flag, "yes")
    with pytest.raises(ValueError, match=flag):
        glimmer_dense()


def test_engram_recipe_requires_engram(monkeypatch):
    monkeypatch.setenv("PLE", "ple")
    with pytest.raises(ValueError, match="PLE=engram_legacy"):
        glimmer_dense()


@pytest.mark.parametrize("checkpoint", [False, True])
def test_update_rejects_aliased_input_mutation(checkpoint):
    torch.manual_seed(42)
    _, module = modules(torch.float32, checkpoint)
    hidden = torch.randn(37, 16, requires_grad=True)
    output = module(hidden, torch.arange(37) % 31, torch.arange(37) % 7)
    with torch.no_grad():
        hidden.detach().add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()
