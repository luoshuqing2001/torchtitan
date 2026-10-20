# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.fsdp import _ignored_params
from torchtitan.experiments.ple_dram_shard.model import RowShardedGlimmer
from torchtitan.experiments.ple_dram_shard.recipe import glimmer_dense as shard_recipe
from torchtitan.models.glimmer_dense.flavors import build_model_config
from torchtitan.models.glimmer_dense.ple import LayerPLE, PerLayerEmbedder
from torchtitan_recipes.models.glimmer_dense import glimmer_dense


@pytest.mark.parametrize("checkpointed", [False, True])
def test_default_ple_uses_native_autograd(checkpointed):
    torch.manual_seed(42)
    module = PerLayerEmbedder.Config(
        vocab_size=31,
        dim=16,
        num_layers=3,
        ple_dim=8,
        activation_checkpoint=checkpointed,
    ).build()
    layer = LayerPLE.Config(dim=16, ple_dim=8).build()
    for parameter in [*module.parameters(), *layer.parameters()]:
        torch.nn.init.normal_(parameter, std=0.1)
    reference = [
        parameter.detach().clone().requires_grad_()
        for parameter in [*module.parameters(), *layer.parameters()]
    ]
    tokens = torch.tensor([4, 1, 4, 8, 1, 0])
    hidden = torch.randn(6, 16, requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_()
    table, context_weight, context_norm, gate, projection, norm = reference
    context = F.linear(ref_hidden, context_weight) * (16**-0.5)
    context = F.rms_norm(context.unflatten(-1, (3, 8)), (8,), context_norm, 1e-6)
    token_rows = F.embedding(tokens, table)
    token_rows.mul_(8**0.5).add_(context.flatten(-2)).mul_(0.5**0.5)
    ple = token_rows.unflatten(-1, (3, 8))[:, 1]
    gated = F.gelu(F.linear(ref_hidden, gate)) * ple
    expected = ref_hidden + F.rms_norm(F.linear(gated, projection), (16,), norm, 1e-6)
    with (
        patch(
            "torchtitan.models.glimmer_dense.ple.prefetch_rows",
            side_effect=AssertionError("Default path must not prefetch DRAM rows"),
        ),
        patch(
            "torchtitan.models.glimmer_dense.ple.stage",
            side_effect=AssertionError("Default path must not stage weights"),
        ),
    ):
        actual = layer(hidden, module(tokens, hidden)[:, 1])
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=0, atol=0)
    for actual_param, expected_param in zip(
        [*module.parameters(), *layer.parameters()], reference, strict=True
    ):
        torch.testing.assert_close(
            actual_param.grad, expected_param.grad, rtol=0, atol=0
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_model_placement_and_fsdp_exclusion(enabled):
    config = build_model_config("d200m", memory="ple", seq_len=32)
    config.ple_async_training = enabled
    config.set_sharding_(ParallelismConfig())
    assert (config.ple.table.sharding_config is None) == enabled
    assert (config.layers[0].layer_ple.sharding_config is None) == enabled
    with torch.device("meta"):
        model = config.build()
    assert model.ple.config.async_training == enabled
    assert model.ple.context.config.async_training == enabled
    assert all(
        layer.layer_ple.config.async_training == enabled
        for layer in model.layers.values()
    )
    ignored = _ignored_params(model)
    if enabled:
        expected = set(model.ple.parameters())
        for layer in model.layers.values():
            expected.update(layer.layer_ple.parameters())
        assert ignored == expected
    else:
        assert ignored is None


@pytest.mark.parametrize("flag", [None, "0", "1"])
def test_recipe_switch(monkeypatch, flag):
    for name in ["PLE_ASYNC_TRAINING", "PLE_COMPILE_FP8"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("TRAIN_STEPS", "2")
    monkeypatch.setenv("PLE_CPU_FUSED_ADAM", "1")
    if flag is not None:
        monkeypatch.setenv("PLE_ASYNC_TRAINING", flag)
    base = glimmer_dense(seq_len=32)
    config = shard_recipe()
    enabled = flag == "1"
    assert base.model.ple_async_training == enabled
    assert isinstance(config.model, RowShardedGlimmer.Config) == enabled
    assert config.model.ple_async_training == enabled
    assert "float8_quantize" not in config.compile.regions


def test_compile_switch_is_independent(monkeypatch):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("TRAIN_STEPS", "2")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "0")
    monkeypatch.setenv("PLE_COMPILE_FP8", "1")
    config = shard_recipe()
    assert not config.model.ple_async_training
    assert not isinstance(config.model, RowShardedGlimmer.Config)
    assert "float8_quantize" in config.compile.regions


def test_invalid_switch(monkeypatch):
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "yes")
    with pytest.raises(ValueError, match="PLE_ASYNC_TRAINING must be 0 or 1"):
        glimmer_dense(seq_len=32)
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    monkeypatch.setenv("PLE", "0")
    with pytest.raises(ValueError, match="requires PLE=ple"):
        glimmer_dense(seq_len=32)


@pytest.mark.parametrize("flag", [None, "0", "1"])
def test_input_gradient_switch(monkeypatch, flag):
    monkeypatch.delenv("PLE_UNBIND_INPUTS", raising=False)
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "0")
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("TRAIN_STEPS", "2")
    if flag is not None:
        monkeypatch.setenv("PLE_UNBIND_INPUTS", flag)
    assert glimmer_dense(seq_len=32).model.ple_unbind_inputs == (flag == "1")
    assert shard_recipe().model.ple_unbind_inputs == (flag == "1")


def test_invalid_input_gradient_switch(monkeypatch):
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "0")
    monkeypatch.setenv("PLE_UNBIND_INPUTS", "yes")
    with pytest.raises(ValueError, match="PLE_UNBIND_INPUTS must be 0 or 1"):
        glimmer_dense(seq_len=32)
    monkeypatch.setenv("PLE_UNBIND_INPUTS", "1")
    monkeypatch.setenv("PLE", "0")
    with pytest.raises(ValueError, match="requires PLE=ple"):
        glimmer_dense(seq_len=32)


def test_context_norm_reuse_requires_private_checkpoint_path(monkeypatch):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    monkeypatch.setenv("PLE_REUSE_CONTEXT_NORM_INPUT", "1")
    monkeypatch.setenv("PLE_CONTEXT_CHECKPOINT", "0")
    with pytest.raises(ValueError, match="requires context checkpointing"):
        shard_recipe()
    for flag in (
        "PLE_SAVE_MATMULS",
        "PLE_OVERLAP_REPLICATED_GRADIENT",
        "PLE_CONTEXT_CHECKPOINT",
    ):
        monkeypatch.setenv(flag, "1")
    config = shard_recipe()
    assert config.model.ple.context.reuse_norm_input


def test_context_norm_reuse_rejects_invalid_switch(monkeypatch):
    monkeypatch.setenv("PLE_REUSE_CONTEXT_NORM_INPUT", "yes")
    with pytest.raises(ValueError, match="PLE_REUSE_CONTEXT_NORM_INPUT must be 0 or 1"):
        shard_recipe()


def test_saved_input_staging_dependencies_and_config(monkeypatch):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    monkeypatch.setenv("PLE_OFFLOAD_SAVED_INPUTS", "1")
    with pytest.raises(ValueError, match="requires saved matmuls and early fetch"):
        shard_recipe()
    monkeypatch.setenv("PLE_SAVE_MATMULS", "1")
    monkeypatch.setenv("PLE_EARLY_FETCH", "1")
    monkeypatch.setenv("PLE_COMPACT_SAVED_INPUTS", "1")
    config = shard_recipe()
    assert config.model.ple.offload_saved_inputs
    assert config.model.ple.compact_saved_inputs
    assert not config.model.ple.audit_compact_inputs


def test_compact_inputs_require_staging(monkeypatch):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    monkeypatch.setenv("PLE_COMPACT_SAVED_INPUTS", "1")
    with pytest.raises(ValueError, match="require saved input staging"):
        shard_recipe()


def test_saved_input_staging_rejects_multiple_microbatches():
    from types import SimpleNamespace
    from torchtitan.experiments.ple_dram_shard.prefetch import EarlyFetchTrainer

    trainer = object.__new__(EarlyFetchTrainer)
    trainer.engine = SimpleNamespace(
        model_parts=[SimpleNamespace(ple=SimpleNamespace(saved_input_staging=object()))]
    )
    trainer.gradient_accumulation_steps = 2
    with pytest.raises(ValueError, match="one microbatch per step"):
        trainer.train_step(iter(()))
