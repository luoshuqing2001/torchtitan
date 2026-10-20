# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.experiments.ple_dram_shard.prefetch import (
    EarlyFetchGlimmer,
    EarlyFetchPLE,
    EarlyFetchTrainer,
    _PrefetchBatch,
)
from torchtitan.experiments.ple_dram_shard.recipe import glimmer_dense

FLAGS = (
    "PLE_GPU_NORM",
    "PLE_RELEASE_FETCH_CACHE",
    "PLE_OWNER_ROUTING",
    "PLE_DIRECT_GRADIENT",
    "PLE_FUSED_TABLE_UPDATE",
    "PLE_OVERLAP_REPLICATED_GRADIENT",
    "PLE_EARLY_FETCH",
)


@pytest.mark.parametrize("value", [None, "0", "1", "26"])
def test_cpu_update_thread_configuration(monkeypatch, value):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    monkeypatch.setenv("PLE_OVERLAP_CPU_OPTIMIZER", "1")
    monkeypatch.setenv("PLE_FUSED_TABLE_UPDATE", "1")
    monkeypatch.delenv("PLE_CPU_UPDATE_THREADS", raising=False)
    if value is not None:
        monkeypatch.setenv("PLE_CPU_UPDATE_THREADS", value)
    assert glimmer_dense().optim.cpu_update_threads == int(value or "0")


@pytest.mark.parametrize("value", ["-1", "yes", "1.5"])
def test_invalid_cpu_update_thread_configuration(monkeypatch, value):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_CPU_UPDATE_THREADS", value)
    with pytest.raises(ValueError, match="PLE_CPU_UPDATE_THREADS"):
        glimmer_dense()


def test_cpu_update_threads_require_fused_update(monkeypatch):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_CPU_UPDATE_THREADS", "26")
    monkeypatch.setenv("PLE_FUSED_TABLE_UPDATE", "0")
    with pytest.raises(ValueError, match="requires PLE_FUSED_TABLE_UPDATE"):
        glimmer_dense()


def test_zero_saved_layers_retains_gradient_overlap(monkeypatch):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    monkeypatch.setenv("PLE_SAVE_MATMULS", "1")
    monkeypatch.setenv("PLE_SAVE_LAYERS", "0")
    monkeypatch.setenv("PLE_OVERLAP_REPLICATED_GRADIENT", "1")
    config = glimmer_dense()
    assert config.activation_checkpoint.num_layers_to_save == 0
    assert config.model.layers[0].layer_ple.overlap_replicated_gradient


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("value", [None, "0", "1"])
def test_optional_transport_switches(monkeypatch, flag, value):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("FLAVOR", "d200m")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    monkeypatch.setenv("PLE_DEFER_TABLE_GRADIENT", "1")
    monkeypatch.setenv("PLE_OVERLAP_CPU_OPTIMIZER", "1")
    monkeypatch.setenv("PLE_SAVE_MATMULS", "1")
    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)
    if flag == "PLE_RELEASE_FETCH_CACHE":
        monkeypatch.setenv("PLE_EARLY_FETCH", "1")
    if flag == "PLE_GPU_NORM":
        monkeypatch.setenv("PLE_DIRECT_GRADIENT", "1")
    if value is not None:
        monkeypatch.setenv(flag, value)
    config = glimmer_dense()
    actual = {
        "PLE_GPU_NORM": config.model.ple.gpu_norm,
        "PLE_RELEASE_FETCH_CACHE": getattr(config, "release_fetch_cache", False),
        "PLE_OWNER_ROUTING": config.model.ple.owner_routing,
        "PLE_DIRECT_GRADIENT": config.model.ple.direct_gradient_copy,
        "PLE_FUSED_TABLE_UPDATE": config.optim.fused_cpu_update,
        "PLE_OVERLAP_REPLICATED_GRADIENT": config.model.layers[
            0
        ].layer_ple.overlap_replicated_gradient,
        "PLE_EARLY_FETCH": isinstance(config, EarlyFetchTrainer.Config),
    }
    assert actual[flag] is (value == "1")
    if flag == "PLE_EARLY_FETCH" and value == "1":
        assert isinstance(config.model, EarlyFetchGlimmer.Config)
        assert isinstance(config.model.ple, EarlyFetchPLE.Config)


@pytest.mark.parametrize("flag", FLAGS)
def test_transport_rejects_invalid_flag(monkeypatch, flag):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv(flag, "yes")
    with pytest.raises(ValueError, match=flag):
        glimmer_dense()


@pytest.mark.parametrize("release_cache", [False, True])
def test_prefetch_batch_preserves_data_and_request(release_cache):
    batch = TokenizedTrainingMicrobatch(
        input=torch.tensor([[1, 2, 3]]),
        labels=torch.tensor([[2, 3, 4]]),
        positions=torch.arange(3).unsqueeze(0),
        padding_mask=torch.zeros(1, 3, dtype=torch.bool),
        num_valid_tokens=3,
        model_kwargs={"extra": 7},
    )
    wrapped = _PrefetchBatch(batch, release_cache=release_cache)
    request = object()
    wrapped.request = request
    assert wrapped.num_valid_tokens == batch.num_valid_tokens
    assert wrapped.labels is batch.labels
    assert wrapped.input is batch.input
    assert wrapped.loss_kwargs() == batch.loss_kwargs()
    # CPU .to() aliases its input, just as the native microbatch method does.
    result = wrapped.to_input_dict("cpu")
    assert result["input"]._ple_prefetch is request
    assert wrapped.request is (None if release_cache else request)
    assert result["extra"] == 7


@pytest.mark.parametrize("flag", FLAGS)
def test_transports_require_async(monkeypatch, flag):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "0")
    monkeypatch.setenv(flag, "1")
    if flag == "PLE_OVERLAP_REPLICATED_GRADIENT":
        monkeypatch.setenv("PLE_SAVE_MATMULS", "1")
    with pytest.raises(ValueError, match="requires PLE_ASYNC_TRAINING"):
        glimmer_dense()


@pytest.mark.parametrize(
    ("flag", "dependency"),
    [
        ("PLE_DIRECT_GRADIENT", "PLE_DEFER_TABLE_GRADIENT"),
        ("PLE_FUSED_TABLE_UPDATE", "PLE_OVERLAP_CPU_OPTIMIZER"),
        ("PLE_OVERLAP_REPLICATED_GRADIENT", "PLE_SAVE_MATMULS"),
        ("PLE_RELEASE_FETCH_CACHE", "PLE_EARLY_FETCH"),
        ("PLE_GPU_NORM", "PLE_DIRECT_GRADIENT"),
    ],
)
def test_transport_dependencies(monkeypatch, flag, dependency):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    monkeypatch.setenv(flag, "1")
    monkeypatch.setenv(dependency, "0")
    with pytest.raises(ValueError, match=f"requires {dependency}"):
        glimmer_dense()


@pytest.mark.parametrize("asynchronous", ["0", "1"])
def test_replicated_backbone_recipe(monkeypatch, asynchronous):
    from torchtitan.experiments.ple_dram_shard.recipe import glimmer_dense_dp

    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", asynchronous)
    config = glimmer_dense_dp()
    assert config.parallelism.data_parallel_replicate_degree == 4
    assert config.parallelism.data_parallel_shard_degree == 1
    assert config.model.ple_async_training is (asynchronous == "1")

    if asynchronous == "0":
        assert config.optim.optimizer.optimizers[-2].fused is True
        assert config.optim.optimizer.optimizers[-2].foreach is False
