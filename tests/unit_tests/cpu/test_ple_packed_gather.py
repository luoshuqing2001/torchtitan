# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import pytest
import torch
import torch.multiprocessing as mp

from tests.unit_tests.cpu.test_ple_dram_shard import _cuda_worker, _worker
from torchtitan.experiments.ple_dram_shard.gather import gather_packed
from torchtitan.experiments.ple_dram_shard.recipe import glimmer_dense


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_gather_packed_exact(dtype):
    torch.manual_seed(42)
    weight_RD = torch.randn(256, 65)
    weight_RD[0, :8] = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), 1e-40, -1e-40, 1.00390625, -1.00390625]
    )
    for indices_U in (
        torch.tensor([], dtype=torch.int64),
        torch.tensor([0]),
        torch.tensor([17, 0, 255, 17, 128]),
        torch.arange(256),
    ):
        actual_UD = gather_packed(weight_RD, indices_U, dtype, pin_memory=False)
        expected_UD = weight_RD.index_select(0, indices_U).to(dtype)
        view_dtype = torch.int32 if dtype == torch.float32 else torch.int16
        torch.testing.assert_close(
            actual_UD.view(view_dtype), expected_UD.view(view_dtype), rtol=0, atol=0
        )


def test_gather_cpu_training_checkpoint():
    with tempfile.TemporaryDirectory() as path:
        mp.spawn(_worker, args=(path, True), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_gather_cuda_training_checkpoint():
    with tempfile.TemporaryDirectory() as path:
        mp.spawn(_cuda_worker, args=(path, True), nprocs=2, join=True)


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_gather_recipe_switch(monkeypatch, enabled):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    monkeypatch.setenv("PLE_FUSED_ROW_GATHER", enabled)
    config = glimmer_dense()
    assert config.model.ple.fused_row_gather is (enabled == "1")


def test_gather_recipe_rejects_invalid_flag(monkeypatch):
    monkeypatch.setenv("PLE_FUSED_ROW_GATHER", "yes")
    with pytest.raises(ValueError, match="PLE_FUSED_ROW_GATHER must be"):
        glimmer_dense()


def test_gather_recipe_requires_async(monkeypatch):
    monkeypatch.setenv("PLE_FUSED_ROW_GATHER", "1")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "0")
    with pytest.raises(ValueError, match="requires PLE_ASYNC_TRAINING"):
        glimmer_dense()


def test_gather_recipe_default(monkeypatch):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    monkeypatch.delenv("PLE_FUSED_ROW_GATHER", raising=False)
    assert glimmer_dense().model.ple.fused_row_gather is False
