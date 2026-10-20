# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validate device summaries against native CPU chunk norms."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from torchtitan.experiments.ple_dram_shard.gradient_transport import (
    OwnedGradientTransport,
)


def test_device_norm_requires_direct_copy():
    parameter = torch.nn.Parameter(torch.zeros(16, 16))
    with pytest.raises(ValueError, match="require direct gradient copies"):
        OwnedGradientTransport(parameter, 0, gpu_norm=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("chunk_size", [256, 65536])
def test_device_summary_matches_native_cpu(chunk_size):
    from torchtitan.experiments.ple_dram_shard.norm import DeviceChunkNorm

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda", 0)
    cpu_rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(device).clone()
    with ThreadPoolExecutor(max_workers=1) as pool:
        computer = DeviceChunkNorm(device, chunk_size, pool)
        if computer.plan is None:
            pytest.skip("Native CPU norm is not supported by this GPU kernel")
        assert torch.equal(cpu_rng, torch.random.get_rng_state())
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
        for width, num_rows in ((2048, 128), (12288, 256)):
            for fraction in (0.0, 0.01, 0.3, 1.0):
                indices_U = (
                    torch.randperm(num_rows)[: int(num_rows * fraction)].sort().values
                )
                for scale in (1e-22, 1e-10, 1.0, 1e10):
                    values_UD = torch.randn(indices_U.numel(), width) * scale
                    gradient_RD = torch.zeros(num_rows, width)
                    gradient_RD[indices_U] = values_UD
                    expected_C = torch.linalg.vector_norm(
                        gradient_RD.view(-1, chunk_size), ord=2, dim=1
                    )
                    summary = computer.compute(values_UD.cuda(), indices_U, num_rows)
                    summary.ready.synchronize()
                    assert torch.equal(
                        expected_C.view(torch.int32), summary.values.view(torch.int32)
                    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_unmatched_cpu_norm_keeps_cpu_fallback(monkeypatch):
    from torchtitan.experiments.ple_dram_shard.norm import DeviceChunkNorm

    monkeypatch.setattr(DeviceChunkNorm, "_select_plan", lambda self: None)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.warns(UserWarning, match="using CPU summaries"):
            computer = DeviceChunkNorm(torch.device("cuda", 0), 256, pool)
        assert (
            computer.compute(torch.zeros(1, 256, device="cuda"), torch.tensor([0]), 1)
            is None
        )
