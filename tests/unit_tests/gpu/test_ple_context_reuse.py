# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from torchtitan.experiments.ple_dram_shard.replicated_gradient import (
    ReducedNorm,
    ReusingContextNorm,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(37, 3, 64), (257, 12, 1024), (4097, 12, 1024)])
def test_context_norm_reuse_preserves_all_gradients(dtype, shape):
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    inputs = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], dtype=torch.float32)
    gradient = torch.randn_like(inputs)
    original = inputs.clone()
    results = []
    for reuse in (False, True):
        source = inputs.clone().requires_grad_(True)
        parameter = weight.clone().requires_grad_(True)
        private = source * 0.5
        staged = parameter.detach().to(device="cuda", dtype=dtype)
        transport = SimpleNamespace(gradient=lambda p, grad: grad.float().cpu())
        arguments = (parameter, staged, private, 1e-6, transport)
        output = (
            ReusingContextNorm.apply(*arguments)
            if reuse
            else ReducedNorm.apply(*arguments)
        )
        output.backward(gradient)
        assert torch.equal(source.detach(), original)
        if reuse and shape[-1] == 1024 and torch.cuda.get_device_capability() == (9, 0):
            assert private._version == 1
        results.append((output.detach().clone(), source.grad, parameter.grad))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_retained_context_graph_does_not_clobber_saved_input():
    torch.manual_seed(42)
    source = torch.randn(257, 3, 1024, device="cuda", requires_grad=True)
    private = source * 0.5
    expected = private.detach().clone()
    parameter = torch.ones(1024, requires_grad=True)
    transport = SimpleNamespace(gradient=lambda p, grad: grad.float().cpu())
    output = ReusingContextNorm.apply(
        parameter, parameter.detach().cuda(), private, 1e-6, transport
    )
    gradient = torch.randn_like(output)
    output.backward(gradient, retain_graph=True)
    assert torch.equal(private.detach(), expected)
    assert private._version == 0
    gradients = source.grad.clone(), parameter.grad.clone()
    source.grad = parameter.grad = None
    output.backward(gradient)
    if torch.cuda.get_device_capability() == (9, 0):
        assert private._version == 1
    torch.testing.assert_close(gradients, (source.grad, parameter.grad), rtol=0, atol=0)
