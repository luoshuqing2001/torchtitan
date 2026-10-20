# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.experiments.engram_optim.model import FusedEngramTable
from torchtitan.models.glimmer_dense.engram import EngramTable, _segmented_short_conv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [16, 48, 1024])
@pytest.mark.parametrize("remainder", [0, 3])
def test_fused_lookup_matches_native(dtype, dim, remainder):
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    common = {
        "dim": dim,
        "rows_per_head": 31,
        "eos_token_id": 2,
        "remainder": remainder,
    }
    reference = EngramTable.Config(**common).build().to(device="cuda", dtype=dtype)
    candidate = (
        FusedEngramTable.Config(**common, fused_lookup=True)
        .build()
        .to(device="cuda", dtype=dtype)
    )
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_()
    candidate.load_state_dict(reference.state_dict())
    tokens = torch.randint(-10, 100, (37,), device="cuda")
    tokens[5] = 2
    tokens[12] = torch.iinfo(torch.int64).max
    positions = torch.arange(37, device="cuda") % 7
    incoming = torch.randn(37, dim, device="cuda", dtype=dtype)
    outputs = []
    for model in (reference, candidate):
        output = model(tokens, positions)
        output.backward(incoming)
        outputs.append(output)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for first, second in zip(
        reference.parameters(), candidate.parameters(), strict=True
    ):
        torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(
            reference(tokens[::2], positions[::2]),
            candidate(tokens[::2], positions[::2]),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 3, 13, 1025])
@pytest.mark.parametrize(
    "input_grad,weight_grad", [(True, True), (True, False), (False, True)]
)
def test_segmented_conv_matches_native(dtype, tokens, input_grad, weight_grad):
    from torchtitan.experiments.engram_optim.kernels import segmented_conv

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    values = torch.randn(tokens, 64, device="cuda", dtype=dtype)
    weights = torch.randn(4, 64, device="cuda", dtype=dtype)
    segments = torch.arange(tokens, device="cuda") // 7
    incoming = torch.randn_like(values)
    outputs, gradients = [], []
    for op in (_segmented_short_conv, segmented_conv):
        inputs = values.clone().requires_grad_(input_grad)
        weight = weights.clone().requires_grad_(weight_grad)
        result = op(inputs, weight, segments, dilation=3)
        result.backward(incoming)
        outputs.append(result)
        gradients.append((inputs.grad, weight.grad))
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    torch.testing.assert_close(*gradients, rtol=0, atol=0)


def test_conv_higher_order_native_fallback():
    from torchtitan.experiments.engram_optim.kernels import segmented_conv

    torch.manual_seed(42)
    values = torch.randn(13, 16, device="cuda", requires_grad=True)
    weight = torch.randn(4, 16, device="cuda", requires_grad=True)
    segments = torch.arange(13, device="cuda") // 7
    results = []
    for op in (_segmented_short_conv, segmented_conv):
        output = op(values, weight, segments, dilation=3)
        first = torch.autograd.grad(
            output.square().sum(), (values, weight), create_graph=True
        )
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first), (values, weight)
        )
        results.append((first, second))
    torch.testing.assert_close(*results, rtol=2e-6, atol=2e-6)


def test_conv_detects_aliased_mutation():
    from torchtitan.experiments.engram_optim.kernels import segmented_conv

    for op in (_segmented_short_conv, segmented_conv):
        values = torch.randn(13, 16, device="cuda", requires_grad=True)
        weight = torch.randn(4, 16, device="cuda", requires_grad=True)
        output = op(
            values,
            weight,
            torch.zeros(13, device="cuda", dtype=torch.int64),
            dilation=3,
        )
        with torch.no_grad():
            values.detach().add_(1)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            output.sum().backward()


@pytest.mark.parametrize("hook_weight", [False, True])
def test_conv_higher_order_hooks_run_once(hook_weight):
    from torchtitan.experiments.engram_optim.kernels import segmented_conv

    for op in (_segmented_short_conv, segmented_conv):
        values = torch.randn(13, 16, device="cuda", requires_grad=True)
        weight = torch.randn(4, 16, device="cuda", requires_grad=True)
        calls = []

        def hook(gradient, calls=calls):
            calls.append(1)
            return gradient

        (weight if hook_weight else values).register_hook(hook)
        output = op(
            values,
            weight,
            torch.zeros(13, device="cuda", dtype=torch.int64),
            dilation=3,
        )
        torch.autograd.grad(output.square().sum(), (values, weight), create_graph=True)
        assert len(calls) == 1
