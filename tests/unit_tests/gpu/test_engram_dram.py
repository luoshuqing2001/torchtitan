# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.experiments.engram_dram.kernels import scatter_rows


@pytest.fixture(autouse=True)
def strict_training_precision(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    backend = torch.backends.cuda.matmul
    names = (
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_reduced_precision_reduction",
        "allow_tf32",
    )
    previous = {name: getattr(backend, name) for name in names}
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        backend.allow_bf16_reduced_precision_reduction = True
        backend.allow_fp16_reduced_precision_reduction = True
        backend.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        for name, value in previous.items():
            setattr(backend, name, value)
        torch.use_deterministic_algorithms(deterministic)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [16, 64])
def test_unique_scatter_varying_row_counts(dtype, width):
    generator = torch.Generator(device="cuda").manual_seed(42)
    actual = torch.randn(2048, width, device="cuda", dtype=dtype, generator=generator)
    expected = actual.clone()
    for count in (0, 1, 17, 255, 1001):
        indices = torch.randperm(2048, device="cuda", generator=generator)[:count]
        values = torch.randn(
            count, width, device="cuda", dtype=dtype, generator=generator
        )
        scatter_rows(actual, indices, values)
        expected.index_copy_(0, indices, values)
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_value_recomputation_on_copy_stream(dtype):
    from torchtitan.experiments.engram_dram.recompute import ValueRecomputer

    torch.manual_seed(42)
    source = [
        torch.randn(*shape, device="cuda", dtype=dtype)
        for shape in [(257, 64), (192, 64), (64, 192)]
    ]
    incoming = torch.randn_like(source[0])
    results = []
    for enabled in (False, True):
        inputs = [t.clone().requires_grad_(True) for t in source]
        owner = ValueRecomputer()
        if enabled:
            output = owner.project(*inputs)
            owner.prefetch(output)
            assert owner.records[0].ready is not None
        else:
            output = torch.nn.functional.linear(
                torch.nn.functional.linear(inputs[0], inputs[1]), inputs[2]
            )
        output.backward(incoming)
        results.append((output.detach(), [t.grad for t in inputs]))
        assert not owner.records
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("width", [193, 196])
@pytest.mark.parametrize("align_forward", [False, True])
def test_aligned_value_recomputation_matches_native(prefetch, width, align_forward):
    from torchtitan.experiments.engram_dram.recompute import ValueRecomputer

    torch.manual_seed(42)
    source = [
        torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        for shape in [(257, 64), (width, 64), (64, width)]
    ]
    incoming = torch.randn_like(source[0])
    results = []
    for enabled in (False, True):
        inputs = [t.clone().requires_grad_(True) for t in source]
        owner = ValueRecomputer(align=True, align_forward=align_forward)
        if enabled:
            output = owner.project(*inputs)
            if prefetch:
                owner.prefetch(output)
        else:
            output = torch.nn.functional.linear(
                torch.nn.functional.linear(inputs[0], inputs[1]), inputs[2]
            )
        output.backward(incoming)
        results.append((output.detach(), [t.grad for t in inputs]))
        assert not owner.records
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
