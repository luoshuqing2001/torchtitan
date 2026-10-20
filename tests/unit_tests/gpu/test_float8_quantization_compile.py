# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

pytest.importorskip("torchao")

from torchtitan.distributed.local_compile import LocalCompileConfig  # noqa: E402
from torchtitan.quantization.float8 import Float8Linear  # noqa: E402
from torchtitan.quantization.float8.tensor import _quantize_float8  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="Compiled Float8 validation requires SM90 or later",
)


@pytest.fixture(autouse=True)
def restore_compile_config(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch._dynamo.reset()
    yield
    LocalCompileConfig(regions=[]).apply_local_compile()
    torch._dynamo.reset()
    torch.use_deterministic_algorithms(deterministic)


def test_quantization_bytes_and_scales():
    torch.manual_seed(42)
    inputs = [
        torch.randn(256, 128, device="cuda", dtype=torch.bfloat16),
        torch.zeros(256, 128, device="cuda", dtype=torch.bfloat16),
        torch.tensor(
            [0.0, -0.0, 1e-38, -1e-38, 1e-8, 1.0, -1.0, 10000.0],
            device="cuda",
            dtype=torch.bfloat16,
        ).repeat(256, 16),
    ]
    LocalCompileConfig(regions=[]).apply_local_compile()
    expected = [
        _quantize_float8(value, reduction_axis=axis)
        for value in inputs
        for axis in (None, -1, 0)
    ]
    LocalCompileConfig(regions=["float8_quantize"]).apply_local_compile()
    actual = [
        _quantize_float8(value, reduction_axis=axis)
        for value in inputs
        for axis in (None, -1, 0)
    ]
    for (expected_data, expected_scale), (actual_data, actual_scale) in zip(
        expected, actual, strict=True
    ):
        assert torch.equal(
            expected_data.view(torch.uint8), actual_data.view(torch.uint8)
        )
        assert torch.equal(expected_scale, actual_scale)


@pytest.mark.parametrize("recipe_name", ["rowwise", "rowwise_with_gw_hp"])
def test_compiled_float8_training_is_exact(recipe_name):
    def run(regions):
        LocalCompileConfig(regions=regions).apply_local_compile()
        torch.manual_seed(42)
        linear = (
            Float8Linear.Config(
                in_features=128,
                out_features=128,
                bias=True,
                recipe_name=recipe_name,
            )
            .build()
            .cuda()
            .bfloat16()
        )
        optimizer = torch.optim.AdamW(linear.parameters(), lr=0.003, fused=True)
        history = []
        for step in range(4):
            torch.manual_seed(42 + step)
            inputs = torch.randn(
                64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            upstream = torch.randn_like(inputs)
            output = linear(inputs)
            output.backward(upstream)
            history.extend(
                [output.detach().clone(), inputs.grad.clone()]
                + [parameter.grad.clone() for parameter in linear.parameters()]
            )
            optimizer.step()
            history.extend(
                [linear.weight._tensor.detach().clone(), linear.bias.detach().clone()]
            )
            for state in optimizer.state.values():
                history.extend(value.clone() for value in state.values())
            optimizer.zero_grad(set_to_none=True)
        return history

    expected = run([])
    actual = run(["float8_quantize"])
    for reference, compiled in zip(expected, actual, strict=True):
        torch.testing.assert_close(compiled, reference, rtol=0, atol=0)
