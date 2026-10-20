# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from torchtitan.distributed.activation_checkpoint import SelectiveAC


class _CountMatmuls(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.mm.default:
            self.count += 1
        return func(*args, **(kwargs or {}))


class TestSelectiveAC(unittest.TestCase):
    def test_recompute_linear(self):
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16, bias=False),
        )
        inputs = torch.randn(7, 16, requires_grad=True)
        expected = model(inputs)
        expected.square().sum().backward()
        counts = []
        for recompute in (False, True):
            actual = deepcopy(model)
            actual.zero_grad(set_to_none=True)
            checkpointed = SelectiveAC.Config(recompute_linear=recompute).build()
            wrapped = checkpointed._wrap_block(actual)
            observed_inputs = inputs.detach().clone().requires_grad_()
            with _CountMatmuls() as counter:
                observed = wrapped(observed_inputs)
                observed.square().sum().backward()
            counts.append(counter.count)
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                observed_inputs.grad, inputs.grad, rtol=0, atol=0
            )
            for param, ref in zip(actual.parameters(), model.parameters(), strict=True):
                torch.testing.assert_close(param.grad, ref.grad, rtol=0, atol=0)
        self.assertGreater(counts[1], counts[0])


if __name__ == "__main__":
    unittest.main()
