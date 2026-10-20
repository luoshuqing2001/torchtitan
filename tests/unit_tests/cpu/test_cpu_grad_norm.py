# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.utils import clip_grad_norm_


class TestCPUGradNorm(unittest.TestCase):
    def test_parallel_l2_and_clipping(self):
        torch.manual_seed(42)
        for size in (128, 137):
            with self.subTest(size=size):
                parameter = torch.nn.Parameter(torch.zeros(size))
                gradient = torch.randn_like(parameter)
                parameter.grad = gradient.clone()
                expected = gradient.double().norm().float()
                actual = clip_grad_norm_(
                    [parameter], 0.5, foreach=True, cpu_norm_chunk_size=16
                )
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0)
                expected_grad = gradient * (0.5 / (expected + 1e-6)).clamp(max=1)
                torch.testing.assert_close(
                    parameter.grad, expected_grad, rtol=1e-6, atol=1e-7
                )

    def test_native_and_other_norms_unchanged(self):
        torch.manual_seed(42)
        for norm_type, chunk_size in ((2, 0), (1, 16), (float("inf"), 16)):
            reference = torch.nn.Parameter(torch.zeros(137))
            actual = torch.nn.Parameter(torch.zeros_like(reference))
            reference.grad = torch.randn_like(reference)
            actual.grad = reference.grad.clone()
            expected = torch.nn.utils.clip_grad_norm_(
                [reference], 0.5, norm_type=norm_type, foreach=True
            )
            observed = clip_grad_norm_(
                [actual],
                0.5,
                norm_type=norm_type,
                foreach=True,
                cpu_norm_chunk_size=chunk_size,
            )
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual.grad, reference.grad, rtol=0, atol=0)

    def test_empty_and_noncontiguous(self):
        self.assertEqual(clip_grad_norm_([], 1, cpu_norm_chunk_size=16), 0)
        parameter = torch.nn.Parameter(torch.zeros(13, 17).t())
        parameter.grad = torch.randn(13, 17).t()
        expected = parameter.grad.norm()
        actual = clip_grad_norm_([parameter], 1e6, cpu_norm_chunk_size=16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_negative_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            clip_grad_norm_([], 1, cpu_norm_chunk_size=-1)


if __name__ == "__main__":
    unittest.main()
