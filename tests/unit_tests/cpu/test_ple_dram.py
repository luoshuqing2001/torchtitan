# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from types import MethodType

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torchtitan.models.glimmer_dense.ple import PerLayerEmbedder
from torchtitan.models.glimmer_dense.ple_dram import (
    _embedding_grad,
    _embedding_row_grad,
    dram_embedding,
    dram_linear,
    dram_rms_norm,
    prefetch_rows,
    preserve_ple_optimizer_implementation,
)


def _distributed_embedding_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        weight = torch.randn(31, 12, requires_grad=True)
        tokens = torch.tensor([rank, rank, 8, rank + 1])
        dram_embedding(weight, prefetch_rows(weight, tokens)).sum().backward()
        expected = torch.zeros_like(weight)
        expected.index_add_(0, tokens, torch.ones(4, 12))
        dist.all_reduce(expected)
        torch.testing.assert_close(weight.grad, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def _distributed_cuda_embedding_worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        for step in range(10):
            count = 0 if step == 0 else (rank * 7 + step * 3) % 23
            ids = (torch.arange(count, device="cuda") + rank * 7) % 31
            gradient = torch.full((count, 17), rank + 0.125, device="cuda")
            expected = torch.zeros(31, 17, device="cuda")
            expected.index_add_(0, ids, gradient)
            dist.all_reduce(expected)
            actual = _embedding_grad(ids, gradient, 31, torch.float32)
            torch.testing.assert_close(actual, expected.cpu(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class TestDramEmbedding(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_column_chunks_preserve_reduction(self):
        torch.manual_seed(42)
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for width in (1025, 2048):
                inverse = torch.randint(0, 31, (257,), device="cuda")
                flat = torch.randn(257, width, device="cuda", dtype=torch.bfloat16)
                exponents = torch.randint(-16, 17, flat.shape, device="cuda")
                flat = flat * torch.pow(2.0, exponents).bfloat16()
                expected = torch.zeros(31, width, device="cuda")
                expected.index_add_(0, inverse, flat.float())
                actual = _embedding_row_grad(flat, inverse, 31)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            torch.use_deterministic_algorithms(previous_determinism)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Two CUDA devices required")
    def test_distributed_cuda_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _distributed_cuda_embedding_worker,
                args=(f"{directory}/rendezvous",),
                nprocs=2,
                join=True,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_prefetch_dtype_after_parameter_update(self):
        torch.manual_seed(42)
        weight = torch.randn(31, 12, pin_memory=True, requires_grad=True)
        tokens = torch.tensor([4, 1, 4, 8, 1, 0], device="cuda")
        for _ in range(3):
            prefetch = prefetch_rows(weight, tokens, dtype=torch.bfloat16)
            fetched, _, _ = prefetch.rows()
            self.assertEqual(fetched.dtype, torch.bfloat16)
            actual = dram_embedding(weight, prefetch, dtype=torch.bfloat16)
            expected = F.embedding(tokens, weight.detach().cuda().bfloat16())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.float().sum().backward()
            with torch.no_grad():
                weight.add_(weight.grad, alpha=-0.01)
            weight.grad = None

    def check_embedding(self, device):
        torch.manual_seed(42)
        weight = torch.randn(31, 12, requires_grad=True)
        reference = weight.detach().clone().requires_grad_()
        tokens = torch.tensor([4, 1, 4, 8, 1, 0], device=device)
        prefetch = prefetch_rows(weight, tokens)
        actual = dram_embedding(weight, prefetch)
        expected = F.embedding(tokens, reference.to(device))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(weight.grad, reference.grad, rtol=0, atol=0)
        optimizer = torch.optim.AdamW([weight], lr=0.01)
        reference_optimizer = torch.optim.AdamW([reference], lr=0.01)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(weight, reference, rtol=0, atol=0)

    def test_disjoint_rank_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _distributed_embedding_worker,
                args=(f"{directory}/rendezvous",),
                nprocs=2,
                join=True,
            )

    def test_cpu(self):
        self.check_embedding(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda(self):
        self.check_embedding(torch.device("cuda"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_prefetch_consumed_on_multiple_streams(self):
        weight = torch.randn(31, 12, pin_memory=True)
        tokens = torch.tensor([2, 2, 8, 1], device="cuda")
        prefetch = prefetch_rows(weight, tokens)
        results = []
        for _ in range(3):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                results.append((dram_embedding(weight, prefetch).clone(), stream))
        for result, stream in results:
            stream.synchronize()
            torch.testing.assert_close(result.cpu(), weight[tokens.cpu()])


class TestDramLinear(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_native_mixed_precision_backward(self):
        torch.manual_seed(42)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                weight = torch.randn(17, 13, pin_memory=True, requires_grad=True)
                inputs = torch.randn(37, 13, device="cuda", dtype=dtype)
                inputs.requires_grad_()
                reference_weight = weight.detach().to("cuda", dtype).requires_grad_()
                reference_input = inputs.detach().clone().requires_grad_()
                expected = F.linear(reference_input, reference_weight)
                actual = dram_linear(weight, inputs, mixed_precision_backward=True)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                gradient = torch.randn_like(actual)
                actual.backward(gradient)
                expected.backward(gradient)
                torch.testing.assert_close(
                    weight.grad, reference_weight.grad.float().cpu(), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    inputs.grad, reference_input.grad, rtol=0, atol=0
                )


class TestCheckpointedPLE(unittest.TestCase):
    def check_checkpoint(self, device):
        torch.manual_seed(42)
        reference = PerLayerEmbedder.Config(
            vocab_size=31,
            async_training=True,
            dim=16,
            num_layers=3,
            ple_dim=8,
            activation_checkpoint=False,
        ).build()
        actual = PerLayerEmbedder.Config(
            vocab_size=31,
            async_training=True,
            dim=16,
            num_layers=3,
            ple_dim=8,
            activation_checkpoint=True,
        ).build()
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.normal_(std=0.1)
        actual.load_state_dict(reference.state_dict())
        tokens = torch.tensor([4, 1, 4, 8, 1, 0], device=device)
        expected_hidden = torch.randn(6, 16, device=device, requires_grad=True)
        actual_hidden = expected_hidden.detach().clone().requires_grad_()
        expected = reference(tokens, expected_hidden)
        result = actual(tokens, actual_hidden)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        gradient = torch.randn_like(result)
        expected.backward(gradient)
        result.backward(gradient)
        torch.testing.assert_close(
            actual_hidden.grad, expected_hidden.grad, rtol=0, atol=0
        )
        for actual_parameter, expected_parameter in zip(
            actual.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(
                actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0
            )

    def test_cpu(self):
        self.check_checkpoint(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda(self):
        self.check_checkpoint(torch.device("cuda"))


class TestPLEBufferReuse(unittest.TestCase):
    def check_combine(self, device, dtype):
        def reference_combine(module, hidden, prefetch):
            context = module.context(hidden)
            tokens = dram_embedding(module.table.weight, prefetch, dtype=hidden.dtype)
            tokens = tokens * (module.ple_dim**0.5)
            mixed = (tokens + context) * (0.5**0.5)
            return mixed.unflatten(-1, (module.num_layers, module.ple_dim))

        previous_determinism = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for checkpointed in (False, True):
                torch.manual_seed(42)
                config = PerLayerEmbedder.Config(
                    vocab_size=31,
                    async_training=True,
                    dim=16,
                    num_layers=3,
                    ple_dim=8,
                    activation_checkpoint=checkpointed,
                )
                reference = config.build()
                actual = config.build()
                with torch.no_grad():
                    for parameter in reference.parameters():
                        parameter.normal_(std=0.1)
                actual.load_state_dict(reference.state_dict())
                reference._combine = MethodType(reference_combine, reference)
                tokens = torch.tensor([4, 1, 4, 8, 1, 0], device=device)
                expected_hidden = torch.randn(
                    6, 16, device=device, dtype=dtype, requires_grad=True
                )
                actual_hidden = expected_hidden.detach().clone().requires_grad_()
                expected = reference(tokens, expected_hidden)
                observed = actual(tokens, actual_hidden)
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                gradient = torch.randn_like(observed)
                expected.backward(gradient)
                observed.backward(gradient)
                torch.testing.assert_close(
                    actual_hidden.grad, expected_hidden.grad, rtol=0, atol=0
                )
                for param, ref in zip(
                    actual.parameters(), reference.parameters(), strict=True
                ):
                    torch.testing.assert_close(param.grad, ref.grad, rtol=0, atol=0)
        finally:
            torch.use_deterministic_algorithms(previous_determinism)

    def test_cpu(self):
        self.check_combine("cpu", torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda(self):
        self.check_combine("cuda", torch.float32)
        self.check_combine("cuda", torch.bfloat16)


class TestPLEOptimizerCheckpoint(unittest.TestCase):
    def test_load_foreach_state_into_split_fused_optimizer(self):
        from torchtitan.components.optim import AdamW, OptimizersContainer

        model = torch.nn.Module()
        model.ple = torch.nn.Linear(4, 4)
        model.dense = torch.nn.Linear(4, 4)
        original = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", foreach=True, fused=False)]
        ).build(model_parts=[model])
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        original.step()
        saved = original.state_dict()
        restored = OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(pattern=r"^ple\.", foreach=False, fused=True),
                AdamW.Config(pattern=r".*", foreach=True, fused=False),
            ]
        ).build(model_parts=[model])
        preserve_ple_optimizer_implementation(restored, [model])
        restored.load_state_dict(saved)
        self.assertTrue(restored.optimizers[0].param_groups[0]["fused"])
        self.assertFalse(restored.optimizers[0].param_groups[0]["foreach"])
        self.assertFalse(restored.optimizers[1].param_groups[0]["fused"])
        for optimizer in restored:
            for group in optimizer.param_groups:
                for parameter, name in zip(
                    group["params"], group["param_names"], strict=True
                ):
                    for key in ("step", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(
                            optimizer.state[parameter][key],
                            saved[f"state.{name}.{key}"],
                            rtol=0,
                            atol=0,
                        )
        restored.step()
        self.assertTrue(
            all(torch.isfinite(parameter).all() for parameter in model.parameters())
        )


class TestDramRMSNorm(unittest.TestCase):
    def check_norm(self, device, dtype):
        torch.manual_seed(42)
        for shape in ((61, 64), (19, 12, 64)):
            weight = torch.randn(shape[-1], requires_grad=True)
            reference_weight = weight.detach().clone().requires_grad_()
            inputs = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
            reference_inputs = inputs.detach().clone().requires_grad_()
            expected = F.rms_norm(
                reference_inputs,
                shape[-1:],
                reference_weight.to(device=device, dtype=dtype),
                1e-6,
            )
            actual = dram_rms_norm(weight, inputs, 1e-6)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            gradient = torch.randn_like(actual)
            expected.backward(gradient)
            actual.backward(gradient)
            torch.testing.assert_close(
                inputs.grad, reference_inputs.grad, rtol=0, atol=0
            )
            torch.testing.assert_close(
                weight.grad, reference_weight.grad, rtol=0, atol=0
            )

    def test_cpu(self):
        self.check_norm(torch.device("cpu"), torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                self.check_norm(torch.device("cuda"), dtype)


if __name__ == "__main__":
    unittest.main()
