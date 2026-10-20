# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch import nn
from torch.utils.checkpoint import checkpoint

from torchtitan.components.optim.utils import (
    get_flat_optim_state_dict,
    load_flat_optim_state_dict,
)
from torchtitan.experiments.ple_dram_shard.embedding import FusedOwnedEmbedding
from torchtitan.experiments.ple_dram_shard.gradient_transport import (
    OwnedGradientTransport,
)
from torchtitan.experiments.ple_dram_shard.model import (
    _OwnedEmbedding,
    optimizer_checkpoint_hooks,
    prefetch_owned,
    shard_table,
)
from torchtitan.experiments.ple_dram_shard.optim import clip_owned
from torchtitan.experiments.ple_dram_shard.recipe import glimmer_dense
from torchtitan.models.glimmer_dense import ple_dram as dram


class _FailBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, context):
        return context.clone()

    @staticmethod
    def backward(ctx, gradient):
        raise RuntimeError("owned gradient failure")


def _check_gradient_api_and_failure(tables, owners):
    torch.manual_seed(123)
    tokens = torch.randint(256, (33,), device="cuda")
    context = torch.randn(33, 2048, device="cuda", dtype=torch.bfloat16)
    gradients = []
    for index in range(2):
        leaf = context.clone().requires_grad_()
        prefetch = prefetch_owned(
            tables[index].weight, tokens, owners[index], torch.bfloat16
        )
        rows, inverse, unique, rank_ids = prefetch.rows(torch.bfloat16)
        if index == 0:
            output = _OwnedEmbedding.apply(
                tables[index].weight, rows, inverse, unique, rank_ids, owners[index]
            )
            output.mul_(32.0)
            output.add_(leaf)
            output.mul_(0.5**0.5)
        else:
            output = FusedOwnedEmbedding.apply(
                tables[index].weight,
                rows,
                inverse,
                unique,
                rank_ids,
                owners[index],
                leaf,
                32.0,
            )
        gradients.append(
            torch.autograd.grad(
                output.float().square().sum(), (tables[index].weight, leaf)
            )
        )
        assert tables[index].weight.grad is None
    for first, second in zip(*gradients, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not owners[1].gradient_transport.pending
    leaf = context.clone().requires_grad_()
    broken_context = _FailBackward.apply(leaf)
    prefetch = prefetch_owned(tables[1].weight, tokens, owners[1], torch.bfloat16)
    rows, inverse, unique, rank_ids = prefetch.rows(torch.bfloat16)
    output = FusedOwnedEmbedding.apply(
        tables[1].weight,
        rows,
        inverse,
        unique,
        rank_ids,
        owners[1],
        broken_context,
        32.0,
    )
    with pytest.raises(RuntimeError, match="owned gradient failure"):
        output.sum().backward()
    assert tables[1].weight.grad is None
    assert owners[1].gradient_transport.pending


def _fused_worker(
    rank, path, checkpointed, deferred=False, direct=False, routed=False, gpu_norm=False
):
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group(
        "nccl", init_method=f"file://{path}/rendezvous", rank=rank, world_size=2
    )
    try:
        group = dist.new_group(backend="gloo")
        initial = torch.randn(256, 2048)
        tables, owners, optimizers = [], [], []
        for index in range(2):
            table = nn.Embedding(256, 2048)
            table.weight.data = initial.clone().pin_memory()
            owner = shard_table(table, group)
            owner.owner_routing = routed and index == 1
            if deferred and index == 1:
                owner.gradient_transport = OwnedGradientTransport(
                    table.weight, owner.offset, direct_copy=direct, gpu_norm=gpu_norm
                )
                owner.gradient_transport.norm_chunk_size = 1024
            optimizer = torch.optim.AdamW(
                [{"params": [table.weight], "param_names": ["weight"]}],
                lr=0.003,
                fused=True,
            )
            optimizer_checkpoint_hooks(optimizer, table.weight, owner)
            tables.append(table)
            owners.append(owner)
            optimizers.append(optimizer)
        if deferred:
            _check_gradient_api_and_failure(tables, owners)
        for step in range(8):
            for micro in range(1 if step % 2 == 0 else 2):
                torch.manual_seed(100 + rank * 31 + step * 2 + micro)
                count = 0 if step == 0 and micro == 0 else 257 + step
                tokens_T = torch.randint(256, (count,), device="cuda")
                tokens_T[:100] = 0
                context_TD = torch.randn(
                    count, 2048, device="cuda", dtype=torch.bfloat16
                )
                incoming_TD = torch.randn_like(context_TD)
                if step == 2:
                    incoming_TD.fill_(1e-40)
                if step == 3:
                    incoming_TD.fill_((-1.0) ** rank * 1e10)
                if step == 4:
                    incoming_TD *= torch.pow(
                        2.0, torch.randint(-8, 9, incoming_TD.shape, device="cuda")
                    )
                scale = 32.0 if step % 2 == 0 else 512.0**0.5
                contexts = [context_TD.clone().requires_grad_() for _ in range(2)]
                outputs = []
                for index in range(2):
                    prefetch = prefetch_owned(
                        tables[index].weight,
                        tokens_T,
                        owners[index],
                        torch.bfloat16,
                        fused_gather=True,
                    )

                    def compute(context, index=index, prefetch=prefetch, scale=scale):
                        rows, inverse, unique, rank_ids = prefetch.rows(torch.bfloat16)
                        if index == 0:
                            result = _OwnedEmbedding.apply(
                                tables[index].weight,
                                rows,
                                inverse,
                                unique,
                                rank_ids,
                                owners[index],
                            )
                            result.mul_(scale)
                            result.add_(context)
                            result.mul_(0.5**0.5)
                            return result
                        return FusedOwnedEmbedding.apply(
                            tables[index].weight,
                            rows,
                            inverse,
                            unique,
                            rank_ids,
                            owners[index],
                            context,
                            scale,
                        )

                    result = (
                        checkpoint(compute, contexts[index], use_reentrant=False)
                        if checkpointed
                        else compute(contexts[index])
                    )
                    outputs.append(result)
                    calls = []
                    handle = None
                    if deferred and step == 1 and micro == 0:

                        def hook(gradient, calls=calls):
                            calls.append(1)
                            return gradient * 1.25

                        handle = tables[index].weight.register_hook(hook)
                    result.backward(incoming_TD)
                    if handle is not None:
                        handle.remove()
                        assert calls == [1]
                    if owners[index].gradient_transport is not None:
                        assert not owners[index].gradient_transport.pending
                        if step % 2 == 0:
                            cache = tables[index].weight.grad._ple_owned_norm_cache
                            assert cache[:2] == (
                                1024,
                                tables[index].weight.grad._version,
                            )
                assert torch.equal(
                    outputs[0].view(torch.int16), outputs[1].view(torch.int16)
                )
                torch.testing.assert_close(
                    contexts[0].grad, contexts[1].grad, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    tables[0].weight.grad, tables[1].weight.grad, rtol=0, atol=0
                )
            if deferred and step == 6:
                for table in tables:
                    table.weight.grad.mul_(0.75)
            norms = [
                clip_owned([table.weight], table.weight, owner, 0.7, 1024)
                for table, owner in zip(tables, owners, strict=True)
            ]
            torch.testing.assert_close(*norms, rtol=0, atol=0)
            if deferred and step == 4:
                second_norms = [
                    clip_owned([table.weight], table.weight, owner, 0.7, 1024)
                    for table, owner in zip(tables, owners, strict=True)
                ]
                torch.testing.assert_close(*second_norms, rtol=0, atol=0)
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 0.003 / (step + 1)
                optimizer.step()
            torch.testing.assert_close(
                tables[0].weight, tables[1].weight, rtol=0, atol=0
            )
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(
                    optimizers[0].state[tables[0].weight][key],
                    optimizers[1].state[tables[1].weight][key],
                    rtol=0,
                    atol=0,
                )
            if step == 3:
                state = {
                    "model": tables[1].state_dict(),
                    "optim": get_flat_optim_state_dict(optimizers[1]),
                }
                location = f"{path}/checkpoint"
                dcp.save(state, checkpoint_id=location)
                with torch.no_grad():
                    tables[1].weight.zero_()
                    for value in optimizers[1].state[tables[1].weight].values():
                        value.zero_()
                dcp.load(state, checkpoint_id=location)
                tables[1].load_state_dict(state["model"])
                load_flat_optim_state_dict(optimizers[1], state["optim"])
                torch.testing.assert_close(
                    tables[0].weight, tables[1].weight, rtol=0, atol=0
                )
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_fused_training_and_checkpoint(checkpointed, deferred):
    with tempfile.TemporaryDirectory() as path:
        mp.spawn(
            _fused_worker, args=(path, checkpointed, deferred), nprocs=2, join=True
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [512, 1025, 2051])
def test_row_gradient_rounding_and_strides(width):
    from torchtitan.experiments.ple_dram_shard.kernels import row_gradient

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    for count in (0, 1, 1031):
        gradient_TD = torch.randn(
            count, width * 2, device="cuda", dtype=torch.bfloat16
        )[:, ::2]
        inverse_T = torch.randint(257, (count,), device="cuda")
        for scale in (32.0, 512.0**0.5):
            expected_UD = dram._embedding_row_grad(
                (gradient_TD * (0.5**0.5)) * scale, inverse_T, 257
            )
            actual_UD = row_gradient(
                gradient_TD,
                inverse_T,
                257,
                token_scale=scale,
                mix_scale=0.5**0.5,
            )
            torch.testing.assert_close(actual_UD, expected_UD, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_combine_large_element_offsets():
    from torchtitan.experiments.ple_dram_shard.kernels import combine

    if torch.cuda.get_device_properties(0).total_memory < 24 * 1024**3:
        pytest.skip("At least 24 GiB required for the 64-bit element-offset case")
    torch.manual_seed(42)
    width, count = 12288, 175000
    rows_UD = torch.randn(256, width, device="cuda", dtype=torch.bfloat16)
    inverse_T = torch.randint(256, (count,), device="cuda")
    context_TD = torch.randn(count, width, device="cuda", dtype=torch.bfloat16)
    assert context_TD.numel() > 2**31
    expected_TD = rows_UD.index_select(0, inverse_T)
    expected_TD.mul_(32.0)
    expected_TD.add_(context_TD)
    expected_TD.mul_(0.5**0.5)
    actual_TD = combine(
        rows_UD, inverse_T, context_TD, token_scale=32.0, mix_scale=0.5**0.5
    )
    assert torch.equal(expected_TD.view(torch.int16), actual_TD.view(torch.int16))


@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_fused_recipe_switch(monkeypatch, enabled):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    if enabled is None:
        monkeypatch.delenv("PLE_FUSED_EMBEDDING", raising=False)
    else:
        monkeypatch.setenv("PLE_FUSED_EMBEDDING", enabled)
    assert glimmer_dense().model.ple.fused_embedding is (enabled == "1")


def test_fused_recipe_rejects_invalid_flag(monkeypatch):
    monkeypatch.setenv("PLE_FUSED_EMBEDDING", "yes")
    with pytest.raises(ValueError, match="PLE_FUSED_EMBEDDING must be"):
        glimmer_dense()


def test_fused_recipe_requires_async(monkeypatch):
    monkeypatch.setenv("PLE_FUSED_EMBEDDING", "1")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "0")
    with pytest.raises(ValueError, match="requires PLE_ASYNC_TRAINING"):
        glimmer_dense()


@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_deferred_recipe_switch(monkeypatch, enabled):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    if enabled is None:
        monkeypatch.delenv("PLE_DEFER_TABLE_GRADIENT", raising=False)
    else:
        monkeypatch.setenv("PLE_DEFER_TABLE_GRADIENT", enabled)
    assert glimmer_dense().model.ple.defer_table_gradient is (enabled == "1")


def test_deferred_recipe_rejects_invalid_flag(monkeypatch):
    monkeypatch.setenv("PLE_DEFER_TABLE_GRADIENT", "yes")
    with pytest.raises(ValueError, match="PLE_DEFER_TABLE_GRADIENT must be"):
        glimmer_dense()


def test_deferred_recipe_requires_async(monkeypatch):
    monkeypatch.setenv("PLE_DEFER_TABLE_GRADIENT", "1")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "0")
    with pytest.raises(ValueError, match="requires PLE_ASYNC_TRAINING"):
        glimmer_dense()


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_direct_gradient_training_and_checkpoint(checkpointed):
    if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
        pytest.skip("CUDA 13 required")
    with tempfile.TemporaryDirectory() as path:
        mp.spawn(
            _fused_worker, args=(path, checkpointed, True, True), nprocs=2, join=True
        )


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_owner_routing_training_and_checkpoint(checkpointed, direct):
    if direct and (
        torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13
    ):
        pytest.skip("CUDA 13 required")
    with tempfile.TemporaryDirectory() as path:
        mp.spawn(
            _fused_worker,
            args=(path, checkpointed, True, direct, True),
            nprocs=2,
            join=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_owner_merge_large_element_offsets():
    from torchtitan.experiments.ple_dram_shard.kernels import merge_owned

    width = 12288
    rows = (2**31) // width + 1
    required = rows * width * 4
    if torch.cuda.mem_get_info()[0] < required + 2 * 1024**3:
        pytest.skip("Insufficient free memory for large offset check")
    values = torch.arange(width, device="cuda", dtype=torch.float32).unsqueeze(0)
    mapping = torch.zeros((1, rows), device="cuda", dtype=torch.int64)
    output = merge_owned(values, mapping)
    for row in (0, rows // 2, rows - 1):
        torch.testing.assert_close(output[row], values[0], rtol=0, atol=0)


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_owner_merge_rank_order(world):
    from torchtitan.experiments.ple_dram_shard.kernels import merge_owned

    values = torch.randn(world + 1, 33, device="cuda", dtype=torch.float32)
    values[-1].zero_()
    values[:-1, 0] = -0.0
    values[:-1, 1] = 1e-40
    if world > 1:
        values[0, 2] = 1e10
        values[1, 2] = -1e10
    if world > 2:
        values[2, 2] = 1.0
    mapping = (
        torch.arange(world, device="cuda").unsqueeze(1).expand(world, 7).contiguous()
    )
    expected = torch.zeros(7, 33, device="cuda")
    for rank in range(world):
        expected.add_(values.index_select(0, mapping[rank]))
    actual = merge_owned(values, mapping)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
@pytest.mark.parametrize("checkpointed", [False, True])
def test_gpu_norm_preserves_training(tmp_path, checkpointed):
    mp.spawn(
        _fused_worker,
        args=(str(tmp_path), checkpointed, True, True, True, True),
        nprocs=2,
        join=True,
    )
