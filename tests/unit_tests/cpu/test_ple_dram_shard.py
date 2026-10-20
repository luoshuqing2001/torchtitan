# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch import nn

from torchtitan.components.optim.utils import (
    get_flat_optim_state_dict,
    load_flat_optim_state_dict,
)
from torchtitan.distributed.utils import clip_grad_norm_
from torchtitan.experiments.ple_dram_shard.model import (
    _OwnedEmbedding,
    optimizer_checkpoint_hooks,
    prefetch_owned,
    shard_table,
)
from torchtitan.experiments.ple_dram_shard.optim import clip_owned


def _worker(rank, path, fused_gather=False, prepared_update=False):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group(
        "gloo", init_method=f"file://{path}/rendezvous", rank=rank, world_size=2
    )
    try:
        initial = torch.randn(256, 64)
        table = nn.Embedding(256, 64)
        table.weight.data.copy_(initial)
        ownership = shard_table(table, dist.group.WORLD)
        full = nn.Parameter(initial.clone())
        reference = torch.optim.AdamW(
            [{"params": [full], "param_names": ["weight"]}], lr=0.003, fused=True
        )
        optimizer = torch.optim.AdamW(
            [{"params": [table.weight], "param_names": ["weight"]}],
            lr=0.003,
            fused=True,
        )
        optimizer_checkpoint_hooks(optimizer, table.weight, ownership)
        update = None
        if prepared_update:
            from torchtitan.experiments.ple_dram_shard.cpu_update import CPUTableUpdate

            update = CPUTableUpdate(table.weight, optimizer, num_threads=2)
        for step in range(8):
            torch.manual_seed(42 + rank * 31 + step)
            count = 0 if step == 0 else 37 + rank
            tokens = torch.randint(256, (count,))
            grad = torch.randn(count, 64)
            full.grad = torch.zeros_like(full)
            full.grad.index_add_(0, tokens, grad)
            dist.all_reduce(full.grad)
            owned_slice = slice(ownership.offset, ownership.offset + ownership.num_rows)
            active = full.grad[owned_slice].ne(0).any(dim=1).nonzero().flatten()
            if update is not None:
                update.prepare(active)
            prefetch = prefetch_owned(
                table.weight,
                tokens,
                ownership,
                torch.float32,
                fused_gather=fused_gather,
            )
            rows, inverse, unique, rank_ids = prefetch.rows(torch.float32)
            output = _OwnedEmbedding.apply(
                table.weight, rows, inverse, unique, rank_ids, ownership
            )
            torch.testing.assert_close(
                output, full.detach().index_select(0, tokens), rtol=0, atol=0
            )
            output.backward(grad)
            owned_slice = slice(ownership.offset, ownership.offset + ownership.num_rows)
            torch.testing.assert_close(
                table.weight.grad, full.grad[owned_slice], rtol=0, atol=0
            )
            expected_norm = clip_grad_norm_(
                [full], 0.7, foreach=True, cpu_norm_chunk_size=256
            )
            if update is not None:
                gradient = table.weight.grad
                gradient._ple_owned_rows = (gradient._version, active)
                gradient._ple_owned_norm_cache = (
                    256,
                    gradient._version,
                    torch.linalg.vector_norm(gradient.view(-1, 256), ord=2, dim=1),
                )
            actual_norm = clip_owned(
                [table.weight],
                table.weight,
                ownership,
                0.7,
                256,
                **({"table_update": update} if update is not None else {}),
            )
            torch.testing.assert_close(actual_norm, expected_norm, rtol=0, atol=0)
            if update is None:
                torch.testing.assert_close(
                    table.weight.grad, full.grad[owned_slice], rtol=0, atol=0
                )
            if update is None:
                optimizer.step()
            else:
                update.step()
            reference.step()
            torch.testing.assert_close(
                table.weight.grad, full.grad[owned_slice], rtol=0, atol=0
            )
            torch.testing.assert_close(table.weight, full[owned_slice], rtol=0, atol=0)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    optimizer.state[table.weight][key],
                    reference.state[full][key][owned_slice],
                    rtol=0,
                    atol=0,
                )
            if step in (3, 5):
                state = {
                    "model": table.state_dict(),
                    "optim": get_flat_optim_state_dict(optimizer),
                }
                source = (
                    state
                    if step == 3
                    else {
                        "model": {"weight": full.detach()},
                        "optim": get_flat_optim_state_dict(reference),
                    }
                )
                location = f"{path}/checkpoint_{step}"
                dcp.save(source, checkpoint_id=location)
                if step == 3:
                    replicated = {
                        "model": {"weight": torch.empty_like(full)},
                        "optim": {
                            key: torch.empty_like(value)
                            if isinstance(value, torch.Tensor)
                            else value
                            for key, value in get_flat_optim_state_dict(
                                reference
                            ).items()
                        },
                    }
                    dcp.load(replicated, checkpoint_id=location)
                    torch.testing.assert_close(
                        replicated["model"]["weight"], full, rtol=0, atol=0
                    )
                    for key in ("exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(
                            replicated["optim"][f"state.weight.{key}"],
                            reference.state[full][key],
                            rtol=0,
                            atol=0,
                        )

                with torch.no_grad():
                    table.weight.zero_()
                    for value in optimizer.state[table.weight].values():
                        value.zero_()
                dcp.load(state, checkpoint_id=location)
                table.load_state_dict(state["model"])
                load_flat_optim_state_dict(optimizer, state["optim"])
                torch.testing.assert_close(
                    table.weight, full[owned_slice], rtol=0, atol=0
                )
                for key in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        optimizer.state[table.weight][key],
                        reference.state[full][key][owned_slice],
                        rtol=0,
                        atol=0,
                    )
                torch.testing.assert_close(
                    optimizer.state[table.weight]["step"],
                    reference.state[full]["step"],
                    rtol=0,
                    atol=0,
                )
            optimizer.zero_grad(set_to_none=True)
    finally:
        dist.destroy_process_group()


def _cuda_worker(rank, path, fused_gather=False):
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard

    from torchtitan.models.glimmer_dense import ple_dram as dram

    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group(
        "nccl", init_method=f"file://{path}/rendezvous", rank=rank, world_size=2
    )

    group = dist.new_group(backend="gloo")
    mesh = DeviceMesh.from_group(dist.group.WORLD, "cuda", mesh_dim_names=("dp_shard",))
    initial = torch.randn(256, 64)
    full = nn.Parameter(initial.clone())
    table = nn.Embedding(256, 64)
    table.weight.data = initial.clone().pin_memory()
    ownership = shard_table(table, group)
    ref = torch.optim.AdamW(
        [{"params": [full], "param_names": ["weight"]}], lr=0.003, fused=True
    )
    opt = torch.optim.AdamW(
        [{"params": [table.weight], "param_names": ["weight"]}], lr=0.003, fused=True
    )
    optimizer_checkpoint_hooks(opt, table.weight, ownership)
    for step in range(12):
        torch.manual_seed(42 + rank * 31 + step)
        count = 0 if step == 0 else 257 + rank
        tokens = torch.randint(256, (count,), device="cuda")
        grad = torch.randn(count, 64, device="cuda", dtype=torch.bfloat16)
        if step == 2:
            grad.fill_(1e-40)
        if step == 3:
            grad.fill_((-1.0) ** rank * 1e10)
        if step == 4:
            grad *= torch.pow(2.0, torch.randint(-8, 9, grad.shape, device="cuda"))
        unique, inverse = torch.unique(tokens, return_inverse=True)
        full.grad = dram._embedding_grad(
            unique,
            dram._embedding_row_grad(grad.float(), inverse, unique.numel()),
            256,
            torch.float32,
        )
        prefetch = prefetch_owned(
            table.weight, tokens, ownership, torch.bfloat16, fused_gather=fused_gather
        )
        rows, inverse, unique, rank_ids = prefetch.rows(torch.bfloat16)
        consumer = torch.cuda.Stream()
        with torch.cuda.stream(consumer):
            cached_rows, cached_inverse, cached_unique, _ = prefetch.rows(
                torch.bfloat16
            )
            copied_rows = cached_rows.clone()
            copied_inverse = cached_inverse.clone()
            copied_unique = cached_unique.clone()
        torch.cuda.current_stream().wait_stream(consumer)
        torch.testing.assert_close(copied_rows, rows, rtol=0, atol=0)
        torch.testing.assert_close(copied_inverse, inverse, rtol=0, atol=0)
        torch.testing.assert_close(copied_unique, unique, rtol=0, atol=0)
        output = _OwnedEmbedding.apply(
            table.weight, rows, inverse, unique, rank_ids, ownership
        )
        torch.testing.assert_close(
            output,
            full.detach().cuda().bfloat16().index_select(0, tokens),
            rtol=0,
            atol=0,
        )
        output.backward(grad)
        own = slice(ownership.offset, ownership.offset + ownership.num_rows)
        torch.testing.assert_close(table.weight.grad, full.grad[own], rtol=0, atol=0)
        local = torch.randn(64, 16, device="cuda")
        backbone = nn.Parameter(DTensor.from_local(local, mesh, (Shard(0),)))
        reference_backbone = nn.Parameter(
            DTensor.from_local(local.clone(), mesh, (Shard(0),))
        )
        backbone.grad = DTensor.from_local(local.clone(), mesh, (Shard(0),))
        reference_backbone.grad = DTensor.from_local(local.clone(), mesh, (Shard(0),))
        expected_norm = clip_grad_norm_(
            [full, reference_backbone], 0.7, foreach=True, cpu_norm_chunk_size=256
        )
        actual_norm = clip_owned(
            [table.weight, backbone], table.weight, ownership, 0.7, 256
        )
        torch.testing.assert_close(actual_norm, expected_norm, rtol=0, atol=0)
        torch.testing.assert_close(table.weight.grad, full.grad[own], rtol=0, atol=0)
        torch.testing.assert_close(
            backbone.grad.to_local(), reference_backbone.grad.to_local(), rtol=0, atol=0
        )
        ref.step()
        opt.step()
        torch.testing.assert_close(table.weight, full[own], rtol=0, atol=0)
        for key in ["exp_avg", "exp_avg_sq"]:
            torch.testing.assert_close(
                opt.state[table.weight][key], ref.state[full][key][own], rtol=0, atol=0
            )
        if step == 6:
            state = {
                "model": table.state_dict(),
                "optim": get_flat_optim_state_dict(opt),
                "backbone": backbone.detach(),
            }
            dcp.save(state, checkpoint_id=f"{path}/mixed_checkpoint")
            with torch.no_grad():
                table.weight.zero_()
                backbone.to_local().zero_()
                for value in opt.state[table.weight].values():
                    value.zero_()
            dcp.load(state, checkpoint_id=f"{path}/mixed_checkpoint")
            table.load_state_dict(state["model"])
            load_flat_optim_state_dict(opt, state["optim"])
            torch.testing.assert_close(table.weight, full[own], rtol=0, atol=0)
            torch.testing.assert_close(
                backbone.to_local(), reference_backbone.to_local(), rtol=0, atol=0
            )
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    opt.state[table.weight][key],
                    ref.state[full][key][own],
                    rtol=0,
                    atol=0,
                )
        opt.zero_grad(set_to_none=True)
    dist.destroy_process_group()


class TestRowShardedPLE(unittest.TestCase):
    def test_requires_fp32_master(self):
        with self.assertRaisesRegex(ValueError, "FP32 CPU master"):
            shard_table(nn.Embedding(8, 8, dtype=torch.bfloat16), None)

    def test_cpu_training_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as path:
            mp.spawn(_worker, args=(path,), nprocs=2, join=True)

    @unittest.skipUnless(
        torch.backends.cpu.get_cpu_capability() == "AVX512", "AVX512 required"
    )
    def test_prepared_cpu_training_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as path:
            mp.spawn(_worker, args=(path, False, True), nprocs=2, join=True)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Two CUDA devices required")
    def test_cuda_gradients_and_mixed_norm(self):
        with tempfile.TemporaryDirectory() as path:
            mp.spawn(_cuda_worker, args=(path,), nprocs=2, join=True)
