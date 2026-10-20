# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchtitan.components.optim import Adam, AdamW, OptimizersContainer
from torchtitan.experiments.ple_dram_shard.overlap import (
    CPUOverlapOptimizersContainer,
    _step_cpu_optimizers,
)
from torchtitan.experiments.ple_dram_shard.recipe import glimmer_dense


def _cpu_config(config_type):
    return config_type(
        optimizers=[
            AdamW.Config(pattern="weight", lr=0.003, fused=True, foreach=False),
            AdamW.Config(pattern="bias", lr=0.003, fused=True, foreach=False),
        ]
    )


def test_cpu_updates_hooks_and_state_roundtrip():
    torch.manual_seed(42)
    reference = nn.Linear(16, 8)
    candidate = copy.deepcopy(reference)
    expected = _cpu_config(OptimizersContainer.Config).build(model_parts=[reference])
    actual = _cpu_config(CPUOverlapOptimizersContainer.Config).build(
        model_parts=[candidate]
    )
    main_thread = threading.get_ident()
    calls = []
    actual.register_step_pre_hook(
        lambda *args: calls.append(("pre", threading.get_ident()))
    )
    actual.register_step_post_hook(
        lambda *args: calls.append(("post", threading.get_ident()))
    )
    actual.optimizers[-1].register_step_post_hook(
        lambda *args: calls.append(("cpu_done", threading.get_ident()))
    )
    for step in range(8):
        for container in (expected, actual):
            for optimizer in container:
                for group in optimizer.param_groups:
                    group["lr"] = 0.003 / (step + 1)
        for first, second in zip(
            reference.parameters(), candidate.parameters(), strict=True
        ):
            gradient = torch.randn_like(first) if step != 2 else torch.zeros_like(first)
            first.grad = gradient.clone()
            second.grad = gradient.clone()
        expected.step()
        actual.step()
        for first, second in zip(
            reference.parameters(), candidate.parameters(), strict=True
        ):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(
            expected.state_dict(), actual.state_dict(), rtol=0, atol=0
        )
        assert calls[-3][0] == "pre" and calls[-3][1] == main_thread
        assert calls[-2][0] == "cpu_done" and calls[-2][1] != main_thread
        assert calls[-1] == ("post", main_thread)
        if step == 3:
            # State is complete at step return and can move in either direction.
            actual.load_state_dict(copy.deepcopy(expected.state_dict()))
            expected.load_state_dict(copy.deepcopy(actual.state_dict()))
        expected.zero_grad()
        actual.zero_grad()


def test_joins_cpu_update_on_device_error():
    model = nn.Linear(2, 2)
    optimizer = _cpu_config(CPUOverlapOptimizersContainer.Config).build(
        model_parts=[model]
    )
    completed = threading.Event()
    started = threading.Event()
    optimizer.optimizers[0].register_step_pre_hook(lambda *args: started.set())
    optimizer.optimizers[-1].register_step_post_hook(lambda *args: completed.set())
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    def fail():
        assert started.wait(timeout=5)
        raise RuntimeError("device step failed")

    optimizer._device_optimizers = [SimpleNamespace(step=fail)]
    with pytest.raises(RuntimeError, match="device step failed"):
        optimizer.step()
    assert completed.is_set()


def test_readiness_failure_prevents_cpu_update():
    parameter = nn.Parameter(torch.ones(2))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW([parameter], lr=0.1, fused=True)

    def fail():
        raise RuntimeError("gradient readiness failed")

    with pytest.raises(RuntimeError, match="gradient readiness failed"):
        _step_cpu_optimizers([optimizer], SimpleNamespace(synchronize=fail), None)
    torch.testing.assert_close(parameter, torch.ones(2), rtol=0, atol=0)
    assert not optimizer.state


@pytest.mark.parametrize("has_cache", [False, True])
def test_deferred_join_finishes_at_explicit_boundary(has_cache):
    model = nn.Linear(2, 2)
    config = _cpu_config(CPUOverlapOptimizersContainer.Config)
    config.defer_cpu_join = True
    optimizer = config.build(model_parts=[model])
    tail_release = threading.Event()
    entered = threading.Event()
    finished = threading.Event()
    update = SimpleNamespace(
        pending=object(),
        compute_cache=object() if has_cache else None,
        communication_ready=None,
    )

    def step():
        entered.set()
        if update.compute_cache is not None:
            update.communication_ready.set()
        assert tail_release.wait(timeout=5)
        finished.set()

    update.step = step
    optimizer._cpu_optimizers = [SimpleNamespace(_ple_table_update=update)]
    try:
        optimizer.step()
        assert entered.wait(timeout=5)
        assert not finished.is_set()
        assert optimizer._pending_cpu_step is not None
    finally:
        tail_release.set()
        optimizer.finish_cpu_step()
    assert finished.is_set()
    assert optimizer._pending_cpu_step is None


def test_deferred_join_does_not_advance_lr_before_cpu_completion():
    from torchtitan.experiments.ple_dram_shard.optim import RowShardedOptim

    calls = []
    optim = object.__new__(RowShardedOptim)
    optim._pending_scheduler_step = False
    optim.optimizers = SimpleNamespace(
        defer_cpu_join=True, finish_cpu_step=lambda: calls.append("join")
    )
    optim.lr_schedulers = SimpleNamespace(step=lambda: calls.append("schedule"))
    norm = torch.tensor(1.0)
    optim._run_update = lambda loss: calls.append("update") or norm
    assert optim.step(torch.tensor(2.0), current_step=1) is norm
    assert calls == ["join", "update"]
    optim.finish_step()
    assert calls == ["join", "update", "join", "schedule"]
    optim.finish_step()
    assert calls[-1] == "join"


def test_failed_deferred_update_does_not_advance_scheduler_on_close():
    from torchtitan.experiments.ple_dram_shard.optim import RowShardedOptim

    calls = []
    optim = object.__new__(RowShardedOptim)
    optim._pending_scheduler_step = True

    def fail():
        raise RuntimeError("CPU update failed")

    optim.optimizers = SimpleNamespace(finish_cpu_step=fail)
    optim.lr_schedulers = SimpleNamespace(step=lambda: calls.append("schedule"))
    with pytest.raises(RuntimeError, match="CPU update failed"):
        optim.finish_step()
    optim.optimizers.finish_cpu_step = lambda: None
    optim.finish_step()
    assert not calls


def test_rejects_unsupported_configuration():
    with pytest.raises(ValueError, match="native CPU AdamW"):
        CPUOverlapOptimizersContainer.Config(
            optimizers=[Adam.Config(pattern=".*")]
        ).build(model_parts=[nn.Linear(2, 2)])
    model = nn.Linear(2, 2)
    model.bias = nn.Parameter(torch.empty(2, device="meta"))
    with pytest.raises(ValueError, match="separate CPU and device"):
        CPUOverlapOptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=".*", fused=False)]
        ).build(model_parts=[model])
    config = _cpu_config(CPUOverlapOptimizersContainer.Config)
    with pytest.raises(ValueError, match="CUDA graphs"):
        config.build(model_parts=[nn.Linear(2, 2)], enable_cuda_graph=True)
    optimizer = config.build(model_parts=[nn.Linear(2, 2)])
    with pytest.raises(ValueError, match="closures"):
        optimizer.step(lambda: None)


@pytest.mark.parametrize("flag", [None, "0", "1"])
def test_recipe_overlap_switch(monkeypatch, flag):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("TRAIN_STEPS", "2")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "1")
    monkeypatch.delenv("PLE_OVERLAP_CPU_OPTIMIZER", raising=False)
    if flag is not None:
        monkeypatch.setenv("PLE_OVERLAP_CPU_OPTIMIZER", flag)
    config = glimmer_dense()
    assert isinstance(config.optim.optimizer, CPUOverlapOptimizersContainer.Config) == (
        flag == "1"
    )


def test_invalid_recipe_overlap_switch(monkeypatch):
    monkeypatch.setenv("PLE", "ple")
    monkeypatch.setenv("TRAIN_STEPS", "2")
    monkeypatch.setenv("PLE_ASYNC_TRAINING", "0")
    monkeypatch.setenv("PLE_OVERLAP_CPU_OPTIMIZER", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        glimmer_dense()
    monkeypatch.setenv("PLE_OVERLAP_CPU_OPTIMIZER", "1")
    with pytest.raises(ValueError, match="requires PLE_ASYNC_TRAINING=1"):
        glimmer_dense()


def _cuda_worker(rank, world_size, path):
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor
    from torch.utils._pytree import tree_flatten, tree_map

    from torchtitan.components.optim import DistMuon
    from torchtitan.distributed.flex_shard import BucketConfig, ComputeLayout, Owned
    from torchtitan.experiments.ple_dram_shard.model import (
        optimizer_checkpoint_hooks,
        shard_table,
    )
    from torchtitan.experiments.ple_dram_shard.optim import clip_owned

    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{path}/rendezvous",
        rank=rank,
        world_size=world_size,
    )
    try:
        group = dist.new_group(backend="gloo")
        mesh = DeviceMesh.from_group(
            dist.group.WORLD, "cuda", mesh_dim_names=("dp_shard",)
        )
        initial_table = torch.randn(256, 64)
        initial_matrix = torch.randn(64, 32, device="cuda")

        def build(config_type):
            model = nn.Module()
            model.matrix = nn.Parameter(
                distribute_tensor(initial_matrix.clone(), mesh, (Shard(0),))
            )
            model.norm = nn.Parameter(
                distribute_tensor(torch.ones(64, device="cuda"), mesh, (Shard(0),))
            )
            model.table = nn.Embedding(256, 64)
            model.table.weight.data = initial_table.clone().pin_memory()
            ownership = shard_table(model.table, group)
            optimizer = config_type(
                optimizers=[
                    DistMuon.Config(
                        pattern="matrix",
                        lr=0.003,
                        adjust_lr_fn="match_rms_adamw",
                        compute_sharding_by_fqn={
                            "matrix": ComputeLayout(
                                shardings_by_mesh_axis={"dp_shard": Owned()}
                            )
                        },
                        bucket_configs=[BucketConfig(patterns=("matrix",))],
                    ),
                    AdamW.Config(pattern="table", lr=0.003, fused=True, foreach=False),
                    AdamW.Config(pattern="norm", lr=0.003, fused=True, foreach=False),
                ]
            ).build(model_parts=[model])
            for item in optimizer:
                optimizer_checkpoint_hooks(item, model.table.weight, ownership)
            return model, optimizer, ownership

        reference, expected, ownership = build(OptimizersContainer.Config)
        candidate, actual, _ = build(CPUOverlapOptimizersContainer.Config)

        def local(value):
            return value.to_local() if isinstance(value, DTensor) else value

        for step in range(8):
            torch.manual_seed(1000 + step + rank * 31)
            for container in (expected, actual):
                for optimizer in container:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = 0.003 / (step + 1)
            for first, second in zip(
                reference.parameters(), candidate.parameters(), strict=True
            ):
                gradient = torch.randn_like(local(first))
                if step == 2:
                    gradient.zero_()
                if isinstance(first, DTensor):
                    first.grad = DTensor.from_local(
                        gradient.clone(), mesh, first.placements
                    )
                    second.grad = DTensor.from_local(
                        gradient.clone(), mesh, second.placements
                    )
                else:
                    first.grad = gradient.clone()
                    second.grad = gradient.clone()
            norm_expected = clip_owned(
                list(reference.parameters()),
                reference.table.weight,
                ownership,
                0.7,
                256,
            )
            norm_actual = clip_owned(
                list(candidate.parameters()),
                candidate.table.weight,
                candidate.table.ownership,
                0.7,
                256,
            )
            torch.testing.assert_close(norm_actual, norm_expected, rtol=0, atol=0)
            expected.step()
            actual.step()
            torch.cuda.synchronize()
            for first, second in zip(
                reference.parameters(), candidate.parameters(), strict=True
            ):
                torch.testing.assert_close(local(first), local(second), rtol=0, atol=0)
            first_state, first_spec = tree_flatten(
                tree_map(local, expected.state_dict())
            )
            second_state, second_spec = tree_flatten(
                tree_map(local, actual.state_dict())
            )
            assert first_spec == second_spec
            for first, second in zip(first_state, second_state, strict=True):
                if isinstance(first, torch.Tensor):
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
                else:
                    assert first == second
            if step in (3, 5):
                source_model, source_optimizer, target_model, target_optimizer = (
                    (reference, expected, candidate, actual)
                    if step == 3
                    else (candidate, actual, reference, expected)
                )
                state = {
                    "model": source_model.state_dict(),
                    "optim": source_optimizer.state_dict(),
                }
                location = f"{path}/checkpoint_{step}"
                dcp.save(state, checkpoint_id=location)
                with torch.no_grad():
                    for parameter in target_model.parameters():
                        parameter.zero_()
                loaded = {
                    "model": target_model.state_dict(),
                    "optim": target_optimizer.state_dict(),
                }
                dcp.load(loaded, checkpoint_id=location)
                target_model.load_state_dict(loaded["model"])
                target_optimizer.load_state_dict(loaded["optim"])
                for first, second in zip(
                    reference.parameters(), candidate.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        local(first), local(second), rtol=0, atol=0
                    )
            expected.zero_grad()
            actual.zero_grad()
        if rank == 0:
            print(
                "PASS: exact Moonlight Muon/CPU AdamW/GPU AdamW updates and bidirectional checkpoints",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_cuda_muon_overlap_and_checkpoint():
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as path:
        mp.spawn(_cuda_worker, args=(2, path), nprocs=2, join=True)


@pytest.mark.parametrize("fail_device", [False, True])
def test_cache_communication_follows_device_optimizer(fail_device):
    from torchtitan.experiments.ple_dram_shard.cpu_update import CPUTableUpdate

    model = nn.Linear(2, 2)
    config = _cpu_config(CPUOverlapOptimizersContainer.Config)
    config.defer_cpu_join = True
    optimizer = config.build(model_parts=[model])
    entered = threading.Event()
    calls = []
    update = SimpleNamespace(
        pending=object(),
        communication_ready=None,
        device_update_ready=None,
        compute_cache=SimpleNamespace(refresh_after_step=lambda: calls.append("cache")),
    )

    def cpu_step():
        calls.append("cpu")
        entered.set()
        CPUTableUpdate._refresh_compute_cache(update)
        update.communication_ready.set()

    def device_step():
        assert entered.wait(timeout=5)
        calls.append("device")
        if fail_device:
            raise RuntimeError("device failure")

    update.step = cpu_step
    optimizer._cpu_optimizers = [SimpleNamespace(_ple_table_update=update)]
    optimizer._device_optimizers = [SimpleNamespace(step=device_step)]
    if fail_device:
        with pytest.raises(RuntimeError, match="device failure"):
            optimizer.step()
    else:
        optimizer.step()
    optimizer.finish_cpu_step()
    assert calls == ["cpu", "device", "cache"]
    assert optimizer._pending_cpu_step is None


def test_cache_waits_for_device_completion_event():
    from concurrent.futures import ThreadPoolExecutor

    from torchtitan.experiments.ple_dram_shard.cpu_update import CPUTableUpdate

    gate = threading.Event()
    entered = threading.Event()
    calls = []
    update = SimpleNamespace(
        device_update_ready=(
            gate,
            [SimpleNamespace(synchronize=lambda: calls.append("completed"))],
        ),
        compute_cache=SimpleNamespace(refresh_after_step=lambda: calls.append("cache")),
    )

    def refresh():
        entered.set()
        CPUTableUpdate._refresh_compute_cache(update)

    with ThreadPoolExecutor(1) as pool:
        task = pool.submit(refresh)
        assert entered.wait(timeout=5)
        assert not task.done()
        gate.set()
        task.result(timeout=5)
    assert calls == ["completed", "cache"]
