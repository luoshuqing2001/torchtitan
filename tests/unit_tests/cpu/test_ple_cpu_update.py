# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from types import SimpleNamespace

import pytest
import torch

from torchtitan.experiments.ple_dram_shard.cpu_update import CPUTableUpdate

pytestmark = pytest.mark.skipif(
    torch.backends.cpu.get_cpu_capability() != "AVX512", reason="Requires AVX512"
)


def assert_bits(actual, expected):
    assert torch.equal(
        actual.detach().view(torch.int32), expected.detach().view(torch.int32)
    )


@pytest.mark.parametrize("betas", [(0.9, 0.95), (0.9, 0.999), (0.4, 0.9)])
@pytest.mark.parametrize("num_threads", [0, 1, 4, 26])
def test_updates_restore_and_mutation_match_native(betas, num_threads):
    torch.manual_seed(42)
    reference = torch.nn.Parameter(torch.randn(128, 32))
    candidate = torch.nn.Parameter(reference.detach().clone())
    options = {"lr": 0.003, "betas": betas, "weight_decay": 0.1, "fused": True}
    expected = torch.optim.AdamW([reference], **options)
    actual = torch.optim.AdamW([candidate], **options)
    update = CPUTableUpdate(candidate, actual, num_threads=num_threads)
    for step in range(1, 33):
        rows = torch.randint(96, (7,)).unique()
        gradient = torch.zeros_like(reference)
        gradient[rows] = torch.randn(len(rows), 32)
        if step == 2:
            gradient.mul_(1e-40)
        if step == 3:
            gradient.mul_(1e10)
        reference.grad = gradient.clone()
        candidate.grad = gradient.clone()
        candidate.grad._ple_owned_rows = (candidate.grad._version, rows)
        candidate.grad._ple_owned_norm_cache = (32, candidate.grad._version, None)
        if step == 12:
            # Restore native state, including moments for previously unseen rows.
            with torch.no_grad():
                expected.state[reference]["exp_avg"][127].fill_(0.02)
            actual.load_state_dict(copy.deepcopy(expected.state_dict()))
        if step == 20:
            with torch.no_grad():
                expected.state[reference]["exp_avg"][126].fill_(0.01)
                actual.state[candidate]["exp_avg"][126].fill_(0.01)
        if step == 24:
            # Invalidate the sparse proof through an ordinary in-place write.
            reference.grad[125].fill_(0.03)
            candidate.grad[125].fill_(0.03)
        norm = torch.tensor(3.7)
        coefficient = (1.0 / (norm + 1e-6)).clamp(max=1)
        reference.grad.mul_(coefficient)
        update.defer(norm, 1.0)
        old_versions = (candidate._version, candidate.grad._version)
        expected.step()
        previous_threads = torch.get_num_threads()
        update.step()
        assert torch.get_num_threads() == previous_threads
        assert candidate._version > old_versions[0]
        assert candidate.grad._version > old_versions[1]
        assert_bits(candidate, reference)
        assert_bits(candidate.grad, reference.grad)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert_bits(actual.state[candidate][key], expected.state[reference][key])


def test_hooks_fallback_and_changed_gradient_rejected():
    parameter = torch.nn.Parameter(torch.randn(16, 32))
    optimizer = torch.optim.AdamW([parameter], fused=True)
    update = CPUTableUpdate(parameter, optimizer)
    container = SimpleNamespace(
        optimizers=[optimizer],
        _optimizer_step_pre_hooks={},
        _optimizer_step_post_hooks={},
    )
    assert update.can_fuse(container)
    handle = optimizer.register_step_post_hook(lambda *args: None)
    with pytest.warns(UserWarning, match="native AdamW"):
        assert not update.can_fuse(container)
    handle.remove()
    parameter.grad = torch.ones_like(parameter)
    update.defer(torch.tensor(2.0), 1.0)
    parameter.grad.add_(1)
    with pytest.raises(RuntimeError, match="changed between clipping"):
        update.step()


@pytest.mark.parametrize("num_threads", [-1, True, 1.5])
def test_invalid_update_threads(num_threads):
    parameter = torch.nn.Parameter(torch.zeros(16, 32))
    optimizer = torch.optim.AdamW([parameter], fused=True)
    with pytest.raises(ValueError, match="nonnegative integer"):
        CPUTableUpdate(parameter, optimizer, num_threads=num_threads)


def test_kernel_exception_restores_threads():
    parameter = torch.nn.Parameter(torch.zeros(16, 32))
    CPUTableUpdate(parameter, torch.optim.AdamW([parameter], fused=True))
    previous_threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="row width"):
        torch.ops.torchtitan_ple_cpu.update_seen(
            parameter,
            torch.zeros_like(parameter),
            torch.zeros_like(parameter),
            torch.zeros_like(parameter),
            0.003,
            0.9,
            0.95,
            0.1,
            1e-8,
            1.0,
            1.0,
            torch.ones(16, dtype=torch.bool),
            31,
            4,
        )
    assert torch.get_num_threads() == previous_threads


@pytest.mark.parametrize("fused_table", [False, True])
def test_auxiliary_optimizer_uses_scoped_threads(fused_table):
    parameter = torch.nn.Parameter(torch.ones(16, 32))
    auxiliary = torch.nn.Parameter(torch.ones(16, 32))
    optimizer = torch.optim.AdamW([parameter, auxiliary], fused=True)
    update = CPUTableUpdate(parameter, optimizer, num_threads=4)
    parameter.grad = torch.ones_like(parameter)
    auxiliary.grad = torch.ones_like(auxiliary)
    if fused_table:
        update.defer(torch.tensor(2.0), 1.0)
    observed = []
    handle = optimizer.register_step_post_hook(
        lambda *args: observed.append(torch.get_num_threads())
    )
    previous_threads = torch.get_num_threads()
    update.step()
    handle.remove()
    assert observed == [4]
    assert torch.get_num_threads() == previous_threads
    assert optimizer.state[auxiliary]["step"].item() == 1


def test_optimizer_exception_restores_threads_and_gradient():
    parameter = torch.nn.Parameter(torch.ones(16, 32))
    optimizer = torch.optim.AdamW([parameter], fused=True)
    update = CPUTableUpdate(parameter, optimizer, num_threads=4)
    parameter.grad = torch.ones_like(parameter)
    gradient = parameter.grad
    update.defer(torch.tensor(2.0), 1.0)

    def fail(*args):
        raise RuntimeError("CPU optimizer hook failure")

    handle = optimizer.register_step_post_hook(fail)
    previous_threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="CPU optimizer hook failure"):
        update.step()
    handle.remove()
    assert torch.get_num_threads() == previous_threads
    assert parameter.grad is gradient


@pytest.mark.parametrize("norm", [0.0, 2.0])
def test_gradient_store_preserves_bits(norm):
    parameter = torch.nn.Parameter(torch.zeros(16, 32))
    optimizer = torch.optim.AdamW([parameter], fused=True)
    update = CPUTableUpdate(parameter, optimizer, num_threads=4)
    gradient = (
        torch.tensor([0.0, -0.0, 1e-40, -1e-40, 1.0, -1.0, float("nan"), float("inf")])
        .repeat(64)
        .view(16, 32)
    )
    parameter.grad = gradient.clone()
    norm = torch.tensor(norm)
    expected = gradient * (1.0 / (norm + 1e-6)).clamp(max=1)
    update.defer(norm, 1.0)
    update.step()
    assert_bits(parameter.grad, expected)


@pytest.mark.parametrize("betas", [(0.9, 0.95), (0.4, 0.9)])
@pytest.mark.parametrize("prepare", [False, True])
def test_prepared_updates_and_native_fallback(betas, prepare):
    torch.manual_seed(42)
    reference = torch.nn.Parameter(torch.randn(128, 64))
    candidate = torch.nn.Parameter(reference.detach().clone())
    expected = torch.optim.AdamW([reference], lr=0.003, betas=betas, fused=True)
    actual = torch.optim.AdamW([candidate], lr=0.003, betas=betas, fused=True)
    update = CPUTableUpdate(candidate, actual, num_threads=4)
    for step in range(12):
        rows = torch.arange(step * 3, step * 3 + 5)
        if prepare:
            update.prepare(rows)
        gradient = torch.zeros_like(reference)
        gradient[rows] = torch.randn(len(rows), 64)
        reference.grad = gradient.clone()
        candidate.grad = gradient.clone()
        candidate.grad._ple_owned_rows = (candidate.grad._version, rows)
        candidate.grad._ple_owned_norm_cache = (64, candidate.grad._version, None)
        norm = torch.tensor(3.7)
        coefficient = (1.0 / (norm + 1e-6)).clamp(max=1)
        reference.grad.mul_(coefficient)
        if step in (2, 5):
            # Exercise a native step between fused updates, including unseen rows.
            candidate.grad.mul_(coefficient)
        else:
            update.defer(norm, 1.0)
        expected.step()
        update.step()
        assert_bits(candidate, reference)
        assert_bits(candidate.grad, reference.grad)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert_bits(actual.state[candidate][key], expected.state[reference][key])
    assert update.prepared_steps == (9 if prepare else 0)


@pytest.mark.parametrize("invalidate", ["gradient", "lr", "reload"])
def test_preparation_is_read_only_and_invalidated(invalidate):
    torch.manual_seed(42)
    reference = torch.nn.Parameter(torch.randn(32, 64))
    candidate = torch.nn.Parameter(reference.detach().clone())
    expected = torch.optim.AdamW([reference], lr=0.003, fused=True)
    actual = torch.optim.AdamW([candidate], lr=0.003, fused=True)
    update = CPUTableUpdate(candidate, actual, num_threads=4)
    for parameter in (reference, candidate):
        parameter.grad = torch.ones_like(parameter)
    expected.step()
    update.step()
    old = candidate.detach().clone()
    old_state = copy.deepcopy(actual.state_dict())
    rows = torch.tensor([1, 3, 9])
    update.prepare(rows)
    for future in update.preparations:
        future.result()
    assert_bits(candidate, old)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        assert_bits(actual.state[candidate][key], old_state["state"][0][key])
    if invalidate == "lr":
        expected.param_groups[0]["lr"] = actual.param_groups[0]["lr"] = 0.002
    elif invalidate == "reload":
        actual.load_state_dict(copy.deepcopy(expected.state_dict()))
    gradient = torch.zeros_like(reference)
    gradient[rows] = torch.randn(3, 64)
    reference.grad = gradient.clone()
    candidate.grad = gradient.clone()
    candidate.grad._ple_owned_rows = (candidate.grad._version, rows)
    candidate.grad._ple_owned_norm_cache = (64, candidate.grad._version, None)
    if invalidate == "gradient":
        reference.grad.add_(0.2)
        candidate.grad.add_(0.2)
    norm = torch.tensor(3.7)
    reference.grad.mul_((1.0 / (norm + 1e-6)).clamp(max=1))
    update.defer(norm, 1.0)
    expected.step()
    update.step()
    assert_bits(candidate, reference)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        assert_bits(actual.state[candidate][key], expected.state[reference][key])


@pytest.mark.parametrize("adversarial", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_prepared_compute_cache_matches_full_cast(
    adversarial, packed, streaming, monkeypatch
):
    torch.manual_seed(42)
    parameter = torch.nn.Parameter(torch.randn(128, 64))
    optimizer = torch.optim.AdamW([parameter], lr=0.003, fused=True)
    update = CPUTableUpdate(parameter, optimizer, num_threads=4)
    if adversarial:
        bits = torch.tensor(
            [
                -2147483648,
                0,
                1,
                0x007FFFFF,
                0x00800000,
                0x3F808000,
                0x7F800000,
                0x7FC00001,
            ],
            dtype=torch.int32,
        )
        with torch.no_grad():
            parameter.copy_(bits.view(torch.float32).repeat(1024).reshape_as(parameter))
    cache = SimpleNamespace(
        dtype=torch.bfloat16,
        host=torch.empty_like(parameter, dtype=torch.bfloat16),
        ready=SimpleNamespace(synchronize=lambda: None),
        host_version=None,
        refresh_after_step=lambda: None,
    )
    cache.double_buffer = packed
    cache.packed_host = torch.empty_like(cache.host)
    cache.candidate_layout = None
    cache.table = SimpleNamespace(ownership=SimpleNamespace(group=None))
    cache.wait_host = lambda: None
    cache.preload = lambda layout: setattr(cache, "candidate_layout", layout)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **kw: None)
    update.compute_cache = cache
    update.streaming = streaming
    for step in range(8):
        rows = torch.arange(step * 3, step * 3 + 7)
        layout = (rows, [len(rows)]) if packed else None
        update.prepare(rows, cache_layout=layout)
        gradient = torch.zeros_like(parameter)
        gradient[rows] = torch.randn(len(rows), 64)
        parameter.grad = gradient
        gradient._ple_owned_rows = (gradient._version, rows)
        gradient._ple_owned_norm_cache = (64, gradient._version, None)
        update.defer(torch.tensor(3.7), 1.0)
        update.step()
        if step:
            if packed:
                assert cache.patch_layout is layout
                assert torch.equal(
                    cache.packed_host[: len(rows)].view(torch.int16),
                    parameter.index_select(0, rows).to(cache.dtype).view(torch.int16),
                )
            else:
                assert cache.host_version == parameter._version
                assert torch.equal(
                    cache.host.view(torch.int16),
                    parameter.to(cache.dtype).view(torch.int16),
                )


def test_native_fallback_invalidates_compute_cache():
    parameter = torch.nn.Parameter(torch.randn(16, 64))
    optimizer = torch.optim.AdamW([parameter], fused=True)
    update = CPUTableUpdate(parameter, optimizer, num_threads=4)
    cache = SimpleNamespace(
        host_version=parameter._version, refresh_after_step=lambda: None
    )
    update.compute_cache = cache
    parameter.grad = torch.ones_like(parameter)
    before = parameter.detach().clone()
    update.step()
    assert not torch.equal(parameter, before)
    assert cache.host_version is None


@pytest.mark.parametrize("fused", [False, True])
def test_cache_refresh_overlaps_auxiliary_update(fused):
    parameter = torch.nn.Parameter(torch.ones(16, 64))
    auxiliary = torch.nn.Parameter(torch.ones(16, 64))
    optimizer = torch.optim.AdamW([parameter, auxiliary], lr=0.01, fused=True)
    update = CPUTableUpdate(parameter, optimizer, num_threads=4)
    observations = []
    cache = SimpleNamespace(
        host_version=None,
        refresh_after_step=lambda: observations.append(
            (parameter.detach().clone(), auxiliary.detach().clone())
        ),
    )
    update.compute_cache = cache
    parameter.grad = torch.ones_like(parameter)
    auxiliary.grad = torch.ones_like(auxiliary)
    if fused:
        update.defer(torch.tensor(2.0), 1.0)
    update.step()
    assert len(observations) == 1
    table_seen, auxiliary_seen = observations[0]
    assert torch.equal(table_seen, parameter)
    assert torch.equal(auxiliary_seen, torch.ones_like(auxiliary)) == fused
    assert not torch.equal(auxiliary, torch.ones_like(auxiliary))
