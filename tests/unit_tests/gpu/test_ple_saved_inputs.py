# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Saved-input version, storage and reconstruction checks."""

import gc
import weakref

import pytest
import torch

from torchtitan.experiments.ple_dram_shard.saved_inputs import (
    PackedInput,
    SavedInputStaging,
    load_version_tracker,
    positional_checkpoint,
)


@pytest.fixture(scope="module", autouse=True)
def initialize_version_tracker():
    load_version_tracker()


def test_saved_version_survives_tensor_release():
    value = torch.ones(8)
    tracker = torch.classes.torchtitan_ple_saved_input.SavedVersion(value)
    reference = weakref.ref(value)
    del value
    gc.collect()
    assert reference() is None
    tracker.check()


def test_saved_version_detects_alias_mutation():
    value = torch.ones(8)
    tracker = torch.classes.torchtitan_ple_saved_input.SavedVersion(value)
    value[2:].add_(1)
    with pytest.raises(RuntimeError, match="modified in-place"):
        tracker.check()


def test_unstaged_tensor_preserves_version_check():
    manager = SavedInputStaging(12)
    value = torch.ones(8, requires_grad=True)
    packed = manager.pack(value, ("different",), 0)
    assert manager.packed_count == 0
    assert torch.equal(manager.unpack(packed), value)
    with torch.no_grad():
        value.add_(1)
    with pytest.raises(RuntimeError, match="changed in place"):
        manager.unpack(packed)


def test_keyword_checkpoint_inputs_go_through_saved_hooks():
    import functools

    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(42)
    first = torch.randn(4, 8, requires_grad=True)
    second = torch.randn(4, 8, requires_grad=True)
    seen = []

    def compute(value, *, scale):
        return (value.sin() * scale).sum()

    def pack(value):
        seen.append(value.data_ptr())
        return value.detach()

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        result = positional_checkpoint(
            functools.partial(checkpoint, use_reentrant=False),
            compute,
            first,
            scale=second,
        )
    result.backward()
    expected = [first.grad.clone(), second.grad.clone()]
    first.grad = second.grad = None
    compute(first, scale=second).backward()
    assert second.data_ptr() in seen
    torch.testing.assert_close(expected, [first.grad, second.grad], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_strided_lookup_roundtrip_releases_gpu_input(dtype):
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    parent = torch.randn(37, 12, 16, device="cuda", dtype=dtype)
    value = parent[:, 5]
    expected = value.cpu().clone()
    source = weakref.ref(value)
    stream = torch.cuda.Stream()
    packed = PackedInput(value, stream)
    del value, parent
    gc.collect()
    assert source() is None
    packed.prefetch()
    actual = packed.unpack()
    assert torch.equal(actual.cpu(), expected)
    packed.value = None
    packed.prefetch()
    assert torch.equal(packed.unpack().cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_staged_gpu_alias_mutation_is_rejected():
    value = torch.ones(37, 12, 16, device="cuda")
    packed = PackedInput(value[:, 5], torch.cuda.Stream())
    value.add_(1)
    with pytest.raises(RuntimeError, match="modified in-place"):
        packed.unpack()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_token_index_compression_is_lossless(dtype):
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    unique = torch.randn(17, 12, 64, device="cuda", dtype=dtype)
    inverse = torch.arange(113, device="cuda") % 17
    inverse = inverse[torch.randperm(113, device="cuda")]
    first = torch.full((17,), 113, device="cuda", dtype=torch.int64)
    first.scatter_reduce_(0, inverse, torch.arange(113, device="cuda"), reduce="amin")
    parent = unique.index_select(0, inverse)
    value = parent[:, 5]
    expected = value.cpu()
    packed = PackedInput(
        value, torch.cuda.Stream(), first=first, inverse=inverse, audit=True
    )
    assert packed.host.shape == (17, 64)
    assert bool(packed.audit_check)
    del value, parent, unique
    assert torch.equal(packed.unpack().cpu(), expected)
