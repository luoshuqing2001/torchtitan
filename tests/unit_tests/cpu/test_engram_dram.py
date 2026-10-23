# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.experiments.engram_dram.compact import expand, families, inverse
from torchtitan.experiments.engram_dram.indexing import hash_ids
from torchtitan.experiments.engram_dram.recipe import glimmer_dense
from torchtitan.models.glimmer_dense.engram import EngramTable


@pytest.mark.parametrize("rows", [1, 2, 31, 128, 1204224])
@pytest.mark.parametrize("length", [1, 2, 37, 1025])
def test_compact_indices_match_expanded_hash(rows, length):
    torch.manual_seed(42)
    tokens = torch.randint(0, 100352, (length,))
    if length > 5:
        tokens[5] = 2
    positions = torch.arange(length) % 7
    expected = hash_ids(tokens, positions, rows, 2)
    expected_unique, expected_inverse = torch.unique(expected, return_inverse=True)
    grouped = families(tokens, positions, rows, 2)
    torch.testing.assert_close(
        expand(tuple(g[0] for g in grouped), rows), expected_unique, rtol=0, atol=0
    )
    torch.testing.assert_close(inverse(grouped, rows), expected_inverse, rtol=0, atol=0)
    if rows < 1000:
        table = EngramTable.Config(dim=32, rows_per_head=rows, eos_token_id=2).build()
        torch.nn.init.normal_(table.weight)
        torch.testing.assert_close(
            table(tokens, positions),
            table.weight[expected].reshape(length, 32),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "tokens",
    [
        torch.tensor([-1, 2]),
        torch.tensor([torch.iinfo(torch.int64).max]),
        torch.empty(0, dtype=torch.int64),
    ],
)
def test_compact_overflow_falls_back(tokens):
    assert families(tokens, torch.zeros_like(tokens), 31, 2) is None


def test_dram_default_is_gpu(monkeypatch):
    monkeypatch.setenv("PLE", "engram_legacy")
    monkeypatch.delenv("ENGRAM_ASYNC_TRAINING", raising=False)
    config = glimmer_dense()
    assert config.model.__class__.__qualname__ == "GlimmerDenseModel.Config"


def test_dram_recipe_targets_engram_table(monkeypatch):
    monkeypatch.setenv("PLE", "engram_legacy")
    monkeypatch.setenv("ENGRAM_ASYNC_TRAINING", "1")
    config = glimmer_dense()
    assert config.optim.table_path == "engram.table"
    assert not config.model.ple_async_training


@pytest.mark.parametrize("rank", range(4))
def test_prepared_routing_preserves_order(monkeypatch, rank):
    from types import SimpleNamespace

    from torchtitan.experiments.ple_dram_shard.routing import prepare_routing

    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: rank)
    ownership = SimpleNamespace(group=None, num_rows=32, width=64, vocab_size=128)
    torch.manual_seed(42)
    requests = [torch.randint(128, (48 + i,)).unique() for i in range(4)]
    union = torch.unique(torch.cat(requests))
    local = union[(union >= rank * 32) & (union < (rank + 1) * 32)]
    expected = prepare_routing(requests, ownership, pin_memory=False)
    actual = prepare_routing(requests, ownership, local, pin_memory=False)
    assert torch.equal(actual[0], expected[0])
    assert actual[1:3] == expected[1:3]
    assert torch.equal(actual[3], expected[3])
    selected = [r[(r >= rank * 32) & (r < (rank + 1) * 32)] for r in requests]
    received = torch.cat((*selected, torch.tensor([128])))
    for ids, mapping in zip(selected, actual[3]):
        for key, position in zip(local.tolist(), mapping.tolist()):
            assert received[position].item() == (key if key in ids else 128)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("gpu_projections", [False, True])
def test_full_module_updates_refresh_cpu_parameters(tmp_path, compact, gpu_projections):
    from types import SimpleNamespace

    from torchtitan.experiments.engram_dram.model import DRAMEngram
    from torchtitan.models.glimmer_dense.engram import NGramEngram

    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{tmp_path}/group", rank=0, world_size=1
    )
    try:
        options = {
            "dim": 32,
            "n_layers": 2,
            "ple_dim": 16,
            "eos_token_id": 2,
            "rows_per_head": 31,
            "remainder": 3,
        }
        expected = NGramEngram.Config(**options).build()
        actual = DRAMEngram.Config(
            **options,
            compact_indices=compact,
            cache_stages=True,
            gpu_projections=gpu_projections,
        ).build()
        torch.manual_seed(42)
        with torch.no_grad():
            for parameter in expected.parameters():
                parameter.normal_(std=0.03)
        actual.load_state_dict(expected.state_dict())
        actual.table.ownership = SimpleNamespace(
            group=None,
            offset=0,
            num_rows=496,
            vocab_size=496,
            width=2,
            owner_routing=False,
            tensor_exchange=False,
            gradient_transport=None,
        )
        optimizers = [
            torch.optim.AdamW(model.parameters(), lr=0.01, fused=True)
            for model in (expected, actual)
        ]
        for step in range(4):
            tokens = torch.randint(0, 100, (37,))
            tokens[6] = 2
            positions = torch.arange(37) % 11
            hidden = torch.randn(37, 32)
            target = torch.randn_like(hidden)
            outputs = [model(hidden, tokens, positions) for model in (expected, actual)]
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            for output in outputs:
                (output * target).sum().backward()
            for a, b in zip(expected.parameters(), actual.parameters(), strict=True):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for a, b in zip(expected.parameters(), actual.parameters(), strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    "rows,length", [(1, 0), (1, 103), (31, 1000), (1204224, 262144)]
)
def test_bounded_unique_matches_torch(rows, length):
    from torchtitan.experiments.ple_dram_shard.cpu_update import load_kernel

    load_kernel()
    torch.manual_seed(42)
    values = torch.randint(rows, (length,))
    for return_inverse in (False, True):
        result, inverse = torch.ops.torchtitan_ple_cpu.unique_bounded(
            values, rows, return_inverse
        )
        expected, expected_inverse = torch.unique(values, return_inverse=True)
        assert torch.equal(result, expected)
        assert torch.equal(inverse, expected_inverse if return_inverse else values[:0])


@pytest.mark.skipif(
    torch.backends.cpu.get_cpu_capability() != "AVX512", reason="Requires AVX512"
)
def test_bounded_unique_rejects_bad_keys():
    from torchtitan.experiments.ple_dram_shard.cpu_update import load_kernel

    load_kernel()
    with pytest.raises(RuntimeError, match="outside row range"):
        torch.ops.torchtitan_ple_cpu.unique_bounded(torch.tensor([-1, 32]), 32, True)


@pytest.mark.parametrize("rows", [1, 31, 1204224])
def test_device_inverse_packet_matches_expanded(rows):
    from torchtitan.experiments.engram_dram.compact import (
        _inverse_packet,
        build_inverse,
    )

    torch.manual_seed(42)
    tokens = torch.randint(100352, (1025,))
    positions = torch.arange(1025) % 17
    family = families(tokens, positions, rows, 2)
    assert torch.equal(
        build_inverse(*_inverse_packet(family, rows)), inverse(family, rows)
    )


def test_gpu_projections_keep_native_optimizer_group(monkeypatch):
    import re

    monkeypatch.setenv("PLE", "engram_legacy")
    monkeypatch.setenv("ENGRAM_ASYNC_TRAINING", "1")
    monkeypatch.setenv("ENGRAM_DRAM_GPU_PROJECTIONS", "1")
    config = glimmer_dense()
    pattern = config.optim.optimizer.optimizers[-2].pattern
    assert re.search(pattern, "engram.table.weight")
    assert not re.search(pattern, "engram.update.key_weight")


def test_value_offloader_leaves_cpu_autograd_unchanged():
    from torchtitan.experiments.engram_dram.activation import ValueOffloader

    torch.manual_seed(42)
    left = torch.randn(17, 31, requires_grad=True)
    right = torch.randn(31, 7, requires_grad=True)
    reference_left = left.detach().clone().requires_grad_()
    reference_right = right.detach().clone().requires_grad_()
    offloader = ValueOffloader()
    with offloader.capture(left.shape):
        actual = left @ right
    expected = reference_left @ reference_right
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    assert not offloader.records
    assert torch.equal(actual, expected)
    assert torch.equal(left.grad, reference_left.grad)
    assert torch.equal(right.grad, reference_right.grad)


def test_value_prefetch_hook_preserves_gradient():
    from types import SimpleNamespace

    from torchtitan.experiments.engram_dram.activation import ValueOffloader

    calls = []
    offloader = ValueOffloader()
    offloader.records = [SimpleNamespace(prefetch=lambda: calls.append(True), gpu_TV=1)]
    inputs = torch.randn(7, requires_grad=True)
    output = inputs * 2
    offloader.decoder_forward_hook(None, None, output)
    output.sum().backward()
    assert calls == [True]
    assert torch.equal(inputs.grad, torch.full_like(inputs, 2))
    offloader.clear()
    assert not offloader.records


@pytest.mark.parametrize("requires_grad", [False, True])
def test_value_checkpoint_preserves_projection_gradients(requires_grad):
    from torch.utils.checkpoint import checkpoint

    from torchtitan.experiments.engram_dram.model import DRAMEngram

    torch.manual_seed(42)
    source = (
        torch.randn(37, 16),
        torch.randn(51, 16),
        torch.randn(16, 51),
    )
    incoming = torch.randn(37, 16)
    results = []
    saved = []
    for enabled in (False, True):
        inputs = [
            t.clone().requires_grad_(i > 0 or requires_grad)
            for i, t in enumerate(source)
        ]
        shapes = []

        def pack(tensor, shapes=shapes):
            shapes.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            if enabled:
                output = checkpoint(
                    DRAMEngram._value_projection,
                    *inputs,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                output = DRAMEngram._value_projection(*inputs)
        output.backward(incoming)
        results.append((output.detach(), [t.grad for t in inputs]))
        saved.append(shapes)
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    assert (37, 51) in saved[0]
    assert (37, 51) not in saved[1]


def test_workspace_reuse_requires_double_buffer(monkeypatch):
    monkeypatch.setenv("PLE", "engram_legacy")
    monkeypatch.setenv("ENGRAM_ASYNC_TRAINING", "1")
    monkeypatch.setenv("ENGRAM_DRAM_REUSE_CACHE_WORKSPACE", "1")
    monkeypatch.setenv("ENGRAM_DRAM_DOUBLE_BUFFER_CACHE", "0")
    with pytest.raises(ValueError, match="requires double buffering"):
        glimmer_dense()


@pytest.mark.parametrize("requires_grad", [False, True])
def test_prefetched_value_recomputation_matches_native(requires_grad):
    from torchtitan.experiments.engram_dram.recompute import ValueRecomputer

    torch.manual_seed(42)
    source = [torch.randn(37, 16), torch.randn(51, 16), torch.randn(16, 51)]
    incoming = torch.randn(37, 16)
    results = []
    for enabled in (False, True):
        inputs = [
            t.clone().requires_grad_(i > 0 or requires_grad)
            for i, t in enumerate(source)
        ]
        owner = ValueRecomputer()
        if enabled:
            output = owner.project(*inputs)
        else:
            output = torch.nn.functional.linear(
                torch.nn.functional.linear(inputs[0], inputs[1]), inputs[2]
            )
        output.backward(incoming)
        results.append((output.detach(), [t.grad for t in inputs]))
        assert not owner.records
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


def test_value_recomputation_preserves_second_derivatives():
    from torchtitan.experiments.engram_dram.recompute import ValueRecomputer

    torch.manual_seed(42)
    inputs = tuple(
        torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        for shape in [(3, 2), (4, 2), (2, 4)]
    )
    assert torch.autograd.gradgradcheck(ValueRecomputer().project, inputs)


def test_value_recomputation_rejects_aliased_mutation():
    from torchtitan.experiments.engram_dram.recompute import ValueRecomputer

    inputs = torch.randn(3, 2, requires_grad=True)
    up = torch.randn(4, 2, requires_grad=True)
    down = torch.randn(2, 4, requires_grad=True)
    output = ValueRecomputer().project(inputs, up, down)
    with torch.no_grad():
        inputs.detach().add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()


def test_value_recomputation_releases_original_activation(monkeypatch):
    import weakref

    from torchtitan.experiments.engram_dram.recompute import ValueRecomputer

    native = torch.nn.functional.linear
    references = []

    def linear(*args, **kwargs):
        result = native(*args, **kwargs)
        if result.shape == (37, 51):
            references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(torch.nn.functional, "linear", linear)
    inputs = torch.randn(37, 16, requires_grad=True)
    up = torch.randn(51, 16, requires_grad=True)
    down = torch.randn(16, 51, requires_grad=True)
    owner = ValueRecomputer()
    output = owner.project(inputs, up, down)
    assert len(references) == 1
    assert references[0]() is None
    output.sum().backward()
    assert not owner.records


def test_aligned_value_linear_second_derivatives():
    from torchtitan.experiments.engram_dram.recompute import AlignedValueLinear

    torch.manual_seed(42)
    inputs = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(12, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(AlignedValueLinear.apply, (inputs, weight))
    assert torch.autograd.gradgradcheck(AlignedValueLinear.apply, (inputs, weight))


def test_aligned_value_forward_requires_aligned_recomputation(monkeypatch):
    monkeypatch.setenv("PLE", "engram_legacy")
    monkeypatch.setenv("ENGRAM_ASYNC_TRAINING", "1")
    monkeypatch.setenv("ENGRAM_DRAM_ALIGN_VALUE_FORWARD", "1")
    with pytest.raises(ValueError, match="requires aligned prefetched values"):
        glimmer_dense()


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_cached_forward_joins_routing_without_backward(monkeypatch, grad_enabled):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from torchtitan.experiments.engram_dram import cache, model

    future = Mock()
    request = SimpleNamespace(future=future)
    tokens = torch.arange(4)
    tokens._engram_prefetch = request
    embeddings = torch.ones(4, 8)
    monkeypatch.setattr(model, "hash_ids", lambda *args: tokens)
    monkeypatch.setattr(cache.CachedEmbedding, "apply", lambda *args: embeddings)
    table = SimpleNamespace(
        weight=torch.ones(4, 8),
        remainder_weight=None,
        gpu_cache=SimpleNamespace(value=lambda dtype: embeddings),
    )
    module = SimpleNamespace(
        table=table,
        rows_per_head=4,
        eos_token_id=99,
        value_offloader=None,
        _apply_update=lambda hidden, values, tokens, positions: values,
    )
    with torch.set_grad_enabled(grad_enabled):
        output = model.DRAMEngram.forward(module, embeddings, tokens, tokens)
    assert output.shape == embeddings.shape
    assert future.result.call_count == int(not grad_enabled)
    assert not hasattr(tokens, "_engram_prefetch")


def test_trainer_close_finishes_final_validation_work(monkeypatch):
    from types import SimpleNamespace

    from torchtitan.experiments.engram_dram.prefetch import EarlyEngramTrainer
    from torchtitan.models.glimmer_dense import ple_dram
    from torchtitan.trainer import Trainer

    calls = []
    pool = SimpleNamespace(
        submit=lambda task: SimpleNamespace(result=lambda: calls.append("fetch"))
    )
    monkeypatch.setattr(ple_dram, "_POOL", pool)
    monkeypatch.setattr(Trainer, "close", lambda self: calls.append("engine"))
    table = SimpleNamespace(
        ownership=SimpleNamespace(
            prepared_update=SimpleNamespace(
                join_preparations=lambda: calls.append("prepare")
            )
        ),
        gpu_cache=SimpleNamespace(
            wait_host=lambda: calls.append("cache"),
            cancel_lookup=lambda: calls.append("cancel"),
        ),
    )
    trainer = object.__new__(EarlyEngramTrainer)
    trainer.engine = SimpleNamespace(
        model_parts=[SimpleNamespace(engram=SimpleNamespace(table=table))],
        optim=SimpleNamespace(finish_step=lambda: calls.append("update")),
    )
    trainer.close()
    assert calls == ["update", "cancel", "fetch", "prepare", "cache", "engine"]


def test_single_cache_rejects_multiple_training_readers():
    from types import SimpleNamespace

    from torchtitan.experiments.engram_dram.cache import TableCache

    cache = object.__new__(TableCache)
    cache.single_buffer = True
    cache.lookup_version = None
    cache.table = SimpleNamespace(weight=torch.zeros(3, 4))
    cache.begin_lookup(training=True)
    with pytest.raises(ValueError, match="one training lookup"):
        cache.begin_lookup(training=True)
    cache.table.weight.add_(1)
    cache.begin_lookup(training=True)
    cache.begin_lookup(training=False)
    assert not cache.preload_enabled


@pytest.mark.parametrize("ordered", [False, True])
def test_single_cache_waits_for_reader_and_can_cancel(ordered):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from torchtitan.experiments.engram_dram.cache import TableCache

    cache = object.__new__(TableCache)
    cache.ordered_collectives = ordered
    cache.single_buffer = True
    cache.lookup_version = None
    cache.table = SimpleNamespace(weight=torch.zeros(3, 4))
    cache.begin_lookup(training=True)
    entered = threading.Event()

    def preload():
        entered.set()
        return cache.preload(None)

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(preload)
        assert entered.wait(timeout=5)
        assert not future.done()
        cache.cancel_lookup()
        assert future.result(timeout=5) is False


def test_cache_preload_finishes_before_last_loss_chunk(monkeypatch):
    from types import SimpleNamespace

    from torchtitan.experiments.engram_dram.cache import TableCache

    calls = []
    cache = object.__new__(TableCache)
    cache.preload_enabled = True
    cache.loss_chunk = 0
    cache.loss_chunks = 4
    cache.device = None
    cache.next_ready = object()
    cache.lookup_request = SimpleNamespace(
        future=SimpleNamespace(result=lambda: calls.append("metadata"))
    )
    future = SimpleNamespace(
        result=lambda: calls.append("prepare") or (None,) * 5 + (object(),)
    )
    cache.table = SimpleNamespace(
        ownership=SimpleNamespace(
            prepared_update=SimpleNamespace(preparations=[future])
        )
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(
            wait_event=lambda event: calls.append(("wait", event))
        ),
    )
    for _ in range(3):
        cache.before_loss_chunk()
    assert calls == []
    cache.before_loss_chunk()
    assert calls == ["metadata", "prepare", ("wait", cache.next_ready)]


@pytest.mark.parametrize("counts", [(0, 1, 9, 3), (8, 4, 0, 12), (1, 0, 0, 0)])
@pytest.mark.parametrize("rank", range(4))
def test_single_cache_chunk_patch_preserves_other_rows(monkeypatch, counts, rank):
    from types import SimpleNamespace

    from torchtitan.experiments.engram_dram import kernels
    from torchtitan.experiments.engram_dram.cache import TableCache

    cache = object.__new__(TableCache)
    cache.table = SimpleNamespace(ownership=SimpleNamespace(group=None))
    cache.group = None
    cache.patch_chunk_rows = chunk = 4
    cache.transient_workspace = False
    cache.patch_workspace = torch.empty(4, chunk, 16)
    cache.full = torch.arange(64 * 16, dtype=torch.float32).reshape(64, 16)
    expected = cache.full.clone()
    cache.next_ids = torch.arange(sum(counts)) * 2
    updates = torch.arange(sum(counts) * 16, dtype=torch.float32).reshape(-1, 16).neg()
    cache.packed_host = updates.split(counts)[rank].clone()
    expected.index_copy_(0, cache.next_ids, updates)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: rank)
    offset = 0

    def gather(output, local, *, group):
        nonlocal offset
        for owner, values in enumerate(updates.split(counts)):
            count = max(0, min(chunk, len(values) - offset))
            if owner == rank:
                assert torch.equal(local[:count], values[offset : offset + count])
            cache.patch_workspace[owner, :count].copy_(values[offset : offset + count])
        offset += chunk

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    monkeypatch.setattr(
        kernels, "scatter_rows", lambda dst, ids, src: dst.index_copy_(0, ids, src)
    )
    cache._patch_chunks(counts)
    assert torch.equal(cache.full, expected)


def test_single_cache_requires_prepared_reuse_path(monkeypatch):
    monkeypatch.setenv("PLE", "engram_legacy")
    monkeypatch.setenv("ENGRAM_ASYNC_TRAINING", "1")
    monkeypatch.setenv("ENGRAM_DRAM_SINGLE_BUFFER_CACHE", "1")
    with pytest.raises(ValueError, match="Single buffering requires"):
        glimmer_dense()
