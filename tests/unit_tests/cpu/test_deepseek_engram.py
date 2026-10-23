# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from torchtitan.models.glimmer_dense import build_model_config
from torchtitan.models.glimmer_dense.deepseek_engram import (
    HEADS_PER_ORDER,
    NGRAM_ORDERS,
)


def _layer(flavor: str = "d200m", ple_dim: int = 256):
    torch.manual_seed(0)
    model = build_model_config(flavor, memory="engram", ple_dim=ple_dim).build()
    model.init_states()
    return model.deepseek_engram.layers["1"]


def _reference_hashes(layer, ids: np.ndarray) -> np.ndarray:
    """``NgramHashMapping._get_ngram_hashes`` from deepseek-ai/Engram, one row."""
    config = layer.config
    x = layer._lookup[ids][None]
    length = x.shape[1]
    multipliers = np.array(config.multipliers)

    def shift(k):
        if k == 0:
            return x
        return np.pad(x, ((0, 0), (k, 0)), constant_values=config.pad_id)[:, :length]

    base = [shift(k) for k in range(max(NGRAM_ORDERS))]
    hashes = []
    for order_index, n in enumerate(NGRAM_ORDERS):
        mix = base[0] * multipliers[0]
        for k in range(1, n):
            mix = np.bitwise_xor(mix, base[k] * multipliers[k])
        for head in range(HEADS_PER_ORDER):
            hashes.append(
                mix % config.table_sizes[order_index * HEADS_PER_ORDER + head]
            )
    return np.stack(hashes, axis=2)[0]


def test_hash_matches_reference_implementation():
    layer = _layer()
    ids = torch.randint(0, 100278, (2048,), generator=torch.Generator().manual_seed(1))
    rows = layer.hash_indices(ids, torch.arange(ids.shape[0])).numpy()
    expected = _reference_hashes(layer, ids.numpy()) + np.array(layer._offsets)
    np.testing.assert_array_equal(rows, expected)


def test_heads_collide_independently():
    layer = _layer()
    ids = torch.randint(
        0, 100278, (200_000,), generator=torch.Generator().manual_seed(2)
    )
    rows = layer.hash_indices(ids, torch.arange(ids.shape[0])).numpy()
    order = np.argsort(rows[:, 0], kind="stable")
    same = rows[order[1:], 0] == rows[order[:-1], 0]
    first, second = order[1:][same], order[:-1][same]
    assert same.sum() > 0
    shared = (rows[first, 1] == rows[second, 1]) & (ids[first] != ids[second]).numpy()
    assert shared.mean() < 0.01


def test_context_stops_at_document_boundary():
    layer = _layer()
    ids = torch.randint(0, 100278, (64,), generator=torch.Generator().manual_seed(3))
    positions = torch.cat((torch.arange(32), torch.arange(32)))
    changed = ids.clone()
    changed[30:32] = 7
    before = layer.hash_indices(ids, positions)
    after = layer.hash_indices(changed, positions)
    torch.testing.assert_close(before[32:], after[32:])
    assert (before[30:32] != after[30:32]).any()


def test_every_parameter_trains_from_init():
    layer = _layer()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    ids = torch.randint(0, 100278, (256,), generator=torch.Generator().manual_seed(4))
    positions = torch.arange(ids.shape[0])
    hidden = torch.randn(ids.shape[0], layer.config.dim)
    for _ in range(2):
        optimizer.zero_grad()
        layer(hidden, ids, positions).pow(2).sum().backward()
        optimizer.step()
    for name, param in layer.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, name


@pytest.mark.parametrize("flavor", ["d200m", "d400m", "d700m", "d1b"])
@pytest.mark.parametrize("ple_dim", [256, 512, 1024])
def test_total_parameters_match_ple(flavor, ple_dim):
    with torch.device("meta"):
        ple = build_model_config(flavor, memory="ple", ple_dim=ple_dim).build()
        engram = build_model_config(flavor, memory="engram", ple_dim=ple_dim).build()
    ple_total = sum(p.numel() for p in ple.parameters())
    engram_total = sum(p.numel() for p in engram.parameters())
    assert engram_total <= ple_total
    assert ple_total - engram_total < 1e-4 * ple_total
