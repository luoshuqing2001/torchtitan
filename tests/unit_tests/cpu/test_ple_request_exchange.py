# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchtitan.experiments.ple_dram_shard.request_exchange import exchange_indices


def _exchange_worker(rank, path):
    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", init_method=f"file://{path}/group", rank=rank, world_size=2
    )
    try:
        for case in range(5):
            families = tuple(
                torch.arange(0 if case == 0 else rank * 7 + case * i, dtype=torch.int64)
                * (rank + 1)
                - 17
                for i in range(1, case + 2)
            )
            expected = [None] * 2
            dist.all_gather_object(expected, families)
            actual = exchange_indices(families, dist.group.WORLD)
            for request, reference in zip(actual, expected, strict=True):
                for result, value in zip(request, reference, strict=True):
                    assert torch.equal(result, value)
    finally:
        dist.destroy_process_group()


def test_tensor_exchange_matches_object_exchange():
    with tempfile.TemporaryDirectory() as path:
        mp.spawn(_exchange_worker, args=(path,), nprocs=2, join=True)


@pytest.mark.parametrize(
    "families", [(), (torch.ones(2),), (torch.ones(2, 2, dtype=torch.int64),)]
)
def test_exchange_rejects_invalid_families(families):
    with pytest.raises(ValueError, match="CPU int64 vectors"):
        exchange_indices(families, None)
