# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exchange variable-length CPU index families without Python serialization."""

import torch
import torch.distributed as dist


def exchange_indices(families, group):
    """Return each rank's families in their original rank and element order."""
    if not families or any(
        value.device.type != "cpu" or value.dtype != torch.int64 or value.ndim != 1
        for value in families
    ):
        raise ValueError(
            "Index exchange requires nonempty families of CPU int64 vectors"
        )
    world = dist.get_world_size(group)
    counts = torch.tensor([value.numel() for value in families], dtype=torch.int64)
    sizes = torch.empty((world, len(families)), dtype=torch.int64)
    dist.all_gather(list(sizes.unbind()), counts, group=group)
    maximum = int(sizes.sum(dim=1).max())
    packed = torch.zeros(maximum, dtype=torch.int64)
    offset = 0
    for family in families:
        packed[offset : offset + family.numel()].copy_(family)
        offset += family.numel()
    received = torch.empty((world, maximum), dtype=torch.int64)
    dist.all_gather(list(received.unbind()), packed, group=group)
    requests = []
    for rank, lengths in enumerate(sizes.tolist()):
        offset = 0
        request = []
        for count in lengths:
            request.append(received[rank, offset : offset + count])
            offset += count
        requests.append(tuple(request))
    return requests
