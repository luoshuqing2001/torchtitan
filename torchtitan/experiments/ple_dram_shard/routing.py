# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Send ordered FP32 table gradients only to their vocabulary-row owner.

Shape legend: U = owned union rows, D = table width, W = data-parallel ranks.
"""

import torch
import torch.distributed as dist

from torchtitan.models.glimmer_dense import ple_dram as dram


class PreparedRequests(list):
    """Per-request indices and a scratch-only backward routing future."""

    routing_plan = None


def _layout(rank_ids, ownership, local_U=None):
    world = len(rank_ids)
    rank = dist.get_rank(ownership.group)
    width = ownership.width
    boundaries = torch.arange(world + 1, dtype=torch.int64) * ownership.num_rows
    cuts = [torch.searchsorted(ids, boundaries).tolist() for ids in rank_ids]
    local_ids = [
        ids[positions[rank] : positions[rank + 1]]
        for ids, positions in zip(rank_ids, cuts)
    ]
    if local_U is None:
        local_U = torch.unique(torch.cat(local_ids))
    send = [(b - a) * width for a, b in zip(cuts[rank][:-1], cuts[rank][1:])]
    receive = [ids.numel() * width for ids in local_ids]
    return local_U, send, receive, local_ids


def _mapping(local_U, local_ids, received_rows, vocab_size):
    mappings = []
    cursor = 0
    for ids in local_ids:
        padded = torch.cat((ids, ids.new_tensor([vocab_size])))
        positions = torch.searchsorted(padded, local_U)
        mappings.append(
            torch.where(padded[positions] == local_U, positions + cursor, received_rows)
        )
        cursor += ids.numel()
    return torch.stack(mappings)


def prepare_routing(rank_ids, ownership, local_U=None, *, pin_memory=True):
    local_U, send, receive, local_ids = _layout(rank_ids, ownership, local_U)
    mapping = _mapping(
        local_U, local_ids, sum(receive) // ownership.width, ownership.vocab_size
    )
    if pin_memory:
        mapping = mapping.pin_memory()
    return local_U, send, receive, mapping


def owned_gradient(unique_U, gradient_UD, rank_ids, ownership):
    from .kernels import merge_owned
    from .model import _allocate_gradient

    future = getattr(rank_ids, "gradient_allocation", None)
    if future is None:
        future = dram._POOL.submit(_allocate_gradient, ownership, gradient_UD.device)
    width = ownership.width
    plan = getattr(rank_ids, "routing_plan", None)
    if plan is None:
        local_U, send, receive, local_ids = _layout(rank_ids, ownership)
    else:
        local_U, send, receive, mapping_WU = plan.result()
    received_rows = sum(receive) // width
    values_UD = gradient_UD.new_empty((received_rows + 1, width))
    values_UD[-1].zero_()
    dist.all_to_all_single(
        values_UD[:-1].view(-1),
        gradient_UD.contiguous().view(-1),
        output_split_sizes=receive,
        input_split_sizes=send,
    )
    if plan is None:
        mapping_WU = _mapping(
            local_U, local_ids, received_rows, ownership.vocab_size
        ).pin_memory()
    mapping_WU = mapping_WU.to(gradient_UD.device, non_blocking=True)
    merged_UD = merge_owned(values_UD, mapping_WU)
    transport = ownership.gradient_transport
    if transport is not None and transport.can_defer():
        transport.schedule(future, merged_UD, local_U)
        return None
    values_cpu_UD = dram._cpu_grad(merged_UD, torch.float32)
    full_RD = future.result()
    full_RD.index_add_(0, local_U - ownership.offset, values_cpu_UD)
    return full_RD
