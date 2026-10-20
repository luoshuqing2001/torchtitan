# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exchange two hash families instead of sixteen expanded head requests.

Shape legend: T = tokens, U = unique hash bases, H = heads, D = row width, K = hash families.
"""

import torch
import torch.distributed as dist

from torchtitan.experiments.ple_dram_shard.gather import gather_packed
from torchtitan.experiments.ple_dram_shard.model import _fetch_owned
from torchtitan.models.glimmer_dense.engram import (
    _MIX_PREV,
    _MIX_PREV2,
    _earlier_ids,
    _segment_ids,
)

from .indexing import hash_ids


def _unique(values, rows, *, bounded, return_inverse):
    if bounded and values.numel() >= rows // 128:
        return torch.ops.torchtitan_ple_cpu.unique_bounded(values, rows, return_inverse)
    if return_inverse:
        return torch.unique(values, return_inverse=True)
    return torch.unique(values), None


def families(tokens_T, positions_T, rows, eos, *, bounded=False):
    # Translation modulo rows is valid only without intermediate int64 overflow.
    limit = (torch.iinfo(torch.int64).max - 31 * _MIX_PREV2) // (
        1 + _MIX_PREV + _MIX_PREV2
    )
    if (
        not tokens_T.numel()
        or min(int(tokens_T.min()), eos) < 0
        or max(int(tokens_T.max()), eos) > limit
    ):
        return None
    segments_T = _segment_ids(tokens_T, positions_T, fill_id=eos)
    ids_T = tokens_T.to(torch.int64)
    previous_T = _earlier_ids(ids_T, segments_T, shift=1, fill_id=eos)
    previous2_T = _earlier_ids(ids_T, segments_T, shift=2, fill_id=eos)
    bigram_T = ids_T + previous_T * _MIX_PREV
    trigram_T = bigram_T + previous2_T * _MIX_PREV2
    return tuple(
        _unique(key.remainder(rows), rows, bounded=bounded, return_inverse=True)
        for key in (bigram_T, trigram_T)
    )


def _shift(head, rows):
    return ((2 * head + 1) * (_MIX_PREV2 if head < 8 else 1)) % rows


def _cut(base_U, head, rows):
    return int(torch.searchsorted(base_U, rows - _shift(head, rows)))


def _rotate(values_U, cut):
    return torch.cat((values_U[cut:], values_U[:cut]))


def _rotate_inverse(inverse_T, cut, size):
    shifted_T = inverse_T - cut
    return torch.where(shifted_T < 0, shifted_T + size, shifted_T)


def expand(bases, rows, *, first=0, last=16):
    result = []
    for head in range(first, last):
        base_U = bases[head // 8]
        cut = _cut(base_U, head, rows)
        shift = _shift(head, rows)
        result.append(
            torch.cat((base_U[cut:] + shift - rows, base_U[:cut] + shift)) + head * rows
        )
    return torch.cat(result)


def inverse(family, rows):
    result_TH = torch.empty(family[0][1].numel(), 16, dtype=torch.int64)
    prefix = 0
    for head in range(16):
        base_U, inverse_T = family[head // 8]
        result_TH[:, head] = (
            _rotate_inverse(inverse_T, _cut(base_U, head, rows), base_U.numel())
            + prefix
        )
        prefix += base_U.numel()
    return result_TH.flatten()


def build_inverse(inverse_KT, cuts_H, sizes_H, prefixes_H):
    heads_H = torch.arange(16, device=inverse_KT.device) // 8
    shifted_TH = inverse_KT[heads_H].transpose(0, 1) - cuts_H
    return (
        torch.where(shifted_TH < 0, shifted_TH + sizes_H, shifted_TH) + prefixes_H
    ).flatten()


_device_inverse = torch.compile(build_inverse, fullgraph=True, dynamic=True)


def _inverse_packet(family, rows):
    cuts = []
    sizes = []
    prefixes = []
    prefix = 0
    for head in range(16):
        base, _ = family[head // 8]
        cuts.append(_cut(base, head, rows))
        sizes.append(base.numel())
        prefixes.append(prefix)
        prefix += base.numel()
    return (
        torch.stack([item[1] for item in family]),
        torch.tensor(cuts),
        torch.tensor(sizes),
        torch.tensor(prefixes),
    )


def fetch(
    weight_RD,
    tokens_T,
    positions_T,
    device,
    stream,
    dtype,
    ownership,
    rows,
    eos,
    fused,
    bounded=False,
    gpu_indices=False,
    tensor_exchange=False,
):
    family = families(tokens_T, positions_T, rows, eos, bounded=bounded)
    world = dist.get_world_size(ownership.group)
    eligible = torch.tensor(
        int(family is not None and 16 % world == 0), dtype=torch.int32
    )
    dist.all_reduce(eligible, op=dist.ReduceOp.MIN, group=ownership.group)
    if not bool(eligible):
        ids = hash_ids(tokens_T, positions_T, rows, eos)
        return _fetch_owned(
            weight_RD, ids, None, device, stream, dtype, ownership, fused
        )
    bases = tuple(item[0] for item in family)
    if tensor_exchange:
        from torchtitan.experiments.ple_dram_shard.request_exchange import (
            exchange_indices,
        )

        requests = exchange_indices(bases, ownership.group)
    else:
        requests = [None] * world
        dist.all_gather_object(requests, bases, group=ownership.group)
    global_bases = tuple(
        _unique(
            torch.cat([request[kind] for request in requests]),
            rows,
            bounded=bounded,
            return_inverse=False,
        )[0]
        for kind in range(2)
    )
    rank_ids = [expand(request, rows) for request in requests]
    unique_U = rank_ids[dist.get_rank(ownership.group)]
    packet = (
        _inverse_packet(family, rows) if gpu_indices and device.type == "cuda" else None
    )
    inverse_T = inverse(family, rows) if packet is None else None
    heads = 16 // world
    first = dist.get_rank(ownership.group) * heads
    local_U = expand(global_bases, rows, first=first, last=first + heads)
    if getattr(ownership, "metadata_only", False):
        packed_UD = weight_RD.new_empty((0, weight_RD.shape[1]), dtype=dtype)
    elif fused:
        packed_UD = gather_packed(
            weight_RD,
            local_U - ownership.offset,
            dtype,
            pin_memory=device.type == "cuda",
        )
    else:
        packed_UD = weight_RD.index_select(0, local_U - ownership.offset).to(dtype)
    owner_counts = [
        sum(
            global_bases[h // 8].numel()
            for h in range(rank * heads, (rank + 1) * heads)
        )
        for rank in range(world)
    ]
    local_counts = [
        sum(bases[h // 8].numel() for h in range(rank * heads, (rank + 1) * heads))
        for rank in range(world)
    ]
    mappings = []
    positions = tuple(
        torch.searchsorted(global_bases[kind], bases[kind]) for kind in range(2)
    )
    prefix = 0
    for head in range(16):
        kind = head // 8
        cut = _cut(bases[kind], head, rows)
        global_cut = _cut(global_bases[kind], head, rows)
        mapping = _rotate_inverse(
            _rotate(positions[kind], cut), global_cut, global_bases[kind].numel()
        )
        mappings.append(mapping + prefix)
        prefix += global_bases[kind].numel()
    mapping_U = torch.cat(mappings)
    event = None
    if device.type == "cuda":
        packed_UD = packed_UD if packed_UD.is_pinned() else packed_UD.pin_memory()
        with torch.cuda.device(device), torch.cuda.stream(stream):
            packed_UD = packed_UD.to(device, non_blocking=True)
            if packet is None:
                inverse_T = inverse_T.pin_memory().to(device, non_blocking=True)
            else:
                packet = tuple(
                    value.pin_memory().to(device, non_blocking=True) for value in packet
                )
                inverse_T = _device_inverse(*packet)
            unique_U = unique_U.pin_memory().to(device, non_blocking=True)
            mapping_U = mapping_U.pin_memory().to(device, non_blocking=True)
            event = stream.record_event()
    if getattr(ownership, "prepare_routing", False):
        from torchtitan.experiments.ple_dram_shard.routing import (
            PreparedRequests,
            prepare_routing,
        )
        from torchtitan.models.glimmer_dense.ple_dram import _POOL

        rank_ids = PreparedRequests(rank_ids)
        rank_ids.routing_plan = _POOL.submit(
            prepare_routing, rank_ids, ownership, local_U
        )
    if getattr(ownership, "early_gradient", False):
        from torchtitan.experiments.ple_dram_shard.model import _allocate_gradient

        rank_ids.gradient_allocation = _POOL.submit(
            _allocate_gradient, ownership, device
        )
    preparation = getattr(ownership, "prepared_update", None)
    if preparation is not None:
        cache = getattr(ownership, "compute_cache", None)
        layout = (
            (expand(global_bases, rows), owner_counts)
            if cache is not None and cache.double_buffer and cache.preload_enabled
            else None
        )
        preparation.prepare(local_U - ownership.offset, cache_layout=layout)
    return (
        packed_UD,
        unique_U,
        inverse_T,
        None,
        rank_ids,
        owner_counts,
        local_counts,
        mapping_U,
        event,
    )
