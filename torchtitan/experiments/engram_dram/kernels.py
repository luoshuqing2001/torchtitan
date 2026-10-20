# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Copy unique owned-row updates without deterministic duplicate-index sorting."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["count"])
def _scatter_rows(
    destination,
    indices,
    values,
    count,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = offsets // width
    columns = offsets % width
    row_ids = tl.load(indices + rows, rows < count, other=0)
    data = tl.load(values + offsets, offsets < count * width, other=0)
    tl.store(destination + row_ids * width + columns, data, offsets < count * width)


def scatter_rows(destination_RD, indices_U, values_UD):
    """Indices must be unique; metadata construction guarantees this invariant."""
    if (
        destination_RD.device.type != "cuda"
        or indices_U.device != destination_RD.device
        or values_UD.device != destination_RD.device
        or indices_U.dtype != torch.int64
        or values_UD.dtype != destination_RD.dtype
        or values_UD.shape != (indices_U.numel(), destination_RD.shape[1])
        or not all(
            value.is_contiguous() for value in (destination_RD, indices_U, values_UD)
        )
    ):
        raise ValueError(
            "Cache scatter requires matching contiguous CUDA rows and int64 indices"
        )
    if indices_U.numel():
        _scatter_rows[(triton.cdiv(values_UD.numel(), 1024),)](
            destination_RD,
            indices_U,
            values_UD,
            indices_U.numel(),
            destination_RD.shape[1],
            BLOCK=1024,
        )
