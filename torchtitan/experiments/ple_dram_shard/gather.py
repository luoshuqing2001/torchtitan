# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gather FP32 CPU rows directly into a packed transfer buffer.

Shape legend: R = owned rows, U = requested rows, D = concatenated layer width.
"""

import torch


def _gather_into(weight_RD, indices_U, packed_UD):
    packed_UD.copy_(weight_RD[indices_U])


_gather_into_compiled = torch.compile(_gather_into, fullgraph=True, dynamic=True)


def gather_packed(weight_RD, indices_U, dtype, *, pin_memory):
    packed_UD = torch.empty(
        (indices_U.numel(), weight_RD.shape[1]), dtype=dtype, pin_memory=pin_memory
    )
    # Dynamo specializes zero/one-sized dimensions; avoid separate graphs for
    # these small requests and retain the dynamic graph for ordinary batches.
    if indices_U.numel() <= 1:
        packed_UD.copy_(weight_RD.index_select(0, indices_U))
    else:
        _gather_into_compiled(weight_RD, indices_U, packed_UD)
    return packed_UD
