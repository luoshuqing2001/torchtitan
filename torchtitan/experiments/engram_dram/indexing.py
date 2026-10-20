# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The original Engram hash, evaluated before CPU row ownership routing."""

import torch

from torchtitan.models.glimmer_dense.engram import (
    _BIGRAM_SALTS,
    _MIX_PREV,
    _MIX_PREV2,
    _TRIGRAM_SALTS,
    _earlier_ids,
    _segment_ids,
)


def hash_ids(tokens_T, positions_T, rows_per_head, eos_token_id):
    segments_T = _segment_ids(tokens_T, positions_T, fill_id=eos_token_id)
    ids_T = tokens_T.to(torch.int64)
    previous_T = _earlier_ids(ids_T, segments_T, shift=1, fill_id=eos_token_id)
    previous2_T = _earlier_ids(ids_T, segments_T, shift=2, fill_id=eos_token_id)
    bigram_TH = (
        ids_T[:, None]
        + previous_T[:, None] * _MIX_PREV
        + torch.tensor(_BIGRAM_SALTS, device=ids_T.device) * _MIX_PREV2
    )
    trigram_TH = (
        ids_T[:, None]
        + previous_T[:, None] * _MIX_PREV
        + previous2_T[:, None] * _MIX_PREV2
        + torch.tensor(_TRIGRAM_SALTS, device=ids_T.device)
    )
    rows_TH = torch.cat((bigram_TH, trigram_TH), dim=-1).remainder(rows_per_head)
    rows_TH += torch.arange(16, device=ids_T.device) * rows_per_head
    return rows_TH.flatten()
