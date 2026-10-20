# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preserve native norm ordering while staging CPU scalars asynchronously."""

import torch
from torch.distributed.tensor import DTensor
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype


@torch.no_grad()
def total_norm_async(gradients):
    if not gradients or gradients[0].device.type != "cuda":
        return torch.nn.utils.get_total_norm(gradients, 2.0, foreach=True)
    device = gradients[0].device
    groups = _group_tensors_by_device_and_dtype([gradients])
    norms = []
    for [values], _ in groups.values():
        for norm in torch._foreach_norm(values, 2.0):
            if norm.device.type == "cpu":
                norm = norm.pin_memory().to(device, non_blocking=True)
            else:
                norm = norm.to(device)
            norms.append(norm)
    result = torch.linalg.vector_norm(torch.stack(norms), 2.0)
    return result.full_tensor() if isinstance(result, DTensor) else result
