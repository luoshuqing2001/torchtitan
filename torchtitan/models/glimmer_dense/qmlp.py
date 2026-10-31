# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Nonlinear query projection for the FFN-free Glimmer ladder.

The query pathway of a sliding-window layer is a two-layer GELU MLP,
``q = W_down GELU(W_up x)``, as in scalable attention (SMLA/SGQA). Keys and
values keep a fused linear projection, so heads, head dim and the KV cache
are unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

# E[GELU(gZ)^2] = 1 for Z ~ N(0, 1), so a fan-in init with this gain keeps unit
# second moment through the activation (Monte Carlo, as in the SMLA paper).
GELU_GAIN = 1.469


def query_mlp_param_init(
    dim: int, intermediate: int
) -> tuple[dict[str, Callable], dict[str, Callable]]:
    """Activation-calibrated fan-in init for ``W_up`` and ``W_down``.

    ``W_up ~ N(0, g^2 / dim)`` puts unit-variance inputs (the attention norm
    output) at unit-variance pre-activations. ``W_down ~ N(0, 1 / intermediate)``
    then gives queries with unit second moment before q-norm. Because q-norm
    removes the query scale, ``W_down``'s scale only sets the effective step size.
    """
    up = {"weight": partial(nn.init.normal_, std=GELU_GAIN / math.sqrt(dim))}
    down = {"weight": partial(nn.init.normal_, std=1.0 / math.sqrt(intermediate))}
    return up, down


class QueryMLPKVLinear(Module):
    """Query from a GELU MLP, keys and values from one fused linear.

    Drop-in for ``QKVLinear``: returns ``(xq, xk, xv)`` shaped
    ``[T, heads, head_dim]``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        n_heads: int
        n_kv_heads: int
        q_up: Linear.Config
        q_down: Linear.Config
        # Fused [n_kv_heads, 2, head_dim] key/value rows.
        wkv: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.q_up = config.q_up.build()
        self.q_down = config.q_down.build()
        self.wkv = config.wkv.build()

    @spmd.local_map(
        out_types=(
            (
                {"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
                spmd.PartitionSpec(("dp", "cp"), "tp", None),
            ),
        )
        * 3
    )
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = x.shape[0]
        xq = self.q_down(F.gelu(self.q_up(x)))
        kv = self.wkv(x)
        with spmd.local():
            xq = xq.view(num_tokens, -1, self.head_dim)
            kv = kv.view(num_tokens, -1, 2, self.head_dim)
        xk, xv = kv.unbind(dim=-2)
        return xq.contiguous(), xk.contiguous(), xv.contiguous()
