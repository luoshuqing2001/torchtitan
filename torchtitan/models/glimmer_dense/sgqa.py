# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SGQA-style sliding-window attention projections with decoupled partial RoPE.

Each query/key head of ``head_dim`` splits into ``head_dim - rotary_dim``
non-positional dimensions and ``rotary_dim`` rotated dimensions (the last ones).
The rotated key part is a single head shared by every KV group, as in decoupled
RoPE (DeepSeek-V2) and SGQA. Values stay ``head_dim`` wide and linear. The
query is linear, or a query MLP optionally added to a linear query.
"""

from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F

from torchtitan.models.common import ComplexRoPE
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module


class PartialRoPE(ComplexRoPE):
    """Rotate only the last ``config.dim`` dimensions of each head."""

    @dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        pass

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        rotary = self.config.dim
        q_pass, q_rot = query[..., :-rotary], query[..., -rotary:].contiguous()
        if key is None:
            q_rot = super().forward(q_rot, None, positions, inverse=inverse)
            return torch.cat([q_pass, q_rot], dim=-1)
        k_pass, k_rot = key[..., :-rotary], key[..., -rotary:].contiguous()
        q_rot, k_rot = super().forward(q_rot, k_rot, positions, inverse=inverse)
        return torch.cat([q_pass, q_rot], dim=-1), torch.cat([k_pass, k_rot], dim=-1)


class SGQAProjection(Module):
    """Q/K/V for one SGQA-style layer; drop-in for ``QKVLinear``.

    ``q = wq(x)`` and/or ``q = q_down(GELU(q_up(x)))`` (summed when both are
    present). ``k_g = [wk(x)_g ; wk_rope(x)]`` for each KV group ``g``.
    Returns ``(xq, xk, xv)`` shaped ``[T, heads, head_dim]``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        n_heads: int
        n_kv_heads: int
        rotary_dim: int
        wk: Linear.Config  # dim -> n_kv_heads * (head_dim - rotary_dim)
        wk_rope: Linear.Config  # dim -> rotary_dim, shared by all KV groups
        wv: Linear.Config  # dim -> n_kv_heads * head_dim
        wq: Linear.Config | None = None
        q_up: Linear.Config | None = None
        q_down: Linear.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        if config.wq is None and config.q_up is None:
            raise ValueError("SGQAProjection needs wq, a query MLP, or both")
        if (config.q_up is None) != (config.q_down is None):
            raise ValueError("q_up and q_down must be set together")
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.wq = None if config.wq is None else config.wq.build()
        self.q_up = None if config.q_up is None else config.q_up.build()
        self.q_down = None if config.q_down is None else config.q_down.build()
        self.wk = config.wk.build()
        self.wk_rope = config.wk_rope.build()
        self.wv = config.wv.build()

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
        xq = None if self.wq is None else self.wq(x)
        if self.q_up is not None:
            mlp = self.q_down(F.gelu(self.q_up(x)))
            xq = mlp if xq is None else xq + mlp
        k_nope = self.wk(x)
        k_rope = self.wk_rope(x)
        xv = self.wv(x)
        with spmd.local():
            xq = xq.view(num_tokens, -1, self.head_dim)
            k_nope = k_nope.view(num_tokens, -1, self.head_dim - self.rotary_dim)
            n_kv = k_nope.shape[1]
            k_rope = k_rope.view(num_tokens, 1, self.rotary_dim).expand(
                num_tokens, n_kv, self.rotary_dim
            )
            xk = torch.cat([k_nope, k_rope], dim=-1)
            xv = xv.view(num_tokens, -1, self.head_dim)
        return xq.contiguous(), xk.contiguous(), xv.contiguous()
