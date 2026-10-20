# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma 4 per-layer embeddings for the dense Glimmer ladder.

Each token is looked up once in a table of shape
``[vocab, num_layers * ple_dim]`` and combined with a projection of the
token embedding. Decoder layers then gate their own slice back into the
residual after attention and the FFN. The table is a lookup, so it is not
part of the per-token matmul. ``ple_dim`` is Gemma 4's 256. ``PLE_DIM_TIERS``
also allows 512 and 1024, which keep the same math and widen every per-layer
slice. The vocabulary is this ladder's OLMo3 size, 100352, because the row
is indexed by the training token id. Parameters follow normal device placement
unless asynchronous DRAM training is explicitly enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torchtitan.models.common.embedding import Embedding
from torchtitan.protocols.module import Module

from .ple_dram import (
    RowPrefetch,
    Staged,
    dram_embedding,
    dram_linear,
    dram_rms_norm,
    prefetch_rows,
    stage,
)

_PLE_DIM = 256
# Official Gemma 4 width, then 2x and 4x. The same widths size Engram.
PLE_DIM_TIERS = (256, 512, 1024)
_NORM_EPS = 1e-6


class PerLayerContext(Module):
    """Project token embeddings into one ``ple_dim`` vector per layer."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_layers: int
        ple_dim: int = _PLE_DIM
        mixed_precision_backward: bool = False
        async_training: bool = False

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.ple_dim = config.ple_dim
        self.weight = torch.nn.Parameter(
            torch.empty(config.num_layers * config.ple_dim, config.dim)
        )
        self.norm_weight = torch.nn.Parameter(torch.empty(config.ple_dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return ``[tokens, num_layers * ple_dim]`` after the Gemma 4 scale."""
        if self.config.async_training:
            projected = dram_linear(
                self.weight,
                hidden,
                mixed_precision_backward=self.config.mixed_precision_backward,
            )
        else:
            projected = F.linear(hidden, self.weight.to(dtype=hidden.dtype))
        projected = projected * (self.config.dim**-0.5)
        projected = projected.unflatten(-1, (self.num_layers, self.ple_dim))
        if self.config.async_training:
            projected = dram_rms_norm(self.norm_weight, projected, _NORM_EPS)
        else:
            projected = F.rms_norm(
                projected,
                (self.ple_dim,),
                self.norm_weight.to(dtype=projected.dtype),
                _NORM_EPS,
            )
        return projected.flatten(-2, -1)


class PerLayerEmbedder(Module):
    """Build the per-layer inputs shared by every decoder block."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        vocab_size: int
        dim: int
        num_layers: int
        ple_dim: int = _PLE_DIM
        activation_checkpoint: bool = False
        async_training: bool = False
        table: Embedding.Config | None = None
        context: PerLayerContext.Config | None = None

        def __post_init__(self) -> None:
            if self.ple_dim <= 0:
                raise ValueError(f"ple_dim must be positive, got {self.ple_dim}")
            if self.num_layers <= 0:
                raise ValueError(f"num_layers must be positive, got {self.num_layers}")
            width = self.num_layers * self.ple_dim
            if self.table is None:
                self.table = Embedding.Config(
                    num_embeddings=self.vocab_size,
                    embedding_dim=width,
                )
            if self.context is None:
                self.context = PerLayerContext.Config(
                    dim=self.dim,
                    num_layers=self.num_layers,
                    ple_dim=self.ple_dim,
                )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.ple_dim = config.ple_dim
        assert config.table is not None and config.context is not None
        self.table = config.table.build()
        config.context.async_training = config.async_training
        self.context = config.context.build()

    def prefetch_rows(
        self, tokens: torch.Tensor, *, dtype: torch.dtype | None = None
    ) -> RowPrefetch:
        """Gather and transfer rows while the PLE context projection runs."""
        return prefetch_rows(self.table.weight, tokens, dtype=dtype)

    def forward(
        self,
        tokens: torch.Tensor,
        hidden: torch.Tensor,
        row_prefetch: RowPrefetch | None = None,
    ) -> torch.Tensor:
        """Return ``[tokens, num_layers, ple_dim]``.

        Token rows are scaled by ``sqrt(ple_dim)``. The projected embedding
        is RMS-normalized per layer. The two are averaged in quadrature,
        matching Gemma 4's ``(token + context) / sqrt(2)``. The context
        projection runs while the lookup rows are still being gathered.
        """
        if hidden.ndim != 2:
            raise ValueError(
                f"PLE expects packed hidden states [T, dim], got {tuple(hidden.shape)}"
            )
        if tokens.shape[0] != hidden.shape[0]:
            raise ValueError(
                "PLE tokens and hidden sequence lengths must match, "
                f"got tokens {tuple(tokens.shape)} and hidden {tuple(hidden.shape)}"
            )
        if self.config.async_training:
            if row_prefetch is None:
                row_prefetch = self.prefetch_rows(tokens, dtype=hidden.dtype)
            lookup = row_prefetch
        else:
            lookup = tokens
        if self.config.activation_checkpoint and torch.is_grad_enabled():
            return checkpoint(
                self._combine,
                hidden,
                lookup,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self._combine(hidden, lookup)

    def _combine(
        self, hidden: torch.Tensor, lookup: RowPrefetch | torch.Tensor
    ) -> torch.Tensor:
        context_rows = self.context(hidden)
        if self.config.async_training:
            token_rows = dram_embedding(self.table.weight, lookup, dtype=hidden.dtype)
        else:
            token_rows = self.table(lookup).to(dtype=hidden.dtype)
        # Embedding backward saves indices, not this newly allocated output.
        # Reuse it to avoid multiple token-by-all-layers activation buffers.
        token_rows.mul_(self.ple_dim**0.5)
        token_rows.add_(context_rows)
        token_rows.mul_(0.5**0.5)
        return token_rows.unflatten(-1, (self.num_layers, self.ple_dim))


class LayerPLE(Module):
    """Gate one layer's PLE slice into the residual stream."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        ple_dim: int = _PLE_DIM
        mixed_precision_backward: bool = False
        async_training: bool = False

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.gate_weight = torch.nn.Parameter(torch.empty(config.ple_dim, config.dim))
        self.proj_weight = torch.nn.Parameter(torch.empty(config.dim, config.ple_dim))
        self.norm_weight = torch.nn.Parameter(torch.empty(config.dim))

    def prefetch(self, like: torch.Tensor) -> dict[str, Staged]:
        """Queue copies of this layer's weights before attention and the FFN."""
        return {
            "gate": stage(self.gate_weight, like.device, like.dtype),
            "proj": stage(self.proj_weight, like.device, like.dtype),
            "norm": stage(self.norm_weight, like.device, like.dtype),
        }

    def forward(
        self,
        hidden: torch.Tensor,
        ple: torch.Tensor,
        *,
        staged: dict[str, Staged] | None = None,
    ) -> torch.Tensor:
        """Add the gated per-layer embedding after attention and the FFN."""
        if ple.shape[-1] != self.config.ple_dim:
            raise ValueError(
                f"PLE slice width must be {self.config.ple_dim}, got {ple.shape[-1]}"
            )
        if not self.config.async_training:
            gate_TP = F.linear(hidden, self.gate_weight.to(dtype=hidden.dtype))
            gated_TP = F.gelu(gate_TP) * ple.to(dtype=hidden.dtype)
            update_TD = F.linear(gated_TP, self.proj_weight.to(dtype=hidden.dtype))
            update_TD = F.rms_norm(
                update_TD,
                (self.config.dim,),
                self.norm_weight.to(dtype=update_TD.dtype),
                _NORM_EPS,
            )
            return hidden + update_TD
        if staged is None:
            staged = self.prefetch(hidden)
        if ple.dtype != hidden.dtype:
            ple = ple.to(dtype=hidden.dtype)
        gated = (
            F.gelu(
                dram_linear(
                    self.gate_weight,
                    hidden,
                    staged["gate"],
                    mixed_precision_backward=self.config.mixed_precision_backward,
                )
            )
            * ple
        )
        update = dram_linear(
            self.proj_weight,
            gated,
            staged["proj"],
            mixed_precision_backward=self.config.mixed_precision_backward,
        )
        update = dram_rms_norm(self.norm_weight, update, _NORM_EPS, staged["norm"])
        return hidden + update
