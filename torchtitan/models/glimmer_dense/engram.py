# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""N-gram Engram for the dense Glimmer ladder.

Eight bigram heads and eight trigram heads are hashed into one table and
added, through a short convolution and a sigmoid gate, at decoder layer 2.
``rows_per_head`` is chosen so this module's total parameter count matches
the Gemma 4 per-layer module at the same ``ple_dim``. The value projection
widens with ``ple_dim`` so the matmul-active count matches too. A learned
scalar gate bias would make that count odd, so the gate offset stays at -4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.protocols.module import Module

_NGRAM_HEADS = 16
_CONV_KERNEL = 4
_CONV_DILATION = 3
_BIGRAM_SALTS = (1, 3, 5, 7, 9, 11, 13, 15)
_TRIGRAM_SALTS = (17, 19, 21, 23, 25, 27, 29, 31)
_MIX_PREV = 1_000_003
_MIX_PREV2 = 1_000_003 * 1_000_003
# Inject before this decoder layer, matching Qwen3.8-Flash-Next.
ENGRAM_LAYER_INDEX = 1


def ple_active_params(dim: int, n_layers: int, ple_dim: int) -> int:
    """Matmul parameters in the Gemma PLE module, excluding the lookup."""
    return ple_dim * (3 * n_layers * dim + 1) + n_layers * dim


def value_width(dim: int, n_layers: int, ple_dim: int) -> int:
    """Width of the Engram value bottleneck at this PLE tier.

    The two value matrices, the ``dim x dim`` gate, the convolution, and a
    ``ple_dim`` channel scale then have the same numel as ``ple_active_params``.
    """
    twice = 3 * n_layers * ple_dim + n_layers - dim - _CONV_KERNEL
    if twice <= 0 or twice % 2 != 0:
        raise ValueError(
            f"Engram value width is not a positive even split for "
            f"dim={dim}, n_layers={n_layers}, ple_dim={ple_dim}."
        )
    return twice // 2


def engram_active_params(dim: int, n_layers: int, ple_dim: int) -> int:
    """Matmul parameters in the Engram update, excluding the hash table."""
    width = value_width(dim, n_layers, ple_dim)
    return dim * dim + 2 * width * dim + ple_dim + _CONV_KERNEL * dim


def rows_per_head_matching_ple(
    dim: int,
    n_layers: int,
    *,
    vocab_size: int,
    ple_dim: int,
) -> tuple[int, int]:
    """Pick a row count whose Engram module matches the Gemma PLE module.

    Sixteen heads of width ``dim / 16`` make the rectangular table
    ``rows_per_head * dim`` parameters. The target table size is the PLE
    lookup, because the two updates already have the same numel. The second
    return value is the leftover ``target % dim``, stored beside the table
    so the totals still match when that division is not exact.
    """
    if dim % _NGRAM_HEADS != 0:
        raise ValueError(
            f"Engram concatenates {_NGRAM_HEADS} heads into dim, "
            f"but dim={dim} is not divisible by {_NGRAM_HEADS}."
        )
    if engram_active_params(dim, n_layers, ple_dim) != ple_active_params(
        dim, n_layers, ple_dim
    ):
        raise ValueError(
            "Engram active parameters drifted away from the PLE module."
        )
    target_table = vocab_size * n_layers * ple_dim
    rows, remainder = divmod(target_table, dim)
    if rows < 1:
        raise ValueError(
            f"Engram table would have {rows} rows per head for "
            f"dim={dim}, n_layers={n_layers}, ple_dim={ple_dim}."
        )
    return rows, remainder


class EngramTable(Module):
    """Hash-row embedding. Counted as a lookup, not a per-token matmul."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        rows_per_head: int
        eos_token_id: int
        remainder: int = 0

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.rows_per_head = config.rows_per_head
        self.eos_token_id = config.eos_token_id
        head_dim = config.dim // _NGRAM_HEADS
        rows = _NGRAM_HEADS * config.rows_per_head
        self.weight = torch.nn.Parameter(torch.empty(rows, head_dim))
        if config.remainder < 0 or config.remainder >= config.dim:
            raise ValueError(
                f"Engram table remainder must be in [0, dim), got {config.remainder}"
            )
        self.remainder_weight = (
            None
            if config.remainder == 0
            else torch.nn.Parameter(torch.empty(config.remainder))
        )

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Return one concatenated n-gram vector per token, shaped ``[T, dim]``."""
        segments = _segment_ids(tokens, positions, fill_id=self.eos_token_id)
        ids = tokens.to(dtype=torch.int64)
        prev = _earlier_ids(ids, segments, shift=1, fill_id=self.eos_token_id)
        prev2 = _earlier_ids(ids, segments, shift=2, fill_id=self.eos_token_id)
        device = ids.device
        bigram_salt = torch.tensor(_BIGRAM_SALTS, device=device, dtype=torch.int64)
        trigram_salt = torch.tensor(_TRIGRAM_SALTS, device=device, dtype=torch.int64)
        current = ids.unsqueeze(-1)
        bigram = current + prev.unsqueeze(-1) * _MIX_PREV + bigram_salt * _MIX_PREV2
        trigram = (
            current
            + prev.unsqueeze(-1) * _MIX_PREV
            + prev2.unsqueeze(-1) * _MIX_PREV2
            + trigram_salt
        )
        hashed = torch.cat((bigram, trigram), dim=-1)
        local_rows = torch.remainder(hashed, self.rows_per_head)
        head_offsets = torch.arange(_NGRAM_HEADS, device=device) * self.rows_per_head
        flat_index = (local_rows + head_offsets).reshape(-1)
        gathered = F.embedding(flat_index, self.weight).reshape(ids.shape[0], -1)
        if self.remainder_weight is None:
            return gathered
        width = self.remainder_weight.shape[0]
        head = gathered[:, :width] + self.remainder_weight
        return torch.cat((head, gathered[:, width:]), dim=-1)


class EngramUpdate(Module):
    """Project looked-up n-grams and add a gated residual.

    The value path is ``dim -> width -> dim``, with ``width`` from
    ``value_width``. A scale of length ``ple_dim`` multiplies the leading
    channels. Together with the gate matrix and the convolution, that is the
    same parameter count as the Gemma PLE projections at this tier.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        n_layers: int
        ple_dim: int
        eos_token_id: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.value_width = value_width(config.dim, config.n_layers, config.ple_dim)
        self.key_weight = torch.nn.Parameter(torch.empty(config.dim, config.dim))
        self.value_up = torch.nn.Parameter(
            torch.empty(self.value_width, config.dim)
        )
        self.value_down = torch.nn.Parameter(
            torch.empty(config.dim, self.value_width)
        )
        self.value_scale = torch.nn.Parameter(torch.empty(config.ple_dim))
        self.conv_weight = torch.nn.Parameter(
            torch.empty(_CONV_KERNEL, config.dim)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        embeddings: torch.Tensor,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        segments = _segment_ids(tokens, positions, fill_id=self.config.eos_token_id)
        keys = F.linear(embeddings, self.key_weight)
        values = F.linear(embeddings, self.value_up)
        values = F.linear(values, self.value_down)
        values = _segmented_short_conv(
            values,
            self.conv_weight,
            segments,
            dilation=_CONV_DILATION,
        )
        scale_width = self.value_scale.shape[0]
        scaled = values[:, :scale_width] * self.value_scale
        values = torch.cat((scaled, values[:, scale_width:]), dim=-1)
        scale = hidden.shape[-1] ** -0.5
        gate = torch.sigmoid((hidden * keys).sum(dim=-1, keepdim=True) * scale - 4)
        return hidden + gate * values


class NGramEngram(Module):
    """Hash packed token ids into an n-gram table and add a gated residual."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        n_layers: int
        ple_dim: int
        eos_token_id: int
        rows_per_head: int
        remainder: int = 0
        table: EngramTable.Config | None = None
        update: EngramUpdate.Config | None = None

        def __post_init__(self) -> None:
            if self.dim % _NGRAM_HEADS != 0:
                raise ValueError(
                    f"Engram concatenates {_NGRAM_HEADS} heads into dim, "
                    f"but dim={self.dim} is not divisible by {_NGRAM_HEADS}."
                )
            if self.rows_per_head <= 0:
                raise ValueError(
                    f"rows_per_head must be positive, got {self.rows_per_head}"
                )
            if self.table is None:
                self.table = EngramTable.Config(
                    dim=self.dim,
                    rows_per_head=self.rows_per_head,
                    eos_token_id=self.eos_token_id,
                    remainder=self.remainder,
                )
            if self.update is None:
                self.update = EngramUpdate.Config(
                    dim=self.dim,
                    n_layers=self.n_layers,
                    ple_dim=self.ple_dim,
                    eos_token_id=self.eos_token_id,
                )
            else:
                self.update.eos_token_id = self.eos_token_id
                self.update.dim = self.dim
                self.update.n_layers = self.n_layers
                self.update.ple_dim = self.ple_dim

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.dim // _NGRAM_HEADS
        self.rows_per_head = config.rows_per_head
        self.eos_token_id = config.eos_token_id
        assert config.table is not None and config.update is not None
        self.table = config.table.build()
        self.update = config.update.build()

    def forward(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Add the layer-2 Engram delta.

        ``hidden`` and ``tokens`` are packed sequences shaped ``[T, dim]`` and
        ``[T]``. A position of 0 starts a new document, matching the packing
        used by the ClimbMix loader. N-gram context does not cross that
        boundary or an EOS token.
        """
        if hidden.ndim != 2:
            raise ValueError(
                "Engram expects packed hidden states [T, dim], "
                f"got {tuple(hidden.shape)}"
            )
        if tokens.shape != positions.shape or tokens.shape[0] != hidden.shape[0]:
            raise ValueError(
                "Engram tokens, positions, and hidden sequence lengths must match, "
                f"got tokens {tuple(tokens.shape)}, positions {tuple(positions.shape)}, "
                f"hidden {tuple(hidden.shape)}"
            )
        embeddings = self.table(tokens, positions)
        return self.update(hidden, embeddings, tokens, positions)


def _segment_ids(
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    fill_id: int,
) -> torch.Tensor:
    """Document id per token. A new span starts at position 0 and after EOS."""
    ids = token_ids.to(dtype=torch.int64)
    after_eos = torch.zeros_like(ids, dtype=torch.bool)
    if ids.shape[0] > 1:
        after_eos[1:] = ids[:-1] == fill_id
    starts = (positions == 0) | after_eos
    return starts.to(dtype=torch.int64).cumsum(dim=0)


def _earlier_ids(
    token_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    shift: int,
    fill_id: int,
) -> torch.Tensor:
    """Token ``shift`` steps back, or ``fill_id`` across a document boundary."""
    out = torch.full_like(token_ids, fill_id)
    if shift <= 0 or shift >= token_ids.shape[0]:
        return token_ids if shift <= 0 else out
    same_doc = segment_ids[shift:] == segment_ids[:-shift]
    out[shift:] = torch.where(same_doc, token_ids[:-shift], out[shift:])
    return out


def _segmented_short_conv(
    values: torch.Tensor,
    weight: torch.Tensor,
    segments: torch.Tensor,
    *,
    dilation: int,
) -> torch.Tensor:
    """Causal depthwise conv that reads zeros across a document boundary."""
    out = torch.zeros_like(values)
    for tap in range(weight.shape[0]):
        shift = tap * dilation
        src = torch.zeros_like(values)
        if shift == 0:
            src = values
        elif shift < values.shape[0]:
            same_doc = segments[shift:] == segments[:-shift]
            src[shift:] = torch.where(
                same_doc.unsqueeze(-1),
                values[:-shift],
                src[shift:],
            )
        out = out + src * weight[tap]
    return out
