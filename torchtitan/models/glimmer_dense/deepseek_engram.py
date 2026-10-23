# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek Engram (arXiv:2601.07372) for the dense Glimmer ladder.

Follows the paper and the reference ``engram_demo_v1.py`` in
github.com/deepseek-ai/Engram, without the hyper-connection branches:

- Token ids are first collapsed by the tokenizer compression map (NFKC, NFD,
  accent strip, lowercase, whitespace folding).
- At each position, the suffix 2-gram and 3-gram of compressed ids are mixed
  with a multiplicative-XOR hash. Each order has 8 heads, and each head takes
  that mix modulo its own prime table size, so heads collide independently.
  Multipliers and primes differ per Engram layer.
- The 16 retrieved head vectors are concatenated into ``e``. With the hidden
  state ``h``, ``gate = sigmoid(signed_sqrt(RMSNorm(h) . RMSNorm(W_K e) /
  sqrt(dim)))`` and ``v = gate * W_V e``. The layer adds
  ``v + SiLU(ShortConv(RMSNorm(v)))`` to the residual stream. The causal
  depthwise convolution has kernel 4 and dilation 3 (the largest n-gram
  order), and starts at zero.
- Modules sit before decoder layers 2 and ``n_layers / 2``, the placements the
  paper uses for its 12-layer (2, 6) and 30-layer (2, 15) models.

N-gram context and the convolution never cross a document boundary (position
0) or an EOS token; a missing context token is the compressed EOS id.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from torchtitan.protocols.module import Module

NGRAM_ORDERS = (2, 3)
HEADS_PER_ORDER = 8
NUM_HEADS = len(NGRAM_ORDERS) * HEADS_PER_ORDER
CONV_KERNEL = 4
CONV_DILATION = max(NGRAM_ORDERS)
_NORM_EPS = 1e-5
# Reference seeding: layer multipliers come from seed + _LAYER_SEED_PRIME * layer.
_LAYER_SEED_PRIME = 10007


def engram_layer_indices(n_layers: int) -> tuple[int, int]:
    """Decoder indices of the two Engram modules (layers 2 and n_layers / 2)."""
    second = n_layers // 2 - 1
    if second <= 1:
        raise ValueError(f"Engram needs at least 6 layers, got {n_layers}")
    return (1, second)


@functools.lru_cache(maxsize=4)
def compression_lookup(tokenizer_path: str | None, vocab_size: int) -> np.ndarray:
    """Map each token id to its compressed id, as the reference tokenizer does.

    Ids past the tokenizer (embedding padding) map to fresh compressed ids.
    Without a tokenizer the map is the identity.
    """
    if tokenizer_path is None:
        return np.arange(vocab_size, dtype=np.int64)
    from tokenizers import normalizers, Regex, Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    known = tokenizer.get_vocab_size(with_added_tokens=True)
    if known > vocab_size:
        raise ValueError(f"tokenizer has {known} ids but vocab_size is {vocab_size}")
    key_to_new: dict[str, int] = {}
    lookup = np.empty(vocab_size, dtype=np.int64)
    for token_id in range(known):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = tokenizer.id_to_token(token_id) or text
        else:
            key = normalizer.normalize_str(text) or text
        lookup[token_id] = key_to_new.setdefault(key, len(key_to_new))
    lookup[known:] = np.arange(len(key_to_new), len(key_to_new) + vocab_size - known)
    return lookup


def _next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while True:
        if candidate not in seen and _is_prime(candidate):
            return candidate
        candidate += 1


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for d in range(3, math.isqrt(n) + 1, 2):
        if n % d == 0:
            return False
    return True


def head_table_sizes(
    rows_per_head: int, num_layers: int
) -> tuple[tuple[int, ...], ...]:
    """Distinct prime row counts, one per head and layer, as in the reference.

    Primes are searched upward from ``rows_per_head - 1`` and never reused, so
    every head of every layer has its own modulus.
    """
    seen: set[int] = set()
    sizes = []
    for _ in range(num_layers):
        layer_sizes = []
        for _ in NGRAM_ORDERS:
            start = rows_per_head - 1
            for _ in range(HEADS_PER_ORDER):
                prime = _next_prime(start, seen)
                seen.add(prime)
                layer_sizes.append(prime)
                start = prime
        sizes.append(tuple(layer_sizes))
    return tuple(sizes)


def layer_multipliers(
    layer_id: int, compressed_vocab: int, seed: int = 0
) -> tuple[int, ...]:
    """Odd hash multipliers for one layer, bounded so ``id * m`` fits in int64."""
    half_bound = max(1, (np.iinfo(np.int64).max // compressed_vocab) // 2)
    rng = np.random.default_rng(seed + _LAYER_SEED_PRIME * layer_id)
    r = rng.integers(low=0, high=half_bound, size=(max(NGRAM_ORDERS),), dtype=np.int64)
    return tuple(int(x) * 2 + 1 for x in r)


def active_params_per_layer(dim: int, memory_dim: int) -> int:
    """W_K, W_V, the query/key/conv RMSNorm gains, and the depthwise conv."""
    return 2 * dim * memory_dim + 3 * dim + CONV_KERNEL * dim


class HashedTable(Module):
    """All heads' rows of one Engram layer, stacked; a lookup, not a matmul."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_rows: int
        head_dim: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(config.num_rows, config.head_dim))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """Concatenate each token's head vectors: ``[T, heads]`` -> ``[T, heads * head_dim]``."""
        return F.embedding(rows.reshape(-1), self.weight).reshape(rows.shape[0], -1)


class EngramLayer(Module):
    """One Engram module: multi-head hashed n-gram lookup and gated fusion."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        memory_dim: int
        layer_id: int
        table_sizes: tuple[int, ...]
        multipliers: tuple[int, ...]
        pad_id: int
        eos_token_id: int
        vocab_size: int
        tokenizer_path: str | None = None
        table: HashedTable.Config | None = None

        def __post_init__(self) -> None:
            if self.table is None:
                self.table = HashedTable.Config(
                    num_rows=sum(self.table_sizes),
                    head_dim=self.memory_dim // NUM_HEADS,
                )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.memory_dim % NUM_HEADS != 0:
            raise ValueError(
                f"memory_dim={config.memory_dim} must split into {NUM_HEADS} heads"
            )
        if len(config.table_sizes) != NUM_HEADS:
            raise ValueError(f"expected {NUM_HEADS} table sizes")
        self.config = config
        self.head_dim = config.memory_dim // NUM_HEADS
        offsets = np.concatenate(([0], np.cumsum(config.table_sizes[:-1])))
        self._offsets = tuple(int(x) for x in offsets)
        self._lookup = compression_lookup(config.tokenizer_path, config.vocab_size)
        self._device_constants: dict[torch.device, tuple[torch.Tensor, ...]] = {}
        assert config.table is not None
        self.table: HashedTable = config.table.build()
        self.key_weight = torch.nn.Parameter(torch.empty(config.dim, config.memory_dim))
        self.value_weight = torch.nn.Parameter(
            torch.empty(config.dim, config.memory_dim)
        )
        self.query_norm_weight = torch.nn.Parameter(torch.empty(config.dim))
        self.key_norm_weight = torch.nn.Parameter(torch.empty(config.dim))
        self.conv_norm_weight = torch.nn.Parameter(torch.empty(config.dim))
        self.conv_weight = torch.nn.Parameter(torch.empty(CONV_KERNEL, config.dim))

    def _constants(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        cached = self._device_constants.get(device)
        if cached is None:
            cached = (
                torch.from_numpy(self._lookup).to(device),
                torch.tensor(self.config.multipliers, dtype=torch.int64, device=device),
                torch.tensor(self.config.table_sizes, dtype=torch.int64, device=device),
                torch.tensor(self._offsets, dtype=torch.int64, device=device),
            )
            self._device_constants[device] = cached
        return cached

    def hash_indices(
        self, tokens: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Row of each head's table for each token, shaped ``[T, NUM_HEADS]``."""
        lookup, multipliers, moduli, offsets = self._constants(tokens.device)
        ids = lookup[tokens.to(torch.int64)]
        segments = _segment_ids(
            tokens, positions, eos_token_id=self.config.eos_token_id
        )
        shifted = [ids] + [
            _earlier_ids(ids, segments, shift=k, fill_id=self.config.pad_id)
            for k in range(1, max(NGRAM_ORDERS))
        ]
        mixes = []
        mix = shifted[0] * multipliers[0]
        for k in range(1, max(NGRAM_ORDERS)):
            mix = torch.bitwise_xor(mix, shifted[k] * multipliers[k])
            if k + 1 in NGRAM_ORDERS:
                mixes.append(mix)
        per_order = torch.stack(mixes, dim=-1).repeat_interleave(
            HEADS_PER_ORDER, dim=-1
        )
        return torch.remainder(per_order, moduli) + offsets

    def forward(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``hidden`` plus this layer's Engram output; inputs are packed."""
        if hidden.ndim != 2:
            raise ValueError(
                f"Engram expects packed hidden states [T, dim], got {tuple(hidden.shape)}"
            )
        rows = self.hash_indices(tokens, positions)
        embeddings = self.table(rows).to(hidden.dtype)
        key = F.rms_norm(
            F.linear(embeddings, self.key_weight),
            (hidden.shape[-1],),
            self.key_norm_weight,
            _NORM_EPS,
        )
        query = F.rms_norm(
            hidden, (hidden.shape[-1],), self.query_norm_weight, _NORM_EPS
        )
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(hidden.shape[-1])
        gate = torch.sigmoid(gate.abs().clamp_min(1e-6).sqrt() * gate.sign())
        value = gate * F.linear(embeddings, self.value_weight)
        segments = _segment_ids(
            tokens, positions, eos_token_id=self.config.eos_token_id
        )
        conv_in = F.rms_norm(
            value, (value.shape[-1],), self.conv_norm_weight, _NORM_EPS
        )
        conv = _segmented_short_conv(
            conv_in, self.conv_weight, segments, dilation=CONV_DILATION
        )
        return hidden + value + F.silu(conv)


class DeepSeekEngram(Module):
    """The Engram modules of one model, keyed by the decoder layer they precede."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layers: dict[int, EngramLayer.Config]

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleDict(
            {str(index): layer.build() for index, layer in config.layers.items()}
        )

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(int(key) for key in self.layers)

    def tables(self) -> tuple[HashedTable, ...]:
        tables = []
        for layer in self.layers.values():
            assert isinstance(layer, EngramLayer)
            tables.append(layer.table)
        return tuple(tables)

    def forward(
        self,
        index: int,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers[str(index)](hidden, tokens, positions)


def _segment_ids(
    token_ids: torch.Tensor, positions: torch.Tensor, *, eos_token_id: int
) -> torch.Tensor:
    """Document id per token. A new span starts at position 0 and after EOS."""
    after_eos = torch.zeros_like(token_ids, dtype=torch.bool)
    if token_ids.shape[0] > 1:
        after_eos[1:] = token_ids[:-1] == eos_token_id
    starts = (positions == 0) | after_eos
    return starts.to(dtype=torch.int64).cumsum(dim=0)


def _earlier_ids(
    ids: torch.Tensor, segments: torch.Tensor, *, shift: int, fill_id: int
) -> torch.Tensor:
    """Id ``shift`` steps back, or ``fill_id`` across a document boundary."""
    out = torch.full_like(ids, fill_id)
    if shift >= ids.shape[0]:
        return out
    same_doc = segments[shift:] == segments[:-shift]
    out[shift:] = torch.where(same_doc, ids[:-shift], out[shift:])
    return out


def _segmented_short_conv(
    values: torch.Tensor,
    weight: torch.Tensor,
    segments: torch.Tensor,
    *,
    dilation: int,
) -> torch.Tensor:
    """Causal depthwise conv that reads zeros across a document boundary."""
    out = values * weight[0]
    for tap in range(1, weight.shape[0]):
        shift = tap * dilation
        if shift >= values.shape[0]:
            break
        same_doc = (segments[shift:] == segments[:-shift]).unsqueeze(-1)
        src = torch.zeros_like(values)
        src[shift:] = torch.where(same_doc, values[:-shift], src[shift:])
        out = out + src * weight[tap]
    return out
