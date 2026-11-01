# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Routed SwiGLU FFN with an optional separately routed gate pool.

The standard layer (``split=False``) is a token-choice MoE: one router picks
``top_k`` of ``num_experts`` experts, expert ``e`` is
``(W_gate[e], W_up[e], W_down[e])``, and

    y = sum_e s_e W_down[e] (SiLU(W_gate[e] x) * W_up[e] x).

The split layer (``split=True``) keeps the up/down pairs as one pool and the
gate matrices as a second pool, each with its own router and top-k. The
composite expert ``(i, j)`` is ``(W_gate[j], W_up[i], W_down[i])`` with weight
``s_i * t_j``, so there are ``num_experts * num_gate_experts`` experts and
``top_k * gate_top_k`` of them active per token. ``W_down`` and the product
with ``W_up x`` are linear in the gate vector, so the sum over all active
pairs is

    y = sum_i s_i W_down[i] (G * W_up[i] x),  G = sum_j t_j SiLU(W_gate[j] x),

which costs the same FLOPs as the standard layer with the same pool sizes.

Both layers use softmax router scores renormalized over the top-k, and the
auxiliary-loss-free expert bias of DeepSeek-V3 (one bias per pool), updated
before every optimizer step by ``register_factored_moe_balancing_hook``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_remat as remat
from torch import nn

from torchtitan.ops.scatter_add import deterministic_scatter_add
from torchtitan.protocols.module import Module


def _grouped_mm(
    x_RI: torch.Tensor, weight_EOI: torch.Tensor, offsets_E: torch.Tensor
) -> torch.Tensor:
    """``x_RI @ weight_EOI[e].T`` for the rows of each expert ``e``."""
    if x_RI.is_cuda:
        return torch._grouped_mm(
            x_RI.bfloat16(),
            weight_EOI.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
    outputs, start = [], 0
    for expert, end in enumerate(offsets_E.tolist()):
        outputs.append(x_RI[start:end] @ weight_EOI[expert].to(x_RI.dtype).T)
        start = end
    return torch.cat(outputs)


def _scatter_rows(
    out_TO: torch.Tensor, token_R: torch.Tensor, rows_RO: torch.Tensor
) -> torch.Tensor:
    index = token_R.reshape(-1, 1).expand(-1, rows_RO.shape[-1])
    return deterministic_scatter_add(out_TO, index, rows_RO)


class FactoredMoE(Module):
    """Token-choice SwiGLU MoE; ``split`` routes the gate rows separately."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int
        top_k: int
        split: bool = False
        # Gate pool of the split layer; default to the up/down pool's sizes.
        num_gate_experts: int | None = None
        gate_top_k: int | None = None
        load_balance_coeff: float = 1e-3

        def __post_init__(self) -> None:
            if not self.split and (
                self.num_gate_experts not in (None, self.num_experts)
                or self.gate_top_k not in (None, self.top_k)
            ):
                raise ValueError("gate pool sizes need split=True")
            if self.num_gate_experts is None:
                self.num_gate_experts = self.num_experts
            if self.gate_top_k is None:
                self.gate_top_k = self.top_k
            if not 0 < self.top_k <= self.num_experts:
                raise ValueError("top_k must be in [1, num_experts]")
            if not 0 < self.gate_top_k <= self.num_gate_experts:
                raise ValueError("gate_top_k must be in [1, num_gate_experts]")

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        dim, hidden = config.dim, config.hidden_dim
        n_ud, n_gate = config.num_experts, config.num_gate_experts
        assert n_gate is not None
        self.split = config.split
        self.top_k = config.top_k
        self.gate_top_k = config.gate_top_k
        self.load_balance_coeff = config.load_balance_coeff
        self.router_weight = nn.Parameter(torch.empty(n_ud, dim))
        self.w_gate = nn.Parameter(torch.empty(n_gate, hidden, dim))
        self.w_up = nn.Parameter(torch.empty(n_ud, hidden, dim))
        self.w_down = nn.Parameter(torch.empty(n_ud, dim, hidden))
        self.register_buffer("expert_bias", torch.zeros(n_ud), persistent=True)
        self.register_buffer("tokens_per_expert", torch.zeros(n_ud), persistent=False)
        if self.split:
            self.gate_router_weight = nn.Parameter(torch.empty(n_gate, dim))
            self.register_buffer("gate_expert_bias", torch.zeros(n_gate), persistent=True)
            self.register_buffer(
                "tokens_per_gate_expert", torch.zeros(n_gate), persistent=False
            )
        else:
            self.gate_router_weight = None

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device or self.expert_bias.device
        for name in ("expert_bias", "tokens_per_expert"):
            setattr(self, name, torch.zeros_like(getattr(self, name), device=device))
        if self.split:
            for name in ("gate_expert_bias", "tokens_per_gate_expert"):
                setattr(
                    self, name, torch.zeros_like(getattr(self, name), device=device)
                )

    def pools(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """``(expert_bias, tokens_per_expert)`` of each routed pool."""
        pools = [(self.expert_bias, self.tokens_per_expert)]
        if self.split:
            pools.append((self.gate_expert_bias, self.tokens_per_gate_expert))
        return pools

    def active_fraction(self) -> dict[str, float]:
        """Share of each expert weight that one token uses."""
        n_ud = self.w_up.shape[0]
        n_gate = self.w_gate.shape[0]
        return {
            "w_gate": self.gate_top_k / n_gate,
            "w_up": self.top_k / n_ud,
            "w_down": self.top_k / n_ud,
        }

    def _route(
        self,
        x_TD: torch.Tensor,
        weight_ED: torch.Tensor,
        bias_E: torch.Tensor,
        counts_E: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k softmax routing; the bias only changes which experts win."""
        scores_TE = F.linear(x_TD.float(), weight_ED.float()).softmax(dim=-1)
        with torch.no_grad():
            ids_TK = torch.topk(scores_TE + bias_E, k=top_k, dim=-1, sorted=False).indices
        weights_TK = scores_TE.gather(-1, ids_TK)
        weights_TK = weights_TK / (weights_TK.sum(dim=-1, keepdim=True) + 1e-20)
        if self.training and not remat.is_recomputing():
            with torch.no_grad():
                flat = ids_TK.reshape(-1)
                counts_E.scatter_add_(0, flat, torch.ones_like(flat, dtype=counts_E.dtype))
        return weights_TK, ids_TK

    @staticmethod
    def _sort(ids_TK: torch.Tensor, num_experts: int):
        """Expert-sorted order of the ``T * K`` assignments, their tokens and
        the cumulative row end of each expert."""
        flat_N = ids_TK.reshape(-1)
        order_N = torch.argsort(flat_N, stable=True)
        counts_E = torch.zeros(num_experts, dtype=torch.int32, device=flat_N.device)
        counts_E.scatter_add_(0, flat_N, torch.ones_like(flat_N, dtype=torch.int32))
        offsets_E = torch.cumsum(counts_E, dim=0, dtype=torch.int32)
        return order_N, order_N // ids_TK.shape[-1], offsets_E

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_TD = x.reshape(-1, shape[-1])
        n_ud = self.w_up.shape[0]
        weights_TK, ids_TK = self._route(
            x_TD, self.router_weight, self.expert_bias, self.tokens_per_expert, self.top_k
        )
        order_N, token_N, offsets_E = self._sort(ids_TK, n_ud)
        x_ND = x_TD[token_N]
        up_NF = _grouped_mm(x_ND, self.w_up, offsets_E)
        if self.split:
            gate_TF = self._mixed_gate(x_TD)
            hidden_NF = gate_TF[token_N] * up_NF
        else:
            gate_NF = _grouped_mm(x_ND, self.w_gate, offsets_E)
            hidden_NF = F.silu(gate_NF) * up_NF
        out_ND = _grouped_mm(hidden_NF, self.w_down, offsets_E)
        out_ND = (out_ND.float() * weights_TK.reshape(-1)[order_N, None]).to(x.dtype)
        out_TD = _scatter_rows(torch.zeros_like(x_TD), token_N, out_ND)
        return out_TD.reshape(shape)

    def _mixed_gate(self, x_TD: torch.Tensor) -> torch.Tensor:
        """``G = sum_j t_j SiLU(W_gate[j] x)`` over each token's gate experts."""
        assert self.gate_router_weight is not None
        n_gate = self.w_gate.shape[0]
        weights_TK, ids_TK = self._route(
            x_TD,
            self.gate_router_weight,
            self.gate_expert_bias,
            self.tokens_per_gate_expert,
            self.gate_top_k,
        )
        order_N, token_N, offsets_E = self._sort(ids_TK, n_gate)
        gate_NF = F.silu(_grouped_mm(x_TD[token_N], self.w_gate, offsets_E))
        gate_NF = gate_NF.float() * weights_TK.reshape(-1)[order_N, None]
        gate_TF = torch.zeros(
            x_TD.shape[0], gate_NF.shape[-1], dtype=torch.float32, device=x_TD.device
        )
        return _scatter_rows(gate_TF, token_N, gate_NF).to(x_TD.dtype)


def iter_factored_moe(model_parts: list[nn.Module]):
    for part in model_parts:
        for module in part.modules():
            if isinstance(module, FactoredMoE):
                yield module


def register_factored_moe_balancing_hook(
    optimizers, model_parts: list[nn.Module], parallelism_context
) -> None:
    """Update every pool's expert bias before each optimizer step.

    Same rule as ``register_moe_load_balancing_hook``: each expert's bias moves
    by ``coeff * sign(mean_load - load)``, centred, with the loads summed over
    the step's microbatches and all data-parallel ranks.
    """
    layers = list(iter_factored_moe(model_parts))
    if not layers:
        return

    @torch.no_grad()
    def _update(*_args, **_kwargs) -> None:
        pools = [pool for layer in layers for pool in layer.pools()]
        counts = torch.cat([count for _bias, count in pools])
        loss_mesh = parallelism_context.get_optional_mesh("loss")
        if loss_mesh is not None:
            torch.distributed.all_reduce(counts, group=loss_mesh.get_group())
        start = 0
        for layer in layers:
            for bias, count in layer.pools():
                load = counts[start : start + count.numel()]
                start += count.numel()
                delta = layer.load_balance_coeff * torch.sign(load.mean() - load)
                bias.add_(delta - delta.mean())
                count.zero_()

    optimizers.register_step_pre_hook(_update)
