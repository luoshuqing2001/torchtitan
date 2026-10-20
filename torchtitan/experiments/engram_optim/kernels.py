# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exact integer lookup and ordered per-tap convolution kernels.

Shape legend: T = tokens, D = hidden width, R = hash-table rows.
"""

import torch
import triton
import triton.language as tl

from torchtitan.models.glimmer_dense.engram import _segmented_short_conv


@triton.jit
def _conv_forward(
    V,
    W,
    S,
    O,
    T: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    DIL: tl.constexpr,
    B: tl.constexpr,
):
    index = (tl.program_id(0) * B + tl.arange(0, B)).to(tl.int64)
    token = index // D
    channel = index % D
    segment = tl.load(S + token, token < T, other=-1)
    total = tl.full((B,), 0, tl.float32)
    for tap in tl.static_range(K):
        source = token - tap * DIL
        previous = tl.load(S + source, (token < T) & (source >= 0), other=-2)
        valid = (token < T) & (source >= 0) & (previous == segment)
        value = tl.load(V + source * D + channel, valid, other=0).to(tl.float32)
        weight = tl.load(W + tap * D + channel, token < T, other=0).to(tl.float32)
        product = (value * weight).to(O.dtype.element_ty).to(tl.float32)
        total = (total + product).to(O.dtype.element_ty).to(tl.float32)
    tl.store(O + index, total, index < T * D)


@triton.jit
def _conv_backward(
    G,
    W,
    S,
    O,
    T: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    DIL: tl.constexpr,
    B: tl.constexpr,
):
    index = (tl.program_id(0) * B + tl.arange(0, B)).to(tl.int64)
    token = index // D
    channel = index % D
    segment = tl.load(S + token, token < T, other=-1)
    total = tl.full((B,), 0, tl.float32)
    for step in tl.static_range(K):
        tap = K - 1 - step
        destination = token + tap * DIL
        later = tl.load(S + destination, (token < T) & (destination < T), other=-2)
        valid = (token < T) & (destination < T) & (later == segment)
        grad = tl.load(G + destination * D + channel, valid, other=0).to(tl.float32)
        weight = tl.load(W + tap * D + channel, token < T, other=0).to(tl.float32)
        product = (grad * weight).to(O.dtype.element_ty).to(tl.float32)
        product = tl.where(valid, product, 0.0)
        total = (total + product).to(O.dtype.element_ty).to(tl.float32)
    tl.store(O + index, total, index < T * D)


@triton.jit
def _lookup(
    W,
    TOK,
    S,
    O,
    I,
    N: tl.constexpr,
    R: tl.constexpr,
    HD: tl.constexpr,
    EOS: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
):
    row = (tl.program_id(0) * BT * 16 + tl.arange(0, BT * 16)).to(tl.int64)
    token = row // 16
    head = row % 16
    valid = token < N
    current = tl.load(TOK + token, valid, other=EOS).to(tl.int64)
    segment = tl.load(S + token, valid, other=-1)
    previous_segment = tl.load(S + token - 1, valid & (token >= 1), other=-2)
    previous2_segment = tl.load(S + token - 2, valid & (token >= 2), other=-2)
    previous = tl.load(
        TOK + token - 1, valid & (token >= 1) & (segment == previous_segment), other=EOS
    ).to(tl.int64)
    previous2 = tl.load(
        TOK + token - 2,
        valid & (token >= 2) & (segment == previous2_segment),
        other=EOS,
    ).to(tl.int64)
    salt = 2 * head + 1
    bigram = current + previous * 1000003 + salt * 1000006000009
    trigram = current + previous * 1000003 + previous2 * 1000006000009 + salt
    hashed = tl.where(head < 8, bigram, trigram)
    local = ((hashed % R) + R) % R
    index = local + head * R
    tl.store(I + row, index, valid)
    channel = tl.arange(0, BH)
    values = tl.load(
        W + index[:, None] * HD + channel[None, :],
        valid[:, None] & (channel[None, :] < HD),
        other=0,
    )
    tl.store(
        O + row[:, None] * HD + channel[None, :],
        values,
        valid[:, None] & (channel[None, :] < HD),
    )


class _Lookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, tokens, segments, rows, eos):
        width = weight.shape[1]
        output = weight.new_empty((tokens.numel(), 16 * width))
        indices = torch.empty(
            tokens.numel() * 16, device=tokens.device, dtype=torch.int64
        )
        _lookup[(triton.cdiv(tokens.numel(), 4),)](
            weight,
            tokens,
            segments,
            output,
            indices,
            tokens.numel(),
            rows,
            width,
            eos,
            4,
            triton.next_power_of_2(width),
            num_warps=8,
        )
        ctx.save_for_backward(indices)
        ctx.rows, ctx.width = weight.shape
        return output

    @staticmethod
    def backward(ctx, gradient):
        (indices,) = ctx.saved_tensors
        grad_weight = torch.ops.aten.embedding_dense_backward.default(
            gradient.reshape(-1, ctx.width), indices, ctx.rows, -1, False
        )
        return grad_weight, None, None, None, None


def lookup(weight_RD, tokens_T, segments_T, rows_per_head, eos_token_id):
    return _Lookup.apply(weight_RD, tokens_T, segments_T, rows_per_head, eos_token_id)


class _SegmentedConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weight, segments, dilation):
        output = torch.empty_like(values)
        # Round the product and running sum in the original activation dtype.
        _conv_forward[(triton.cdiv(values.numel(), 1024),)](
            values,
            weight,
            segments,
            output,
            *values.shape,
            weight.shape[0],
            dilation,
            1024,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(values, weight, segments)
        ctx.dilation = dilation
        return output

    @staticmethod
    def backward(ctx, gradient):
        values, weight, segments = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Stop the nested gradient at views so leaf hooks run only in the
            # outer backward, while higher-order graphs still reach the inputs.
            values_proxy, weight_proxy = values.view_as(values), weight.view_as(weight)
            active = [
                tensor
                for tensor in (values_proxy, weight_proxy)
                if tensor.requires_grad
            ]
            result = _segmented_short_conv(
                values_proxy, weight_proxy, segments, dilation=ctx.dilation
            )
            grads = iter(
                torch.autograd.grad(result, active, gradient, create_graph=True)
            )
            return (
                next(grads) if values.requires_grad else None,
                next(grads) if weight.requires_grad else None,
                None,
                None,
            )
        gradient = gradient.contiguous()
        inputs = None
        if ctx.needs_input_grad[0]:
            inputs = torch.empty_like(values)
            # Autograd accumulates the tap contributions in reverse order.
            _conv_backward[(triton.cdiv(values.numel(), 1024),)](
                gradient,
                weight,
                segments,
                inputs,
                *values.shape,
                weight.shape[0],
                ctx.dilation,
                1024,
                enable_fp_fusion=False,
            )
        weights = None
        if ctx.needs_input_grad[1]:
            gradients = []
            for tap in range(weight.shape[0]):
                shift = tap * ctx.dilation
                source = values
                if shift:
                    source = torch.zeros_like(values)
                    if shift < values.shape[0]:
                        same = segments[shift:] == segments[:-shift]
                        source[shift:] = torch.where(
                            same[:, None], values[:-shift], source[shift:]
                        )
                # Keep the native dtype, reduction shape and reduction kernel.
                gradients.append((gradient * source).sum(dim=0))
            # SliceBackward adds zero rows when accumulating each tap's weight.
            weights = torch.stack(gradients) + torch.zeros_like(weight)
        return inputs, weights, None, None


def segmented_conv(values_TD, weight, segments_T, *, dilation):
    if (
        values_TD.device.type != "cuda"
        or values_TD.dtype not in (torch.bfloat16, torch.float32)
        or values_TD.dtype != weight.dtype
        or not values_TD.is_contiguous()
        or not weight.is_contiguous()
        or not segments_T.is_contiguous()
    ):
        return _segmented_short_conv(values_TD, weight, segments_T, dilation=dilation)
    return _SegmentedConv.apply(values_TD, weight, segments_T, dilation)
