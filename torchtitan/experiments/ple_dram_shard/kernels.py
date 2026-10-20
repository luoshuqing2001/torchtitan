# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA PLE lookup/blending and native-order FP32 row gradients.

Shape legend: T = tokens, U = unique rows, D = concatenated layer width.
This module is imported lazily so CPU-only recipe imports do not need Triton.
"""

import torch
import triton
import triton.language as tl

from torchtitan.models.glimmer_dense import ple_dram as dram


@triton.jit
def _combine_kernel(
    rows_UD,
    inverse_T,
    context_TD,
    output_TD,
    width: tl.constexpr,
    size: tl.constexpr,
    token_scale: tl.constexpr,
    mix_scale: tl.constexpr,
    block: tl.constexpr,
):
    # Batch 128 at width 12288 exceeds the signed 32-bit element range.
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = index < size
    row = tl.load(inverse_T + index // width, valid, other=0)
    value = tl.load(rows_UD + row * width + index % width, valid, other=0)
    value = value.to(output_TD.dtype.element_ty).to(tl.float32)
    context = tl.load(context_TD + index, valid, other=0).to(tl.float32)
    value = (value * token_scale).to(output_TD.dtype.element_ty).to(tl.float32)
    value = (value + context).to(output_TD.dtype.element_ty).to(tl.float32)
    tl.store(output_TD + index, value * mix_scale, valid)


@triton.jit
def _row_gradient_kernel(
    gradient_TD,
    order_T,
    offsets_U,
    output_UD,
    width: tl.constexpr,
    stride_T: tl.constexpr,
    stride_D: tl.constexpr,
    token_scale: tl.constexpr,
    mix_scale: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * block + tl.arange(0, block)
    start = tl.load(offsets_U + row)
    end = tl.load(offsets_U + row + 1)
    total = tl.full((block,), 0, tl.float32)
    # Match deterministic index_add_'s stable token order, without a reduction
    # tree, FP32 token-by-column temporaries or repeated sorting for each tile.
    for cursor in range(start, end):
        token = tl.load(order_T + cursor)
        value = tl.load(
            gradient_TD + token * stride_T + columns * stride_D,
            columns < width,
            other=0,
        ).to(tl.float32)
        if mix_scale != 1.0:
            value = (value * mix_scale).to(gradient_TD.dtype.element_ty).to(tl.float32)
        if token_scale != 1.0:
            value = (
                (value * token_scale).to(gradient_TD.dtype.element_ty).to(tl.float32)
            )
        total += value
    tl.store(output_UD + row * width + columns, total, columns < width)


def combine(rows_UD, inverse_T, context_TD, *, token_scale, mix_scale):
    output_TD = torch.empty_like(context_TD)
    if output_TD.numel():
        _combine_kernel[(triton.cdiv(output_TD.numel(), 2048),)](
            rows_UD,
            inverse_T,
            context_TD,
            output_TD,
            context_TD.shape[1],
            output_TD.numel(),
            token_scale,
            mix_scale,
            2048,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output_TD


def row_gradient(gradient_TD, inverse_T, num_rows, *, token_scale, mix_scale):
    if (
        gradient_TD.device.type != "cuda"
        or gradient_TD.dtype != torch.bfloat16
        or gradient_TD.shape[1] <= 1024
    ):
        if mix_scale != 1.0:
            gradient_TD = gradient_TD * mix_scale
        if token_scale != 1.0:
            gradient_TD = gradient_TD * token_scale
        return dram._embedding_row_grad(gradient_TD, inverse_T, num_rows)
    width = gradient_TD.shape[1]
    output_UD = torch.empty(
        (num_rows, width), device=gradient_TD.device, dtype=torch.float32
    )
    if num_rows:
        sorted_T, order_T = torch.sort(inverse_T, stable=True)
        offsets_U = torch.searchsorted(
            sorted_T,
            torch.arange(num_rows + 1, device=gradient_TD.device, dtype=torch.int64),
        )
        # Keep the varying row count in the launch grid, not a JIT argument.
        _row_gradient_kernel[(num_rows, triton.cdiv(width, 256))](
            gradient_TD,
            order_T,
            offsets_U,
            output_UD,
            width,
            *gradient_TD.stride(),
            token_scale,
            mix_scale,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output_UD


@triton.jit
def _merge_owned_kernel(
    values,
    mapping,
    output,
    rows,
    width: tl.constexpr,
    world: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    rows = rows.to(tl.int64)
    valid = index < rows * width
    row = index // width
    column = index % width
    result = tl.full((block,), 0, tl.float32)
    for rank in range(world):
        source = tl.load(mapping + rank * rows + row, valid, other=0).to(tl.int64)
        value = tl.load(values + source * width + column, valid, other=0)
        result = result + value
    tl.store(output + index, result, valid)


def merge_owned(values_UD, mapping_WU):
    rows = mapping_WU.shape[1]
    width = values_UD.shape[1]
    output_UD = values_UD.new_empty((rows, width))
    if output_UD.numel():
        _merge_owned_kernel[(triton.cdiv(output_UD.numel(), 1024),)](
            values_UD,
            mapping_WU,
            output_UD,
            rows,
            width,
            mapping_WU.shape[0],
            1024,
            enable_fp_fusion=False,
        )
    return output_UD
