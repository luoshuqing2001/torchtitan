# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in fused BF16 PLE lookup, blending and ordered table gradients.

Shape legend: T = tokens, U = unique rows, R = owned rows, D = layer width.
"""

import torch

from torchtitan.models.glimmer_dense import ple_dram as dram


class FusedOwnedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weight_RD,
        rows_UD,
        inverse_T,
        unique_U,
        rank_ids,
        ownership,
        context_TD,
        token_scale,
    ):
        from .kernels import combine

        ctx.save_for_backward(inverse_T, unique_U)
        ctx.rank_ids = rank_ids
        ctx.ownership = ownership
        ctx.token_scale = token_scale
        return combine(
            rows_UD,
            inverse_T,
            context_TD,
            token_scale=token_scale,
            mix_scale=0.5**0.5,
        )

    @staticmethod
    def backward(ctx, gradient_TD):
        from .kernels import row_gradient
        from .model import owned_gradient

        inverse_T, unique_U = ctx.saved_tensors
        weight_grad_RD = None
        if ctx.needs_input_grad[0]:
            if torch.is_grad_enabled():
                gradient_UD = dram._embedding_row_grad(
                    (gradient_TD * (0.5**0.5)) * ctx.token_scale,
                    inverse_T,
                    unique_U.numel(),
                )
            else:
                gradient_UD = row_gradient(
                    gradient_TD,
                    inverse_T,
                    unique_U.numel(),
                    token_scale=ctx.token_scale,
                    mix_scale=0.5**0.5,
                )
            weight_grad_RD = owned_gradient(
                unique_U, gradient_UD, ctx.rank_ids, ctx.ownership
            )
            del gradient_UD
        context_grad_TD = gradient_TD * (0.5**0.5) if ctx.needs_input_grad[6] else None
        return weight_grad_RD, None, None, None, None, None, context_grad_TD, None
