# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overlap replicated CPU PLE gradient reduction with the same backward task.

Shape legend: T = tokens, D = hidden width, P = projection width.
"""

import torch
import torch.distributed as dist

from torchtitan.models.glimmer_dense import ple_dram as dram


class ReplicatedGradientTransport:
    def __init__(self):
        self.group = None
        self.streams = {}
        self.pending = []
        self.graph_id = None

    def gradient(self, parameter, gradient):
        identifier = torch._C._current_graph_task_id()
        if self.pending and identifier != self.graph_id:
            # Failed backward tasks do not run their final callbacks.
            for _, _, event in self.pending:
                event.synchronize()
            self.pending.clear()
            self.graph_id = None
        eligible = (
            gradient.device.type == "cuda"
            and dist.is_initialized()
            and identifier >= 0
            and not torch.is_grad_enabled()
            and torch.autograd._is_checkpoint_valid()
            and parameter.requires_grad
            and not parameter._backward_hooks
            and not parameter._post_accumulate_grad_hooks
        )
        if gradient.device.type != "cuda" or not dist.is_initialized():
            return dram._cpu_grad(dram._reduce_sum(gradient), parameter.dtype)
        if self.group is None:
            self.group = dist.new_group(backend="nccl")
        device = gradient.device
        stream = self.streams.get(device.index)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self.streams[device.index] = stream
        gradient = gradient.float()
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            dist.all_reduce(gradient, group=self.group)
            reduced = stream.record_event()
        if not eligible:
            # Keep the collective group/order identical when hooks are rank-local.
            torch.cuda.current_stream(device).wait_event(reduced)
            gradient.record_stream(stream)
            return dram._cpu_grad(gradient, parameter.dtype)
        with torch.cuda.stream(stream):
            value = torch.empty(
                gradient.shape, device="cpu", dtype=parameter.dtype, pin_memory=True
            )
            value.copy_(gradient, non_blocking=True)
            gradient.record_stream(stream)
            copied = stream.record_event()
        if self.graph_id != identifier:
            self.graph_id = identifier
            torch.autograd.Variable._execution_engine.queue_callback(
                lambda: self._complete(identifier)
            )
        self.pending.append((parameter, value, copied))
        return None

    def _complete(self, identifier):
        if self.graph_id != identifier:
            return
        values = self.pending
        self.pending = []
        self.graph_id = None
        for parameter, value, copied in values:
            copied.synchronize()
            if parameter.grad is None:
                parameter.grad = value
            else:
                parameter.grad.add_(value)


class ReducedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, staged, inputs, mixed_precision_backward, transport, op):
        ctx.save_for_backward(inputs, staged)
        ctx.parameter = weight
        ctx.transport = transport
        ctx.mixed_precision_backward = mixed_precision_backward
        return op(inputs, staged)

    @staticmethod
    def backward(ctx, grad_out):
        inputs, staged = ctx.saved_tensors
        grad_input = grad_out.matmul(staged)
        flat_grad = grad_out.reshape(-1, grad_out.shape[-1])
        flat_input = inputs.reshape(-1, inputs.shape[-1])
        if not ctx.mixed_precision_backward:
            flat_grad = flat_grad.float()
            flat_input = flat_input.float()
        grad_weight = flat_grad.transpose(0, 1).matmul(flat_input)
        value = ctx.transport.gradient(ctx.parameter, grad_weight)
        return value, None, grad_input, None, None, None


class ReducedNorm(dram._DramRMSNorm):
    @staticmethod
    def forward(ctx, weight, staged, inputs, eps, transport):
        output = dram._DramRMSNorm.forward(ctx, weight, staged, inputs, eps)
        ctx.parameter = weight
        ctx.transport = transport
        return output

    @staticmethod
    def backward(ctx, grad_out):
        if not ctx.fused:
            result = dram._DramRMSNorm.backward(ctx, grad_out)
            return (*result, None)
        inputs, staged, rstd = ctx.saved_tensors
        grad_input, grad_weight = torch.ops.aten._fused_rms_norm_backward.default(
            grad_out, inputs, (inputs.shape[-1],), rstd, staged, (True, True)
        )
        value = ctx.transport.gradient(ctx.parameter, grad_weight)
        return value, None, grad_input, None, None


class ReusingContextNorm(ReducedNorm):
    """Use a sole-use context projection as the native input-gradient buffer.

    SavedContext owns this input: its upstream scale and linear backward do
    not need these values. The validated Hopper kernel loads each owned row
    into registers before storing its input gradient, and computes weight
    partials from those original register values. Grid and reduction order
    match the ordinary native call. Retained graphs keep the original buffer.
    """

    @staticmethod
    def backward(ctx, grad_out):
        if (
            not ctx.fused
            or torch.is_grad_enabled()
            or torch._C._autograd._get_current_graph_task_keep_graph()
        ):
            return ReducedNorm.backward(ctx, grad_out)
        inputs, staged, rstd = ctx.saved_tensors
        from torch._native import registry
        from torch._native.cutedsl import launch
        from torch._native.cutedsl.hw_caps import caps
        from torch._native.ops.norm.rmsnorm_impl import (
            _fused_rms_norm_backward_cond,
            _fused_rms_norm_backward_impl,
        )
        from torch._native.ops.norm.rmsnorm_kernels import compile_rmsnorm
        from torch._native.ops.norm.rmsnorm_launch import backward_launch

        width = inputs.shape[-1]
        hw = caps(inputs.device.index)
        graph = registry._graphs.get(("aten", "_fused_rms_norm_backward", "CUDA"), [])
        selected = next(
            (
                node
                for node in graph
                if node.active and registry._filter_state.check_enabled(node)
            ),
            None,
        )
        if (
            width != 1024
            or hw.cc != (9, 0)
            or not inputs.is_contiguous()
            or not grad_out.is_contiguous()
            or inputs.dtype != grad_out.dtype
            or inputs.data_ptr() % 16
            or grad_out.data_ptr() % 16
            or selected is None
            or selected.impl_fn is not _fused_rms_norm_backward_impl
            or not _fused_rms_norm_backward_cond(
                grad_out, inputs, [width], rstd, staged, [True, True]
            )
        ):
            return ReducedNorm.backward(ctx, grad_out)
        result = inputs.detach()
        flat_input = result.view(-1, width)
        flat_gradient = grad_out.view(-1, width)
        rows = flat_input.shape[0]
        blocks = backward_launch(width, True, inputs.element_size()).blocks(
            rows, hw.sm_count
        )
        partial = torch.empty(blocks, width, device=inputs.device, dtype=torch.float32)
        grad_weight = torch.empty(width, device=inputs.device, dtype=inputs.dtype)
        kernel = compile_rmsnorm(
            "backward", inputs.dtype, width, True, hw.cc, True, grad_out.dtype
        )
        kernel(
            launch.read_only(flat_input),
            launch.read_only(staged),
            launch.read_only(flat_gradient),
            launch.read_only(rstd.view(-1, 1)),
            flat_input,
            partial,
            grad_weight,
            rows,
            blocks,
            launch.stream(inputs.device.index),
        )
        torch.autograd.graph.increment_version(inputs)
        value = ctx.transport.gradient(ctx.parameter, grad_weight)
        return value, None, result, None, None
