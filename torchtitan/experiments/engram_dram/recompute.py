# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recompute the wide value projection shortly before its backward consumer.

Shape legend: T = tokens, D = hidden width, V = value width.
"""

import torch
import torch.nn.functional as F


class _Projection:
    def __init__(self, inputs_TD, weight_VD, *, align=False):
        self.align = align
        self.inputs = inputs_TD.detach()
        self.weight = weight_VD.detach()
        self.versions = (inputs_TD._version, weight_VD._version)
        self.value = self.ready = None
        self.produced = None
        if inputs_TD.is_cuda:
            self.produced = torch.cuda.current_stream(inputs_TD.device).record_event()

    def validate(self):
        if (self.inputs._version, self.weight._version) != self.versions:
            raise RuntimeError(
                "Value recomputation input was modified by an inplace operation"
            )

    def prefetch(self, stream):
        self.validate()
        if self.value is not None:
            return
        width = self.weight.shape[0]
        padding = 4 if self.align and width % 8 == 4 else 0
        current = torch.cuda.current_stream(self.inputs.device)
        # Reuse the consumer arena rather than reserving a second wide buffer
        # permanently on the prefetch stream.
        value_TV = self.inputs.new_empty((self.inputs.shape[0], width + padding))
        stream.wait_stream(current)
        with torch.cuda.device(self.inputs.device), torch.cuda.stream(stream):
            stream.wait_event(self.produced)
            weight_VD = self.weight
            if padding:
                weight_VD = F.pad(weight_VD, (0, 0, 0, padding))
            torch.mm(self.inputs, weight_VD.t(), out=value_TV)
            value_TV.record_stream(stream)
            self.value = value_TV[:, :width]
            self.ready = stream.record_event()

    def project(self, inputs_TD, weight_VD):
        width = weight_VD.shape[0]
        padding = 4 if self.align and width % 8 == 4 else 0
        if padding:
            weight_VD = F.pad(weight_VD, (0, 0, 0, padding))
        return F.linear(inputs_TD, weight_VD)[:, :width]

    def consume(self, inputs_TD, weight_VD):
        self.validate()
        if torch.is_grad_enabled():
            return F.linear(inputs_TD, weight_VD)
        if self.value is None:
            return self.project(inputs_TD, weight_VD)
        stream = torch.cuda.current_stream(inputs_TD.device)
        stream.wait_event(self.ready)
        self.value.record_stream(stream)
        return self.value


class AlignedValueLinear(torch.autograd.Function):
    """Align forward output storage while preserving backward GEMM dimensions."""

    @staticmethod
    def forward(ctx, inputs_TD, weight_VD):
        ctx.save_for_backward(inputs_TD, weight_VD)
        width = weight_VD.shape[0]
        padding = (-width) % 8
        padded_VD = F.pad(weight_VD, (0, 0, 0, padding))
        return F.linear(inputs_TD, padded_VD)[:, :width]

    @staticmethod
    def backward(ctx, gradient_TV):
        inputs_TD, weight_VD = ctx.saved_tensors
        grad_inputs_TD = (
            gradient_TV.matmul(weight_VD) if ctx.needs_input_grad[0] else None
        )
        grad_weight_VD = (
            gradient_TV.t().matmul(inputs_TD) if ctx.needs_input_grad[1] else None
        )
        return grad_inputs_TD, grad_weight_VD


class ValueRecomputer:
    def __init__(self, *, align=False, align_forward=False):
        self.align = align
        self.align_forward = align_forward
        self.records = []
        self.streams = {}

    def project(self, inputs_TD, up_VD, down_DV):
        wide_signature = None
        record = None

        def pack(value):
            nonlocal record
            if (
                wide_signature is not None
                and (value.data_ptr(), value.shape) == wide_signature
            ):
                record = _Projection(inputs_TD, up_VD, align=self.align)
                self.records.append(record)
                return record
            return (value.detach(), value._version)

        def unpack(saved):
            if isinstance(saved, _Projection):
                value = saved.consume(saved.inputs, saved.weight)
                if torch._C._current_graph_task_id() >= 0:
                    torch.autograd.Variable._execution_engine.queue_callback(
                        lambda: self.release(saved)
                    )
                return value
            value, version = saved
            if value._version != version:
                raise RuntimeError(
                    "Value projection input was modified by an inplace operation"
                )
            return value

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            linear = (
                AlignedValueLinear.apply
                if self.align_forward and up_VD.shape[0] % 8 == 4
                else F.linear
            )
            wide_TV = linear(inputs_TD, up_VD)
            # Hooks retain the pack closure; capture metadata, not the value.
            wide_signature = (wide_TV.data_ptr(), wide_TV.shape)
            return F.linear(wide_TV, down_DV)

    def release(self, record):
        record.value = None
        self.records = [r for r in self.records if r is not record]

    def prefetch(self, gradient):
        if gradient.is_cuda:
            stream = self.streams.get(gradient.device)
            if stream is None:
                stream = torch.cuda.Stream(device=gradient.device)
                self.streams[gradient.device] = stream
            with torch.no_grad():
                for record in self.records:
                    record.prefetch(stream)
        return gradient

    def decoder_forward_hook(self, module, inputs, output):
        if output.requires_grad and torch._C._current_graph_task_id() < 0:
            output.register_hook(self.prefetch)

    def clear(self):
        for record in self.records:
            if record.ready is not None:
                record.ready.synchronize()
        self.records.clear()
