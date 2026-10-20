# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offload the wide value activation and restore it during decoder backward.

Shape legend: T = packed tokens, V = Engram value width.
"""

from contextlib import contextmanager

import torch


class _SavedValue:
    def __init__(self, value_TV, owner):
        self.owner = owner
        self.device = value_TV.device
        self.cpu_TV = torch.empty_like(value_TV, device="cpu", pin_memory=True)
        self.gpu_TV = None
        self.graph_id = None
        stream = owner.stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self.cpu_TV.copy_(value_TV.detach(), non_blocking=True)
            value_TV.record_stream(stream)
        self.ready = None

    def prefetch(self):
        if self.gpu_TV is not None:
            return
        stream = self.owner.stream(self.device)
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            self.gpu_TV = torch.empty_like(self.cpu_TV, device=self.device)
            self.gpu_TV.copy_(self.cpu_TV, non_blocking=True)
            self.ready = stream.record_event()

    def unpack(self):
        self.prefetch()
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self.ready)
        self.gpu_TV.record_stream(stream)
        identifier = torch._C._current_graph_task_id()
        if identifier >= 0 and self.graph_id != identifier:
            self.graph_id = identifier
            torch.autograd.Variable._execution_engine.queue_callback(self.release)
        return self.gpu_TV

    def release(self):
        self.gpu_TV = None
        self.owner.records = [
            value for value in self.owner.records if value is not self
        ]


class ValueOffloader:
    def __init__(self):
        self.streams = {}
        self.records = []

    def stream(self, device):
        if device not in self.streams:
            self.streams[device] = torch.cuda.Stream(device=device)
        return self.streams[device]

    @contextmanager
    def capture(self, shape):
        def pack(value):
            if value.device.type == "cuda" and value.shape == shape:
                saved = _SavedValue(value, self)
                self.records.append(saved)
                return saved
            return value

        def unpack(value):
            return value.unpack() if isinstance(value, _SavedValue) else value

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield

    def prefetch(self, gradient):
        for value in self.records:
            value.prefetch()
        return gradient

    def decoder_forward_hook(self, module, inputs, output):
        if output.requires_grad and torch._C._current_graph_task_id() < 0:
            output.register_hook(self.prefetch)

    def clear(self):
        for value in self.records:
            value.gpu_TV = None
        self.records.clear()
