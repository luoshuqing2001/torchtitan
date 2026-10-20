# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stage PLE checkpoint inputs while preserving their original gradient graph.

Shape legend: T = tokens, U = unique token ids, P = per-layer input width.
"""

import functools
import os
import sys
import weakref
from pathlib import Path

import torch

_version_loaded = False


def load_version_tracker():
    global _version_loaded
    if _version_loaded:
        return
    from torch.utils.cpp_extension import load

    previous_path = os.environ.get("PATH", "")
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + previous_path
    try:
        load(
            name="torchtitan_ple_saved_input_version",
            sources=[str(Path(__file__).parent / "csrc" / "saved_version.cpp")],
            extra_cflags=["-O2"],
            is_python_module=False,
            verbose=False,
        )
    finally:
        os.environ["PATH"] = previous_path
    _version_loaded = True


class PackedInput:
    def __init__(self, value_TP, stream, *, first=None, inverse=None, audit=False):
        self.device = value_TP.device
        self.stream = stream
        self.consumer = torch.cuda.current_stream(value_TP.device)
        self.version = torch.classes.torchtitan_ple_saved_input.SavedVersion(value_TP)
        self.shape = tuple(value_TP.shape)
        self.inverse = inverse
        self.audit_check = None
        with torch.no_grad():
            source_UP = (
                value_TP.detach()
                if first is None
                else value_TP.detach().index_select(0, first)
            )
            if audit and first is not None:
                restored_TP = source_UP.index_select(0, inverse)
                self.audit_check = (
                    restored_TP.view(torch.uint8).eq(value_TP.view(torch.uint8)).all()
                )
        self.host = torch.empty(
            source_UP.shape, dtype=value_TP.dtype, device="cpu", pin_memory=True
        )
        self.value = self.ready = None
        stream.wait_stream(self.consumer)
        with torch.cuda.stream(stream), torch.no_grad():
            self.host.copy_(source_UP, non_blocking=True)
            source_UP.record_stream(stream)

    def prefetch(self):
        self.version.check()
        if self.value is not None:
            return
        with torch.cuda.stream(self.consumer):
            value_TP = torch.empty(
                self.shape, dtype=self.host.dtype, device=self.device
            )
            compact_UP = (
                value_TP
                if self.inverse is None
                else torch.empty_like(self.host, device=self.device)
            )
        self.stream.wait_stream(self.consumer)
        with torch.cuda.stream(self.stream), torch.no_grad():
            compact_UP.copy_(self.host, non_blocking=True)
            if self.inverse is not None:
                torch.index_select(compact_UP, 0, self.inverse, out=value_TP)
                self.inverse.record_stream(self.stream)
                compact_UP.record_stream(self.stream)
            self.version.attach(value_TP)
            value_TP.record_stream(self.stream)
            self.ready = self.stream.record_event()
        self.value = value_TP

    def unpack(self):
        self.prefetch()
        consumer = torch.cuda.current_stream(self.device)
        consumer.wait_event(self.ready)
        self.value.record_stream(consumer)
        return self.value


def positional_checkpoint(checkpoint, function, *args, **kwargs):
    # Flatten keyword tensors too: checkpoint's saved-input hooks otherwise
    # need not see tensors captured directly by a recomputation closure.
    count = len(args)
    keys = tuple(kwargs)
    flat = (*args, *(kwargs[key] for key in keys))

    def invoke(*inputs):
        return function(*inputs[:count], **dict(zip(keys, inputs[count:], strict=True)))

    return checkpoint(invoke, *flat)


class SavedInputStaging:
    def __init__(self, num_layers, *, compact=False, audit=False):
        load_version_tracker()
        self.num_layers = num_layers
        self.compact = compact
        self.audit = audit
        self.records = {}
        self.stream = None
        self.first = self.inverse = None
        self.packed_count = self.packed_bytes = self.host_payload_bytes = 0
        self.pending_checks = []
        self.compression_checks = []

    def set_indices(self, inverse_T, num_unique):
        if not self.compact:
            return
        inverse_T = inverse_T.reshape(-1)
        first_U = torch.full(
            (num_unique,), inverse_T.numel(), device=inverse_T.device, dtype=torch.int64
        )
        first_U.scatter_reduce_(
            0,
            inverse_T,
            torch.arange(inverse_T.numel(), device=inverse_T.device),
            reduce="amin",
            include_self=True,
        )
        self.first, self.inverse = first_U, inverse_T

    def pack(self, value, signature, layer):
        actual = value.data_ptr(), tuple(value.shape), tuple(value.stride())
        if actual != signature:
            return value.detach(), value._version
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=value.device)
        packed = PackedInput(
            value, self.stream, first=self.first, inverse=self.inverse, audit=self.audit
        )
        self.packed_count += 1
        self.packed_bytes += value.numel() * value.element_size()
        self.host_payload_bytes += packed.host.numel() * packed.host.element_size()
        if packed.audit_check is not None:
            self.pending_checks.append(packed.audit_check)
        self.records[layer] = weakref.ref(packed)
        return packed

    @staticmethod
    def unpack(value):
        if isinstance(value, PackedInput):
            return value.unpack()
        tensor, version = value
        if tensor._version != version:
            raise RuntimeError("A saved checkpoint input changed in place")
        return tensor

    def prefetch(self, layer):
        reference = self.records.get(layer)
        packed = reference() if reference is not None else None
        if packed is not None:
            packed.prefetch()

    def release(self, layer):
        reference = self.records.get(layer)
        packed = reference() if reference is not None else None
        if packed is not None:
            packed.value = None

    def finish_step(self, *, wait=False):
        # Tensor/pinned allocators track stream lifetimes during training.
        # A full stream join is needed only for shutdown, not every update.
        if wait and self.stream is not None:
            self.stream.synchronize()
        checks = [bool(value) for value in self.pending_checks]
        self.pending_checks.clear()
        self.compression_checks.extend(checks)
        for layer in self.records:
            self.release(layer)
        self.records.clear()
        self.first = self.inverse = None
        if not all(checks):
            raise RuntimeError("PLE token compression changed saved input values")

    def statistics(self):
        return {
            "packed_count": self.packed_count,
            "packed_logical_bytes": self.packed_bytes,
            "host_payload_bytes": self.host_payload_bytes,
            "compression_audits": self.compression_checks,
        }


def wrap_checkpoint_inputs(wrapped, manager, index):
    wrapped.checkpoint_fn = functools.partial(
        positional_checkpoint, wrapped.checkpoint_fn
    )
    original = wrapped.forward

    def forward(*args, **kwargs):
        value = kwargs.get("per_layer_input")
        if value is None or not value.is_cuda or not torch.is_grad_enabled():
            return original(*args, **kwargs)
        signature = value.data_ptr(), tuple(value.shape), tuple(value.stride())

        def pack(tensor):
            # Capture metadata only; retaining value would keep the GPU storage.
            return manager.pack(tensor, signature, index)

        with torch.autograd.graph.saved_tensors_hooks(pack, manager.unpack):
            output = original(*args, **kwargs)
        if index not in manager.records or manager.records[index]() is None:
            raise RuntimeError("The PLE checkpoint input was not staged")
        if index == manager.num_layers - 1:
            manager.prefetch(index)
        if output.requires_grad:

            def prefetch_lower(gradient):
                manager.prefetch(index - 1)
                return gradient

            output.register_hook(prefetch_lower)
        hidden = args[0]
        if hidden.requires_grad:

            def release_current(gradient):
                manager.release(index)
                return gradient

            hidden.register_hook(release_current)
        return output

    wrapped.forward = forward
    return wrapped
