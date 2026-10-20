# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in CPU table clipping and native-arithmetic AdamW in one memory pass."""

import math
import os
import sys
import warnings
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_loaded = False


def load_kernel():
    global _loaded
    if _loaded:
        return
    if torch.backends.cpu.get_cpu_capability() != "AVX512":
        raise ValueError("Fused CPU PLE updates require an AVX512 CPU")
    previous_path = os.environ.get("PATH", "")
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + previous_path
    try:
        load(
            name="torchtitan_ple_cpu_adam",
            sources=[str(Path(__file__).parent / "csrc" / "adam.cpp")],
            extra_cflags=["-O3", "-mavx512f", "-mfma", "-fopenmp", "-ffp-contract=off"],
            extra_ldflags=["-fopenmp"],
            is_python_module=False,
            verbose=False,
        )
    finally:
        os.environ["PATH"] = previous_path
    _loaded = True


class CPUTableUpdate:
    def __init__(self, parameter, optimizer, *, num_threads=0):
        if type(num_threads) is not int or num_threads < 0:
            raise ValueError("PLE CPU update threads must be a nonnegative integer")
        self.num_threads = num_threads
        self.parameter = parameter
        self.optimizer = optimizer
        self.pending = None
        self.seen = None
        self.moments = None
        self.versions = None
        self.warned = False
        self.preparations = []
        self.prepared_buffers = None
        self.prepared_steps = 0
        self.compute_cache = None
        self.communication_ready = None
        self.device_update_ready = None
        self.async_clip = False
        self.streaming = False
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ValueError("Fused PLE updates require FP32 CPU master parameters")
        if (
            parameter.ndim != 2
            or not parameter.is_contiguous()
            or parameter.shape[1] % 16
        ):
            raise ValueError(
                "Fused PLE updates require a contiguous, 16-aligned row width"
            )
        load_kernel()

    def group(self):
        return next(
            group
            for group in self.optimizer.param_groups
            if any(parameter is self.parameter for parameter in group["params"])
        )

    def can_fuse(self, container):
        from torch.optim import optimizer as native

        group = self.group()
        supported = (
            isinstance(self.optimizer, torch.optim.AdamW)
            and group.get("fused", False)
            and not group.get("amsgrad", False)
            and not group.get("maximize", False)
            and not group.get("differentiable", False)
            and not group.get("capturable", False)
            and group.get("decoupled_weight_decay", True)
            and isinstance(group["lr"], (float, int))
            and all(isinstance(beta, (float, int)) for beta in group["betas"])
            and isinstance(group["eps"], (float, int))
            and isinstance(group["weight_decay"], (float, int))
            and group["eps"] > 0
            and getattr(self.optimizer, "grad_scale", None) is None
            and getattr(self.optimizer, "found_inf", None) is None
            and not native._global_optimizer_pre_hooks
            and not native._global_optimizer_post_hooks
            and all(
                not item._optimizer_step_pre_hooks
                and not item._optimizer_step_post_hooks
                for item in (container, *container.optimizers)
            )
        )
        if not supported and not self.warned:
            warnings.warn(
                "PLE uses native AdamW clipping/update for optimizer hooks or unsupported options",
                stacklevel=2,
            )
            self.warned = True
        return supported

    def defer(self, norm, max_norm):
        gradient = self.parameter.grad
        coefficient = (max_norm / (norm + 1e-6)).clamp(max=1).to("cpu").item()
        self.pending = (gradient, gradient._version, coefficient)

    def defer_async(self, norm, max_norm, cpu_parameters):
        if norm.device.type != "cuda":
            torch.nn.utils.clip_grads_with_norm_(
                cpu_parameters, max_norm, norm, foreach=True
            )
            self.defer(norm, max_norm)
            return
        coefficient = (max_norm / (norm + 1e-6)).clamp(max=1)
        host = torch.empty((), dtype=coefficient.dtype, pin_memory=True)
        host.copy_(coefficient, non_blocking=True)
        copied = torch.cuda.current_stream(norm.device).record_event()
        gradient = self.parameter.grad
        entries = [
            (p.grad, p.grad._version) for p in cpu_parameters if p.grad is not None
        ]
        self.pending = (gradient, gradient._version, (host, copied), entries)

    def _signature(self):
        state = self.optimizer.state[self.parameter]
        if not state:
            return None
        group = self.group()
        return (
            id(self.parameter),
            self.parameter._version,
            id(state["exp_avg"]),
            state["exp_avg"]._version,
            id(state["exp_avg_sq"]),
            state["exp_avg_sq"]._version,
            state["step"].item(),
            group["lr"],
            *group["betas"],
            group["weight_decay"],
            group["eps"],
        )

    def prepare(self, rows, *, cache_layout=None):
        """Queue scratch-only zero-gradient updates after the current fetch."""
        from torchtitan.models.glimmer_dense.ple_dram import _POOL

        signature = self._signature()
        if cache_layout is not None:
            import torch.distributed as dist

            cache = self.compute_cache
            eligible = torch.tensor(
                int(signature is not None and cache.ready is not None),
                dtype=torch.int32,
            )
            dist.all_reduce(
                eligible, op=dist.ReduceOp.MIN, group=cache.table.ownership.group
            )
            if not bool(eligible):
                cache_layout = None
        if signature is None:
            return
        self.preparations.append(
            _POOL.submit(self._prepare, rows.clone(), signature, cache_layout)
        )

    @torch.no_grad()
    def _prepare(self, rows, signature, cache_layout=None):
        state = self.optimizer.state[self.parameter]
        if self.prepared_buffers is None:
            self.prepared_buffers = tuple(
                torch.empty_like(self.parameter) for _ in range(4)
            )
            self.prepared_buffers[3].zero_()
        p, m, v, zero = self.prepared_buffers
        p.copy_(self.parameter)
        m.copy_(state["exp_avg"])
        v.copy_(state["exp_avg_sq"])
        group = self.group()
        torch.ops.torchtitan_ple_cpu.update_dense(
            p,
            m,
            v,
            zero,
            group["lr"],
            *group["betas"],
            group["weight_decay"],
            group["eps"],
            (state["step"] + 1).item(),
            1.0,
            self.num_threads,
        )
        cache_host = None
        if self.compute_cache is not None and self.compute_cache.dtype is not None:
            cache = self.compute_cache
            cache_host = cache.host
            if getattr(cache, "double_buffer", False):
                cache.wait_host()
            else:
                cache.ready.synchronize()
            cache.host_version = None
            cache_host.copy_(p)
            if cache_layout is not None and cache.preload(cache_layout) is False:
                cache_layout = None
        active = torch.zeros(self.parameter.shape[0], dtype=torch.bool)
        active[rows] = True
        return signature, rows, active, (p, m, v), cache_host, cache_layout

    def join_preparations(self):
        latest = None
        try:
            for future in self.preparations:
                latest = future.result()
        finally:
            self.preparations.clear()
        return latest

    def _seen_rows(self, state, gradient):
        first, second = state["exp_avg"], state["exp_avg_sq"]
        versions = (first._version, second._version)
        if (
            self.moments is None
            or self.moments[0] is not first
            or self.moments[1] is not second
            or self.versions != versions
        ):
            # Include signed zero and non-finite state bits conservatively.
            self.seen = torch.any(first.view(torch.int32), dim=1) | torch.any(
                second.view(torch.int32), dim=1
            )
            self.moments = (first, second)
        hint = getattr(gradient, "_ple_owned_rows", None)
        cached = getattr(gradient, "_ple_owned_norm_cache", None)
        if (
            hint is None
            or hint[0] != gradient._version
            or cached is None
            or cached[1] != gradient._version
        ):
            self.seen.fill_(True)
        else:
            self.seen[hint[1]] = True
        return self.seen

    @torch.no_grad()
    def step(self):
        previous_threads = (
            torch.ops.torchtitan_ple_cpu.set_worker_threads(self.num_threads)
            if self.num_threads
            else None
        )
        try:
            self._step()
        finally:
            self.device_update_ready = None
            if self.communication_ready is not None:
                self.communication_ready.set()
                self.communication_ready = None
            if previous_threads is not None:
                torch.ops.torchtitan_ple_cpu.set_worker_threads(previous_threads)

    def _step(self):
        prepared = self.join_preparations()
        if self.pending is None:
            # Native fused CPU AdamW does not bump moment versions.
            self.seen = self.moments = self.versions = None
            if self.compute_cache is not None:
                self.compute_cache.host_version = None
            self.optimizer.step()
            if self.compute_cache is not None:
                self._refresh_compute_cache()
            return
        pending = self.pending
        gradient, version, coefficient = pending[:3]
        self.pending = None
        if isinstance(coefficient, tuple):
            host, copied = coefficient
            copied.synchronize()
            coefficient = host.item()
            entries = pending[3]
            if any(value._version != saved for value, saved in entries):
                raise RuntimeError("CPU gradients changed before deferred clipping")
            if entries:
                torch._foreach_mul_([value for value, _ in entries], host)
        if gradient is not self.parameter.grad or gradient._version != version:
            raise RuntimeError(
                "PLE gradient changed between clipping and the fused update"
            )
        group = self.group()
        state = self.optimizer.state[self.parameter]
        if not state:
            state["step"] = torch.tensor(0.0, dtype=torch.float32)
            state["exp_avg"] = torch.zeros_like(self.parameter)
            state["exp_avg_sq"] = torch.zeros_like(self.parameter)
        seen = self._seen_rows(state, gradient)
        hint = getattr(gradient, "_ple_owned_rows", None)
        cached = getattr(gradient, "_ple_owned_norm_cache", None)
        use_prepared = (
            prepared is not None
            and prepared[0] == self._signature()
            and hint is not None
            and hint[0] == gradient._version
            and cached is not None
            and cached[1] == gradient._version
            and torch.equal(hint[1], prepared[1])
            and math.isfinite(coefficient)
            and coefficient >= 0
            and math.copysign(1.0, coefficient) > 0
        )
        self.prepared_steps += int(use_prepared)
        state["step"].add_(1)
        kernel = (
            torch.ops.torchtitan_ple_cpu.update_prepared
            if use_prepared
            else torch.ops.torchtitan_ple_cpu.update_seen
        )
        destinations = (
            prepared[3]
            if use_prepared
            else (self.parameter, state["exp_avg"], state["exp_avg_sq"])
        )
        cache = self.compute_cache
        prepared_host = (
            cache is not None
            and use_prepared
            and prepared[4] is not None
            and prepared[4] is cache.host
        )
        packed_host = None
        if (
            prepared_host
            and prepared[5] is not None
            and prepared[5] is cache.candidate_layout
        ):
            cache.ready.synchronize()
            packed_host = cache.packed_host
        fused_host = (
            cache.host
            if prepared_host and cache.dtype == torch.bfloat16 and packed_host is None
            else None
        )
        kernel(
            *destinations,
            gradient,
            group["lr"],
            *group["betas"],
            group["weight_decay"],
            group["eps"],
            state["step"].item(),
            coefficient,
            seen,
            self.parameter.shape[1],
            self.num_threads,
            *(
                (
                    self.parameter,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    prepared[2],
                    fused_host,
                    packed_host,
                    prepared[1] if packed_host is not None else None,
                    self.streaming,
                )
                if use_prepared
                else ()
            ),
        )
        if use_prepared:
            # Keep Parameter/state objects stable while rotating their DRAM
            # storage. Fetches and checkpoint views are recreated each step.
            old_storage = tuple(
                value.data
                for value in (self.parameter, state["exp_avg"], state["exp_avg_sq"])
            )
            for live, result in zip(
                (self.parameter, state["exp_avg"], state["exp_avg_sq"]),
                destinations,
                strict=True,
            ):
                live.data = result
                torch.autograd.graph.increment_version(live)
            self.prepared_buffers = (*old_storage, self.prepared_buffers[3])
            if prepared_host and fused_host is None and packed_host is None:
                cache.host.index_copy_(
                    0,
                    prepared[1],
                    self.parameter.index_select(0, prepared[1]).to(cache.dtype),
                )
            if packed_host is not None:
                cache.patch_layout = prepared[5]
            elif prepared_host:
                cache.host_version = self.parameter._version
        self.versions = (state["exp_avg"]._version, state["exp_avg_sq"]._version)
        if self.compute_cache is not None:
            self._refresh_compute_cache()
        if self.communication_ready is not None:
            self.communication_ready.set()
        self.parameter.grad = None
        try:
            self.optimizer.step()
        finally:
            self.parameter.grad = gradient

    def _refresh_compute_cache(self):
        # CPU arithmetic still overlaps the device optimizer. Cache NCCL must
        # follow all of Muon's NCCL collectives on every rank, not race them.
        if self.device_update_ready is not None:
            gate, event = self.device_update_ready
            gate.wait()
            if event[0] is not None:
                event[0].synchronize()
        self.compute_cache.refresh_after_step()
