# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep PLE parameters in host memory and stage only the used rows on GPU.

The lookup table is gathered on the CPU for the token ids in this step, then
copied on a side stream. The small projection and norm weights are copied
whole, because every element participates in the matmul. Backward reduces
the gradient across data-parallel ranks and writes it back to the CPU
parameter so AdamW can update it in DRAM.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-dram")
_STREAMS: dict[int, torch.cuda.Stream] = {}


def _copy_stream(device: torch.device) -> torch.cuda.Stream:
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _STREAMS.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=index)
        _STREAMS[index] = stream
    return stream


def _reduce_sum(grad: torch.Tensor) -> torch.Tensor:
    """Match FSDP's sum reduction. The loss is already divided by global tokens."""
    grad = grad.to(dtype=torch.float32)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
    return grad


def _cpu_grad(grad: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if grad.device.type == "cpu" and grad.dtype == dtype:
        return grad.contiguous()
    pinned = grad.device.type == "cuda"
    out = torch.empty(grad.shape, dtype=dtype, device="cpu", pin_memory=pinned)
    out.copy_(grad.to(dtype=dtype), non_blocking=pinned)
    if pinned:
        torch.cuda.current_stream(grad.device).synchronize()
    return out


@dataclass
class Staged:
    """A GPU copy of a DRAM parameter, produced on the copy stream."""

    tensor: torch.Tensor
    event: torch.cuda.Event | None

    def value(self) -> torch.Tensor:
        if self.event is not None:
            stream = torch.cuda.current_stream(self.tensor.device)
            stream.wait_event(self.event)
            self.tensor.record_stream(stream)
        return self.tensor


def stage(param: torch.Tensor, device: torch.device, dtype: torch.dtype) -> Staged:
    """Queue a copy of a whole DRAM tensor. CPU callers get a plain cast."""
    if param.device == device and param.dtype == dtype:
        return Staged(param, None)
    source = param.detach()
    if device.type != "cuda":
        return Staged(source.to(device=device, dtype=dtype), None)
    stream = _copy_stream(device)
    with torch.cuda.stream(stream):
        # A dtype change is a kernel, not a memcpy. Copy first, then cast on
        # this same stream so the event covers both.
        copied = source.to(device=device, non_blocking=source.is_pinned())
        if copied.dtype != dtype:
            copied = copied.to(dtype=dtype)
    return Staged(copied, stream.record_event())


def _gather_rows(
    weight: torch.Tensor, ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    uniq, inverse = torch.unique(ids, return_inverse=True)
    rows = weight.index_select(0, uniq).contiguous()
    return rows, inverse, uniq


def _fetch_rows(weight, ids, device, ids_ready, stream, dtype):
    if ids_ready is not None:
        ids_ready.synchronize()
    rows, inverse, uniq = _gather_rows(weight, ids)
    if device.type != "cuda":
        if dtype is not None:
            rows = rows.to(dtype=dtype)
        return rows, inverse, uniq, None
    packed = torch.empty(
        rows.shape, dtype=dtype or rows.dtype, device="cpu", pin_memory=True
    )
    packed.copy_(rows)
    with torch.cuda.device(device), torch.cuda.stream(stream):
        rows = packed.to(device, non_blocking=True)
        inverse = inverse.pin_memory().to(device, non_blocking=True)
        event = stream.record_event()
    return rows, inverse, uniq, event


@dataclass
class RowPrefetch:
    """CPU gather and H2D transfer submitted before the embedding runs."""

    future: Future

    def rows(
        self, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, inverse, uniq, event = self.future.result()
        if event is not None:
            stream = torch.cuda.current_stream(rows.device)
            stream.wait_event(event)
            rows.record_stream(stream)
            inverse.record_stream(stream)
        if dtype is not None:
            rows = rows.to(dtype=dtype)
        return rows, inverse, uniq


def prefetch_rows(
    weight: torch.Tensor, tokens: torch.Tensor, *, dtype: torch.dtype | None = None
) -> RowPrefetch:
    """Overlap token D2H, host gather and row H2D with model computation."""
    device = tokens.device
    ids = tokens.detach().to(dtype=torch.int64)
    ids_ready = None
    stream = None
    if device.type == "cuda":
        stream = _copy_stream(device)
        producer = torch.cuda.current_stream(device)
        stream.wait_stream(producer)
        with torch.cuda.stream(stream):
            host_ids = torch.empty_like(ids, device="cpu", pin_memory=True)
            host_ids.copy_(ids, non_blocking=True)
            ids.record_stream(stream)
            ids_ready = stream.record_event()
        ids = host_ids
    return RowPrefetch(
        _POOL.submit(
            _fetch_rows, weight.detach(), ids, device, ids_ready, stream, dtype
        )
    )


def _accumulate_rows(full, ids, values, copied):
    copied.synchronize()
    full.index_add_(0, ids, values)


def _exchange_cuda_rows(full, uniq, row_grad, counts):
    """Overlap ordered host accumulation with the next rank's transfer."""
    max_rows = max(counts)
    ids = torch.empty(max_rows, device=row_grad.device, dtype=torch.int64)
    values = row_grad.new_empty((max_rows, row_grad.shape[1]))
    host_ids = [
        torch.empty(max_rows, dtype=torch.int64, pin_memory=True) for _ in range(2)
    ]
    host_values = [
        torch.empty(values.shape, dtype=values.dtype, pin_memory=True) for _ in range(2)
    ]
    pending = [None, None]
    producer = torch.cuda.current_stream(row_grad.device)
    transfer = _copy_stream(row_grad.device)
    copied = None
    for rank, count in enumerate(counts):
        slot = rank % 2
        if pending[slot] is not None:
            pending[slot].result()
        if copied is not None:
            # Reuse one GPU buffer only after its previous D2H has completed.
            producer.wait_event(copied)
        if dist.get_rank() == rank:
            ids[: uniq.numel()].copy_(uniq)
            values[: uniq.numel()].copy_(row_grad)
        dist.broadcast(ids, src=rank)
        dist.broadcast(values, src=rank)
        transfer.wait_stream(producer)
        with torch.cuda.stream(transfer):
            host_ids[slot][:count].copy_(ids[:count], non_blocking=True)
            host_values[slot][:count].copy_(values[:count], non_blocking=True)
            ids.record_stream(transfer)
            values.record_stream(transfer)
            copied = transfer.record_event()
        # One worker preserves the original rank-by-rank FP32 addition order.
        pending[slot] = _POOL.submit(
            _accumulate_rows,
            full,
            host_ids[slot][:count],
            host_values[slot][:count],
            copied,
        )
    for future in pending:
        if future is not None:
            future.result()


def _embedding_grad(uniq, row_grad, vocab, dtype):
    """Exchange only touched rows; materialize the AdamW gradient in DRAM."""
    full = torch.zeros(
        vocab,
        row_grad.shape[1],
        dtype=torch.float32,
        device="cpu",
        pin_memory=row_grad.device.type == "cuda",
    )
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # All ranks must use identical collective shapes even for disjoint tokens.
        size = torch.tensor([uniq.numel()], device=row_grad.device, dtype=torch.int64)
        sizes = [torch.empty_like(size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, size)
        if row_grad.device.type == "cuda":
            _exchange_cuda_rows(full, uniq, row_grad, torch.cat(sizes).cpu().tolist())
            return full.to(dtype=dtype)
        max_rows = max(item.item() for item in sizes)
        ids = torch.zeros(max_rows, device=row_grad.device, dtype=torch.int64)
        values = row_grad.new_zeros((max_rows, row_grad.shape[1]))
        ids[: uniq.numel()].copy_(uniq)
        values[: uniq.numel()].copy_(row_grad)
        # Broadcast rank by rank to bound GPU workspace independently of world size.
        for rank, count in enumerate(sizes):
            rank_ids = ids.clone()
            rank_values = values.clone()
            dist.broadcast(rank_ids, src=rank)
            dist.broadcast(rank_values, src=rank)
            count = count.item()
            full.index_add_(
                0, rank_ids[:count].cpu(), _cpu_grad(rank_values[:count], torch.float32)
            )
    else:
        full.index_copy_(0, uniq.cpu(), _cpu_grad(row_grad, torch.float32))
    return full.to(dtype=dtype)


def _embedding_row_grad(flat, inverse, num_rows):
    width = flat.shape[1]
    row_grad = torch.empty(num_rows, width, device=flat.device, dtype=torch.float32)
    if flat.device.type != "cuda" or flat.dtype == torch.float32 or width <= 1024:
        row_grad.zero_().index_add_(0, inverse, flat.to(dtype=torch.float32))
        return row_grad
    # Partition columns, not token rows, to preserve each column's reduction.
    # A full FP32 token-by-all-layers temporary can otherwise exceed 6 GiB.
    for offset in range(0, width, 1024):
        # Overlap the final tile to keep the same reduction kernel for tails.
        column = min(offset, width - 1024)
        end = column + 1024
        block = torch.zeros(
            num_rows, end - column, device=flat.device, dtype=torch.float32
        )
        block.index_add_(0, inverse, flat[:, column:end].to(dtype=torch.float32))
        row_grad[:, column:end].copy_(block)
    return row_grad


class _PrefetchedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        rows: torch.Tensor,
        inverse: torch.Tensor,
        uniq: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(uniq, inverse)
        ctx.vocab = weight.shape[0]
        ctx.cpu_dtype = weight.dtype
        return rows.index_select(0, inverse)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        uniq, inverse = ctx.saved_tensors
        width = grad_out.shape[-1]
        flat = grad_out.reshape(-1, width)
        inverse = inverse.to(device=flat.device)
        uniq = uniq.to(device=flat.device)
        row_grad = _embedding_row_grad(flat, inverse, uniq.shape[0])
        return (
            _embedding_grad(uniq, row_grad, ctx.vocab, ctx.cpu_dtype),
            None,
            None,
            None,
        )


class _DramLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        staged: torch.Tensor,
        inputs: torch.Tensor,
        mixed_precision_backward: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(inputs, staged)
        ctx.cpu_dtype = weight.dtype
        ctx.mixed_precision_backward = mixed_precision_backward
        return F.linear(inputs, staged)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        inputs, staged = ctx.saved_tensors
        grad_input = grad_out.matmul(staged)
        flat_grad = grad_out.reshape(-1, grad_out.shape[-1])
        flat_input = inputs.reshape(-1, inputs.shape[-1])
        if ctx.mixed_precision_backward:
            grad_weight = flat_grad.transpose(0, 1).matmul(flat_input)
        else:
            grad_weight = (
                flat_grad.to(dtype=torch.float32)
                .transpose(0, 1)
                .matmul(flat_input.to(dtype=torch.float32))
            )
        grad_weight = _reduce_sum(grad_weight)
        return _cpu_grad(grad_weight, ctx.cpu_dtype), None, grad_input, None


class _DramRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        staged: torch.Tensor,
        inputs: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        normalized_shape = (inputs.shape[-1],)
        ctx.eps = eps
        ctx.cpu_dtype = weight.dtype
        ctx.fused = inputs.device.type == "cuda"
        if ctx.fused:
            output, rstd = torch.ops.aten._fused_rms_norm.default(
                inputs, normalized_shape, staged, eps
            )
            ctx.save_for_backward(inputs, staged, rstd)
            return output
        ctx.save_for_backward(inputs, staged)
        return F.rms_norm(inputs, normalized_shape, staged, eps)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if ctx.fused:
            inputs, staged, rstd = ctx.saved_tensors
            grad_input, grad_weight = torch.ops.aten._fused_rms_norm_backward.default(
                grad_out, inputs, (inputs.shape[-1],), rstd, staged, (True, True)
            )
            grad_weight = _reduce_sum(grad_weight)
            return _cpu_grad(grad_weight, ctx.cpu_dtype), None, grad_input, None
        # PyTorch does not register the fused RMSNorm backward on CPU.
        inputs, staged = ctx.saved_tensors
        with torch.enable_grad():
            inputs_req = inputs.detach().requires_grad_(True)
            staged_req = staged.detach().requires_grad_(True)
            out = F.rms_norm(inputs_req, (inputs_req.shape[-1],), staged_req, ctx.eps)
            out.backward(grad_out)
        assert staged_req.grad is not None and inputs_req.grad is not None
        grad_weight = _reduce_sum(staged_req.grad)
        return _cpu_grad(grad_weight, ctx.cpu_dtype), None, inputs_req.grad, None


def _match_inputs(value: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Master weights stay fp32 in DRAM. The matmul uses the activation dtype."""
    if value.dtype == inputs.dtype and value.device == inputs.device:
        return value
    return value.to(device=inputs.device, dtype=inputs.dtype)


def dram_embedding(
    weight: torch.Tensor,
    prefetch: RowPrefetch,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    rows, inverse, uniq = prefetch.rows(dtype)
    return _PrefetchedEmbedding.apply(weight, rows, inverse, uniq)


def dram_linear(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    staged: Staged | None = None,
    *,
    mixed_precision_backward: bool = False,
) -> torch.Tensor:
    if staged is None:
        staged = stage(weight, inputs.device, inputs.dtype)
    return _DramLinear.apply(
        weight, _match_inputs(staged.value(), inputs), inputs, mixed_precision_backward
    )


def dram_rms_norm(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    eps: float,
    staged: Staged | None = None,
) -> torch.Tensor:
    if staged is None:
        staged = stage(weight, inputs.device, inputs.dtype)
    return _DramRMSNorm.apply(
        weight, _match_inputs(staged.value(), inputs), inputs, eps
    )


def move_ple_parameters_to_dram(model: nn.Module) -> None:
    """Pin every PLE parameter in host memory after initialization."""
    from torch.distributed.tensor import DTensor

    for param in _iter_ple_parameters(model):
        if isinstance(param, DTensor) or isinstance(param.data, DTensor):
            raise TypeError(
                "PLE parameters must stay outside FSDP so they can live in DRAM."
            )
        if param.data.device.type == "meta":
            raise RuntimeError("PLE parameters are still on the meta device.")
        cpu = param.data.detach().to(device="cpu", copy=True).contiguous()
        if torch.cuda.is_available():
            cpu = cpu.pin_memory()
        param.data = cpu


def preserve_ple_optimizer_implementation(optimizers, model_parts) -> None:
    """Keep runtime foreach/fused choices when restoring PLE AdamW state."""
    ple_parameters = {
        id(parameter)
        for model in model_parts
        for parameter in _iter_ple_parameters(model)
    }
    for optimizer in optimizers:
        if not isinstance(optimizer, torch.optim.AdamW):
            continue
        if not any(
            id(parameter) in ple_parameters
            for group in optimizer.param_groups
            for parameter in group["params"]
        ):
            continue
        implementation = [
            {key: group[key] for key in ("foreach", "fused")}
            for group in optimizer.param_groups
        ]

        def restore(optimizer, implementation=implementation):
            for group, options in zip(
                optimizer.param_groups, implementation, strict=True
            ):
                group.update(options)

        optimizer.register_load_state_dict_post_hook(restore)


def _iter_ple_parameters(model: nn.Module):
    ple = getattr(model, "ple", None)
    if isinstance(ple, nn.Module):
        yield from ple.parameters()
    layer_ple = getattr(model, "layer_ple", None)
    if isinstance(layer_ple, nn.Module):
        yield from layer_ple.parameters()
    layers = getattr(model, "layers", None)
    if layers is None:
        return
    for layer in layers.values():
        child = getattr(layer, "layer_ple", None)
        if isinstance(child, nn.Module):
            yield from child.parameters()
