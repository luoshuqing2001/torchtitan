# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compute CPU-order FP32 chunk norms beside the owned gradient DMA.

Shape legend: R = owned rows, U = requested rows, D = table width,
C = norm chunks, L = elements per norm chunk.
"""

import warnings
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_norm_kernel(
    values,
    mapping,
    output,
    num_chunks,
    WIDTH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    LANES: tl.constexpr,
    FUSED: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    chunk = (tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
    lane = tl.arange(0, LANES)
    accumulator = tl.full((BLOCK_ROWS, LANES), 0.0, tl.float32)
    for offset in range(CHUNK_SIZE // LANES):
        position = chunk[:, None] * CHUNK_SIZE + offset * LANES + lane[None, :]
        row = position // WIDTH
        source = tl.load(mapping + row, chunk[:, None] < num_chunks, other=-1).to(
            tl.int64
        )
        value = tl.load(
            values + source * WIDTH + position % WIDTH,
            (chunk[:, None] < num_chunks) & (source >= 0),
            other=0.0,
        )
        if FUSED:
            accumulator = tl.fma(value, value, accumulator)
        else:
            accumulator = accumulator + value * value
    total = tl.sum(tl.where(lane[None, :] == 0, accumulator, 0.0), 1)
    for index in tl.static_range(1, LANES):
        total = total + tl.sum(tl.where(lane[None, :] == index, accumulator, 0.0), 1)
    tl.store(output + chunk, tl.sqrt_rn(total), chunk < num_chunks)


def _calculate(values_UD, mapping_R, num_rows, chunk_size, plan):
    num_chunks = num_rows * values_UD.shape[1] // chunk_size
    summary_C = torch.empty(num_chunks, device=values_UD.device, dtype=torch.float32)
    lanes, fused = plan
    _chunk_norm_kernel[(triton.cdiv(num_chunks, 8),)](
        values_UD,
        mapping_R,
        summary_C,
        num_chunks,
        values_UD.shape[1],
        chunk_size,
        lanes,
        fused,
        8,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return summary_C


@dataclass
class DeviceNormSummary:
    values: torch.Tensor
    ready: torch.cuda.Event


class DeviceChunkNorm:
    def __init__(self, device, chunk_size, executor):
        self.device = device
        self.chunk_size = chunk_size
        self.stream = torch.cuda.Stream(device=device)
        # Match the CPU worker's floating-point environment, not just its ISA.
        self.plan = executor.submit(self._select_plan).result()
        if self.plan is None:
            warnings.warn(
                "PLE GPU norms could not match the native CPU reduction; using CPU summaries",
                stacklevel=2,
            )

    @torch.no_grad()
    def _select_plan(self):
        generator = torch.Generator(device="cpu").manual_seed(42)
        probes = []
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            for length in (256, 65536):
                probe_CL = torch.randn(16, length, generator=generator)
                probe_CL.mul_(
                    torch.tensor([1e-22, 1e-10, 1.0, 1e10]).repeat(4).unsqueeze(1)
                )
                expected_C = torch.linalg.vector_norm(probe_CL, ord=2, dim=1)
                probes.append((probe_CL.to(self.device), expected_C))
            mapping_R = torch.arange(16, device=self.device, dtype=torch.int64)
            for lanes in (8, 16, 4):
                if self.chunk_size % lanes:
                    continue
                for fused in (False, True):
                    plan = (lanes, fused)
                    if all(
                        torch.equal(
                            _calculate(probe_CL, mapping_R, 16, probe_CL.shape[1], plan)
                            .cpu()
                            .view(torch.int32),
                            expected_C.view(torch.int32),
                        )
                        for probe_CL, expected_C in probes
                    ):
                        return plan
        return None

    @torch.no_grad()
    def compute(self, values_UD, indices_U, num_rows):
        if self.plan is None:
            return None
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            mapping_R = torch.full((num_rows,), -1, dtype=torch.int64, pin_memory=True)
            mapping_R[indices_U] = torch.arange(indices_U.numel(), dtype=torch.int64)
            mapping_R = mapping_R.to(self.device, non_blocking=True)
            summary_C = _calculate(
                values_UD, mapping_R, num_rows, self.chunk_size, self.plan
            )
            values_C = torch.empty(
                summary_C.shape, device="cpu", dtype=torch.float32, pin_memory=True
            )
            values_C.copy_(summary_C, non_blocking=True)
            values_UD.record_stream(self.stream)
            ready = self.stream.record_event()
        return DeviceNormSummary(values_C, ready)
