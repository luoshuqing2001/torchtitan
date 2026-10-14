# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SM120 FP8 grouped GEMM.

``torch._scaled_grouped_mm`` exists for SM90 and SM100 only. This module is a
rowwise Triton implementation of that contract (fprop/dgrad ``(M,K) @ (E,K,N)``
and wgrad ``(N,M) @ (M,K)``). It matches ``torch._scaled_mm`` bit for bit.

It is not on the training hot path. At the current expert shape, about 4096
tokens by 768 by 1024, one expert already fills the GPU: the cuBLAS loop is
about 0.8 ms / 250 TFLOP/s, while this kernel is several times slower. Call
``triton_fp8_scaled_grouped_mm`` directly if you want the single-launch form.
"""

import torch
import triton
import triton.language as tl


_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = 128

_MAP: dict[int, torch.Tensor] = {}


def _map_buffer(device: torch.device, ntiles: int) -> torch.Tensor:
    index = device.index if device.index is not None else torch.cuda.current_device()
    buf = _MAP.get(index)
    if buf is None or buf.shape[0] < ntiles:
        buf = torch.empty(ntiles, 6, dtype=torch.int32, device=device)
        _MAP[index] = buf
    return buf[:ntiles]


@triton.jit(do_not_specialize=["M", "N", "K", "G", "upper"])
def _build_tile_map(
    offs_ptr,
    map_ptr,
    M,
    N,
    K,
    G,
    upper,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Assign each output tile to an expert. Invalid tiles get group -1.

    Map columns are ``group, m_start, k_start, k_size, m_size, local``.
    """
    pid = tl.program_id(0)
    group = -1
    m_start = 0
    k_start = 0
    k_size = 0
    m_size_out = 0
    local = 0
    if pid < upper:
        prev = 0
        tile_base = 0
        num_n = tl.cdiv(N, BLOCK_N)
        for g in tl.range(G):
            end = tl.load(offs_ptr + g).to(tl.int32)
            if GROUP_M:
                this_m = end - prev
                this_k = K
                this_k_start = 0
                this_m_start = prev
            else:
                this_m = M
                this_k = end - prev
                this_k_start = prev
                this_m_start = 0
            ntiles = tl.cdiv(this_m, BLOCK_M) * num_n
            if this_k <= 0 or this_m <= 0:
                ntiles = 0
            take = (group < 0) & (pid >= tile_base) & (pid < tile_base + ntiles)
            group = tl.where(take, g, group)
            m_start = tl.where(take, this_m_start, m_start)
            k_start = tl.where(take, this_k_start, k_start)
            k_size = tl.where(take, this_k, k_size)
            m_size_out = tl.where(take, this_m, m_size_out)
            local = tl.where(take, pid - tile_base, local)
            tile_base += ntiles
            prev = end
    base = pid * 6
    tl.store(map_ptr + base + 0, group)
    tl.store(map_ptr + base + 1, m_start)
    tl.store(map_ptr + base + 2, k_start)
    tl.store(map_ptr + base + 3, k_size)
    tl.store(map_ptr + base + 4, m_size_out)
    tl.store(map_ptr + base + 5, local)


@triton.jit(do_not_specialize=["N", "stride_am", "stride_ak", "stride_bg", "stride_bk", "stride_bn", "stride_cm", "stride_cn", "stride_cg", "stride_sa_m", "stride_sa_g", "stride_sb_n", "stride_sb_g"])
def _fp8_grouped_mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scale_a_ptr,
    scale_b_ptr,
    map_ptr,
    N,
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_cg,
    stride_sa_m,
    stride_sa_g,
    stride_sb_n,
    stride_sb_g,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * 6
    group = tl.load(map_ptr + base + 0)
    if group >= 0:
        m_start = tl.load(map_ptr + base + 1)
        k_start = tl.load(map_ptr + base + 2)
        k_size = tl.load(map_ptr + base + 3)
        m_size = tl.load(map_ptr + base + 4)
        local = tl.load(map_ptr + base + 5)
        num_m_tiles = tl.cdiv(m_size, BLOCK_M)
        pid_m = local % num_m_tiles
        pid_n = local // num_m_tiles
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = (
            a_ptr
            + (m_start + offs_m[:, None]) * stride_am
            + (k_start + offs_k[None, :]) * stride_ak
        )
        b_ptrs = (
            b_ptr
            + group * stride_bg
            + (k_start + offs_k[:, None]) * stride_bk
            + offs_n[None, :] * stride_bn
        )
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _k in range(0, k_size, BLOCK_K):
            k_left = k_size - _k
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < m_size) & (offs_k[None, :] < k_left),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_left) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        if GROUP_M:
            sa_base = m_start * stride_sa_m
        else:
            sa_base = group * stride_sa_g
        scale_a = tl.load(
            scale_a_ptr + sa_base + offs_m[:, None] * stride_sa_m,
            mask=offs_m[:, None] < m_size,
            other=0.0,
        )
        scale_b = tl.load(
            scale_b_ptr + group * stride_sb_g + offs_n[None, :] * stride_sb_n,
            mask=offs_n[None, :] < N,
            other=0.0,
        )
        c = (acc * scale_a * scale_b).to(tl.bfloat16)
        c_ptrs = (
            c_ptr
            + group * stride_cg
            + (m_start + offs_m[:, None]) * stride_cm
            + offs_n[None, :] * stride_cn
        )
        tl.store(
            c_ptrs,
            c,
            mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < N),
        )


def _upper_tiles(rows: int, cols: int, groups: int, *, group_m: bool) -> int:
    if group_m:
        # sum_g cdiv(m_g, BM) <= cdiv(sum m_g, BM) + G - 1
        return (triton.cdiv(rows, _BLOCK_M) + groups) * triton.cdiv(cols, _BLOCK_N)
    return groups * triton.cdiv(rows, _BLOCK_M) * triton.cdiv(cols, _BLOCK_N)


def _launch(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    offs: torch.Tensor,
    out: torch.Tensor,
    *,
    group_m: bool,
    rows: int,
    cols: int,
    contraction: int,
) -> None:
    groups = offs.shape[0]
    upper = _upper_tiles(rows, cols, groups, group_m=group_m)
    if upper == 0:
        return
    tile_map = _map_buffer(a.device, upper)
    _build_tile_map[(upper,)](
        offs,
        tile_map,
        rows,
        cols,
        contraction,
        groups,
        upper,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        GROUP_M=group_m,
        num_warps=1,
    )
    _fp8_grouped_mm_kernel[(upper,)](
        a,
        b,
        out,
        scale_a,
        scale_b,
        tile_map,
        cols,
        a.stride(0),
        a.stride(1),
        b.stride(0) if b.ndim == 3 else 0,
        b.stride(-2),
        b.stride(-1),
        out.stride(-2),
        out.stride(-1),
        out.stride(0) if out.ndim == 3 else 0,
        scale_a.stride(-1),
        scale_a.stride(0) if scale_a.ndim == 2 else 0,
        scale_b.stride(-1),
        scale_b.stride(0),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        GROUP_M=group_m,
        num_warps=8,
        num_stages=3,
    )


def triton_fp8_scaled_grouped_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    offs: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    """Rowwise FP8 grouped GEMM for the TorchTitan MoE contracts.

    ``b.ndim == 3`` is ``(tokens, K) @ (E, K, N) -> (tokens, N)`` with
    ``scale_a`` shaped ``(tokens,)`` and ``scale_b`` shaped ``(E, N)``.
    ``b.ndim == 2`` is the weight gradient
    ``(N, tokens) @ (tokens, K) -> (E, N, K)`` with scales packed
    expert-major as length ``N * E`` and ``K * E``.
    """
    del use_fast_accum
    if out_dtype != torch.bfloat16:
        raise ValueError(f"SM120 grouped GEMM writes bf16, got {out_dtype}")
    if offs.dtype != torch.int32:
        raise ValueError(f"grouped GEMM offsets must be int32, got {offs.dtype}")
    groups = offs.shape[0]
    if b.ndim == 3:
        tokens, k_dim = a.shape
        b_groups, k_b, n_dim = b.shape
        if b_groups != groups:
            raise ValueError(
                f"grouped GEMM expert count {groups} does not match B {b_groups}"
            )
        if k_dim != k_b:
            raise ValueError(f"grouped GEMM K mismatch: A is {k_dim}, B is {k_b}")
        scale_a = scale_a.reshape(tokens)
        scale_b = scale_b.reshape(groups, n_dim)
        out = torch.empty(tokens, n_dim, device=a.device, dtype=out_dtype)
        _launch(
            a, b, scale_a, scale_b, offs, out,
            group_m=True, rows=tokens, cols=n_dim, contraction=k_dim,
        )
        return out
    if b.ndim == 2:
        n_dim, tokens = a.shape
        tokens_b, k_dim = b.shape
        if tokens != tokens_b:
            raise ValueError(
                f"grouped GEMM token count {tokens} does not match B {tokens_b}"
            )
        scale_a = scale_a.reshape(groups, n_dim)
        scale_b = scale_b.reshape(groups, k_dim)
        out = torch.zeros(groups, n_dim, k_dim, device=a.device, dtype=out_dtype)
        _launch(
            a, b, scale_a, scale_b, offs, out,
            group_m=False, rows=n_dim, cols=k_dim, contraction=tokens,
        )
        return out
    raise ValueError(f"grouped GEMM B must be 2D or 3D, got {b.ndim}D")
