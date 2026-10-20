# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA 13 batch copies into private pinned CPU gradient rows.

Shape legend: R = owned rows, U = requested rows, D = table width.
"""

import ctypes

import numpy as np
import torch


class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _Attributes(ctypes.Structure):
    _fields_ = [
        ("src_access_order", ctypes.c_int),
        ("src_location", _Location),
        ("dst_location", _Location),
        ("flags", ctypes.c_uint),
    ]


_runtime = None


def _get_runtime():
    global _runtime
    if _runtime is None:
        if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
            raise ValueError("Direct PLE gradient copies require CUDA 13 or later")
        _runtime = ctypes.CDLL("libcudart.so.13")
        _runtime.cudaMemcpyBatchAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        _runtime.cudaMemcpyBatchAsync.restype = ctypes.c_int
    return _runtime


def copy_rows(destination_RD, source_UD, indices_U, stream):
    if not indices_U.numel():
        return
    if not destination_RD.is_pinned() or not source_UD.is_contiguous():
        raise ValueError("Direct PLE copies require pinned output and contiguous rows")
    rows = indices_U.numpy()
    starts = np.concatenate(
        (np.array([0], dtype=np.int64), np.flatnonzero(np.diff(rows) != 1) + 1)
    )
    counts = np.diff(np.concatenate((starts, np.array([len(rows)], dtype=np.int64))))
    row_bytes = source_UD.shape[1] * source_UD.element_size()
    destinations = np.uint64(destination_RD.data_ptr()) + rows[starts].astype(
        np.uint64
    ) * np.uint64(row_bytes)
    sources = np.uint64(source_UD.data_ptr()) + starts.astype(np.uint64) * np.uint64(
        row_bytes
    )
    sizes = counts.astype(np.uint64) * np.uint64(row_bytes)
    attributes = _Attributes()
    attributes.src_access_order = 1  # Access source pointers in stream order.
    attributes.flags = 1  # Prefer overlap with compute.
    first = ctypes.c_size_t(0)
    error = _get_runtime().cudaMemcpyBatchAsync(
        destinations.ctypes.data,
        sources.ctypes.data,
        sizes.ctypes.data,
        len(starts),
        ctypes.byref(attributes),
        ctypes.byref(first),
        1,
        stream.cuda_stream,
    )
    if error:
        stream.synchronize()
        raise RuntimeError(f"CUDA batch PLE gradient copy failed with error {error}")
