# SPDX-License-Identifier: Apache-2.0
"""Spatial neural network primitives for the XDNA 2 NPU."""

import numpy as np

TILE_SIZE = 8  # r = s = t = 8 for bf16 with BFP16 emulation


def to_tiled(mat, block_r=TILE_SIZE, block_c=TILE_SIZE):
    """Convert row-major matrix to tiled (blocked) layout for AIE matmul."""
    M, K = mat.shape
    assert M % block_r == 0 and K % block_c == 0
    return (mat.reshape(M // block_r, block_r, K // block_c, block_c)
            .transpose(0, 2, 1, 3)
            .reshape(-1))


def from_tiled(flat, M, K, block_r=TILE_SIZE, block_c=TILE_SIZE):
    """Convert tiled (blocked) layout back to row-major matrix."""
    assert M % block_r == 0 and K % block_c == 0
    return (flat.reshape(M // block_r, K // block_c, block_r, block_c)
            .transpose(0, 2, 1, 3)
            .reshape(M, K))
