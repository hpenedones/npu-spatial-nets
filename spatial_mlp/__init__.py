# SPDX-License-Identifier: Apache-2.0
"""
Spatial neural network primitives for the AMD XDNA 2 NPU.

The AIE2P matmul unit uses BFP16 emulation with tile factors r=s=t=8,
meaning matrices are processed in 8×8 blocks. All matrix dimensions
must be multiples of TILE_SIZE for correct DMA transfer and compute.
"""

import numpy as np

TILE_SIZE = 8  # BFP16 emulation block factor (r = s = t = 8)


def to_tiled(mat, block_rows=TILE_SIZE, block_cols=TILE_SIZE):
    """Convert a row-major matrix to the tiled (blocked) layout the AIE expects.

    The AIE matmul kernel reads matrices in [M/br, K/bc, br, bc] order,
    where br and bc are the block dimensions. This function rearranges a
    standard (M, K) array into a flat buffer with that tile order.

    Args:
        mat: 2D array of shape (M, K).
        block_rows: Tile height (must divide M).
        block_cols: Tile width (must divide K).

    Returns:
        Flat 1D array in tiled layout, same dtype as input.
    """
    M, K = mat.shape
    assert M % block_rows == 0 and K % block_cols == 0, (
        f"Dimensions ({M}, {K}) must be divisible by block size "
        f"({block_rows}, {block_cols})"
    )
    return (mat.reshape(M // block_rows, block_rows, K // block_cols, block_cols)
            .transpose(0, 2, 1, 3)
            .reshape(-1))


def from_tiled(flat, M, K, block_rows=TILE_SIZE, block_cols=TILE_SIZE):
    """Convert a tiled (blocked) flat buffer back to a row-major (M, K) matrix.

    Inverse of :func:`to_tiled`.

    Args:
        flat: 1D array in tiled layout.
        M: Number of rows in the output matrix.
        K: Number of columns in the output matrix.
        block_rows: Tile height (must divide M).
        block_cols: Tile width (must divide K).

    Returns:
        2D array of shape (M, K).
    """
    assert M % block_rows == 0 and K % block_cols == 0, (
        f"Dimensions ({M}, {K}) must be divisible by block size "
        f"({block_rows}, {block_cols})"
    )
    return (flat.reshape(M // block_rows, K // block_cols, block_rows, block_cols)
            .transpose(0, 2, 1, 3)
            .reshape(M, K))
