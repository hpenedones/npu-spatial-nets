"""
Residual MLP on AMD XDNA 2 NPU.

A 32-layer deep residual network where each compute tile holds one layer's
weight matrix and applies:  y = relu(x @ W) + x

The activations flow through all 32 tiles in a serpentine ("snake") path
while the weights remain static in each tile's 64 KB SRAM.

Architecture (8 columns × 4 rows = 32 tiles):

    DDR ──► Tile(0,2)→(0,3)→(0,4)→(0,5)
                                     │ (cross-column)
            (1,2)←(1,3)←(1,4)←(1,5) ◄┘
             │
             ► (2,2)→(2,3)→(2,4)→(2,5)
                                     │
             ...continues snake...   │
             ◄─────────────────────────
            (7,2)←(7,3)←(7,4)←(7,5) ──► DDR

SRAM budget per tile (B=8, H=160, bfloat16):
    Weight matrix:  160×160×2 = 50.0 KB
    Input buffer:   8×160×2   =  2.5 KB
    Output buffer:  8×160×2   =  2.5 KB
    ─────────────────────────────────────
    Total:                      55.0 KB  (of 64 KB)
"""

import numpy as np

TILE_BLOCK = 8  # AIE mmul block size (8×8×8 with BFP16 emulation)


def round_up_to_tile_multiple(value, block=TILE_BLOCK):
    """Round ``value`` up to the next ``block`` multiple."""
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    return ((value + block - 1) // block) * block


def to_tiled(mat, br=TILE_BLOCK, bc=TILE_BLOCK):
    """Row-major (M, K) matrix → tiled [M/br, K/bc, br, bc] flat layout.

    The AIE matmul intrinsic reads data in blocked format. This function
    rearranges a standard 2D array into the expected flat memory order.
    """
    M, K = mat.shape
    assert M % br == 0 and K % bc == 0, \
        f"({M}, {K}) must be divisible by ({br}, {bc})"
    return (mat.reshape(M // br, br, K // bc, bc)
            .transpose(0, 2, 1, 3)
            .reshape(-1))


def from_tiled(flat, M, K, br=TILE_BLOCK, bc=TILE_BLOCK):
    """Inverse of to_tiled: tiled flat buffer → row-major (M, K) matrix."""
    assert M % br == 0 and K % bc == 0
    return (flat.reshape(M // br, K // bc, br, bc)
            .transpose(0, 2, 1, 3)
            .reshape(M, K))
