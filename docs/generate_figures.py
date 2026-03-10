#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate architecture diagrams for the TileFlow white paper.

Produces three figures as PDF files in docs/:

1. ``xdna2_hardware.pdf``  — Physical tile array of the XDNA 2 NPU
2. ``recurrent_mlp.pdf``   — Recurrent MLP dataflow mapped to the array
3. ``performance.pdf``     — Throughput scaling across tile counts

Usage::

    python docs/generate_figures.py

All figures use matplotlib with no external dependencies beyond numpy.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# ── Color palette ─────────────────────────────────────────────────────────

COLORS = {
    "compute":  "#4A90D9",   # blue
    "memtile":  "#F5A623",   # amber
    "shim":     "#7B8D8E",   # grey
    "active":   "#2ECC71",   # green (active tiles)
    "inactive": "#E8E8E8",   # light grey
    "weight":   "#E74C3C",   # red (weight data path)
    "input":    "#3498DB",   # blue (input data path)
    "output":   "#2ECC71",   # green (output data path)
    "bg":       "#FAFAFA",   # background
}


# ── Figure 1: XDNA 2 Hardware Architecture ───────────────────────────────

def draw_hardware_architecture():
    """Draw the XDNA 2 tile array with row/column labels and annotations."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(-1.5, 9.5)
    ax.set_ylim(-1.0, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    tile_w, tile_h = 1.0, 0.85
    gap = 0.15

    row_labels = {
        0: "Row 0\nShim",
        1: "Row 1\nMemTile",
        2: "Row 2",
        3: "Row 3",
        4: "Row 4",
        5: "Row 5",
    }
    row_types = {
        0: "shim",
        1: "memtile",
        2: "compute",
        3: "compute",
        4: "compute",
        5: "compute",
    }
    row_text = {
        0: "DMA",
        1: "512 KB",
        2: "64 KB\nSRAM",
        3: "64 KB\nSRAM",
        4: "64 KB\nSRAM",
        5: "64 KB\nSRAM",
    }

    for row in range(6):
        y = row * (tile_h + gap)
        color = COLORS[row_types[row]]

        # Row label
        ax.text(-1.2, y + tile_h / 2, row_labels[row],
                ha="center", va="center", fontsize=8, fontweight="bold")

        for col in range(8):
            x = col * (tile_w + gap)
            rect = FancyBboxPatch(
                (x, y), tile_w, tile_h,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="#333333",
                linewidth=0.8, alpha=0.85,
            )
            ax.add_patch(rect)

            # Tile text
            fontsize = 6 if row >= 2 else 7
            ax.text(x + tile_w / 2, y + tile_h / 2, row_text[row],
                    ha="center", va="center", fontsize=fontsize, color="white",
                    fontweight="bold")

    # Column labels
    for col in range(8):
        x = col * (tile_w + gap)
        ax.text(x + tile_w / 2, 6 * (tile_h + gap) + 0.15,
                f"Col {col}", ha="center", fontsize=8, fontweight="bold")

    # Legend
    legend_items = [
        mpatches.Patch(color=COLORS["compute"], label="Compute Tile (×32)"),
        mpatches.Patch(color=COLORS["memtile"], label="Memory Tile (×8, 512 KB each)"),
        mpatches.Patch(color=COLORS["shim"], label="Shim Tile (×8, DDR DMA)"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8,
              framealpha=0.9)

    # Title and annotations
    ax.set_title("AMD XDNA 2 NPU Tile Array — Ryzen AI 9 HX 370",
                 fontsize=14, fontweight="bold", pad=15)

    # Annotation: data flow direction
    ax.annotate("", xy=(9.0, 0.4), xytext=(9.0, 5.0),
                arrowprops=dict(arrowstyle="<->", color="#666",
                                lw=1.5, ls="--"))
    ax.text(9.3, 2.7, "Data\nflow", ha="center", va="center",
            fontsize=7, color="#666", style="italic")

    # Annotation: DDR connection
    ax.annotate("Host DDR", xy=(4.0, -0.7), fontsize=9, ha="center",
                color=COLORS["shim"], fontweight="bold")

    fig.tight_layout()
    path = OUTPUT_DIR / "xdna2_hardware.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


# ── Figure 2: Recurrent MLP Dataflow ─────────────────────────────────────

def draw_recurrent_mlp():
    """Draw the recurrent MLP architecture mapped to the NPU tile array."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_xlim(-2.5, 12)
    ax.set_ylim(-2.0, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    tile_w, tile_h = 1.1, 0.9
    gap = 0.15

    # Draw the 3-row × 8-col active array + MemTile + Shim
    rows_config = {
        0: ("shim", COLORS["shim"], "DMA"),
        1: ("memtile", COLORS["memtile"], "MemTile"),
        2: ("compute", COLORS["active"], "W + loop"),
        3: ("compute", COLORS["active"], "W + loop"),
        4: ("compute", COLORS["active"], "W + loop"),
        5: ("compute", COLORS["inactive"], "unused"),
    }

    for row, (rtype, color, label) in rows_config.items():
        y = row * (tile_h + gap)
        alpha = 0.4 if rtype == "compute" and color == COLORS["inactive"] else 0.85

        for col in range(8):
            x = col * (tile_w + gap)
            rect = FancyBboxPatch(
                (x, y), tile_w, tile_h,
                boxstyle="round,pad=0.04",
                facecolor=color, edgecolor="#333",
                linewidth=0.8, alpha=alpha,
            )
            ax.add_patch(rect)
            fontcolor = "white" if alpha > 0.6 else "#999"
            ax.text(x + tile_w / 2, y + tile_h / 2, label,
                    ha="center", va="center", fontsize=5.5,
                    color=fontcolor, fontweight="bold")

    # Row labels
    for row, text in [(0, "Row 0: Shim (DMA)"), (1, "Row 1: MemTile"),
                      (2, "Row 2: Compute"), (3, "Row 3: Compute"),
                      (4, "Row 4: Compute"), (5, "Row 5: (unused)")]:
        y = row * (tile_h + gap) + tile_h / 2
        ax.text(-1.8, y, text, ha="center", va="center", fontsize=7,
                fontweight="bold" if row <= 4 else "normal",
                color="#333" if row <= 4 else "#999")

    # Data flow arrows for column 0 (representative)
    col = 0
    x_center = col * (tile_w + gap) + tile_w / 2

    # Input path: DDR → Shim → MemTile → split → tiles
    arrow_kw = dict(arrowstyle="-|>", lw=2.0, mutation_scale=12)

    # Weight arrow (red): DDR → Shim → MemTile → broadcast
    wx = x_center - 0.2
    for (y1, y2) in [(0.9, 1.05), (1.95, 2.1), (2.95, 3.15), (3.95, 4.1)]:
        ax.annotate("", xy=(wx, y2), xytext=(wx, y1),
                    arrowprops=dict(**arrow_kw, color=COLORS["weight"]))

    # Input arrow (blue): DDR → Shim → MemTile → split
    ix = x_center + 0.0
    for (y1, y2) in [(0.9, 1.05), (1.95, 2.1), (1.95, 3.15), (1.95, 4.1)]:
        ax.annotate("", xy=(ix, y2), xytext=(ix, y1),
                    arrowprops=dict(**arrow_kw, color=COLORS["input"]))

    # Output arrow (green): tiles → join → Shim → DDR
    ox = x_center + 0.2
    for (y1, y2) in [(2.1, 1.95), (3.15, 1.95), (4.1, 1.95), (1.05, 0.9)]:
        ax.annotate("", xy=(ox, y2), xytext=(ox, y1),
                    arrowprops=dict(**arrow_kw, color=COLORS["output"]))

    # Compute tile detail box (expanded view)
    detail_x, detail_y = 9.5, 3.0
    detail_w, detail_h = 2.2, 4.0
    detail_rect = FancyBboxPatch(
        (detail_x, detail_y), detail_w, detail_h,
        boxstyle="round,pad=0.1",
        facecolor="#F8F9FA", edgecolor=COLORS["active"],
        linewidth=2.0,
    )
    ax.add_patch(detail_rect)
    ax.text(detail_x + detail_w / 2, detail_y + detail_h - 0.25,
            "Single Tile Detail", ha="center", fontsize=9,
            fontweight="bold", color=COLORS["active"])

    # Detail contents
    detail_lines = [
        ("W (32 KB)", 3.2, COLORS["weight"]),
        ("", 2.8, "#333"),
        ("x = acquire(input)", 2.4, COLORS["input"]),
        ("y = acquire(output)", 2.1, COLORS["output"]),
        ("", 1.8, "#333"),
        ("for i in range(N):", 1.5, "#333"),
        ("  y = ReLU(x @ W)", 1.2, "#333"),
        ("  x = ReLU(y @ W)", 0.9, "#333"),
        ("", 0.6, "#333"),
        ("copy x → output", 0.3, COLORS["output"]),
    ]
    for text, dy, color in detail_lines:
        if text:
            ax.text(detail_x + 0.15, detail_y + dy, text,
                    fontsize=7, fontfamily="monospace", color=color)

    # Arrow from tile to detail
    ax.annotate("", xy=(detail_x - 0.1, detail_y + detail_h / 2),
                xytext=(8 * (tile_w + gap) - 0.3, 3 * (tile_h + gap) + tile_h / 2),
                arrowprops=dict(arrowstyle="->", color="#999", lw=1.0,
                                connectionstyle="arc3,rad=0.2"))

    # Legend
    legend_items = [
        mpatches.Patch(color=COLORS["weight"], label="Weight (broadcast)"),
        mpatches.Patch(color=COLORS["input"], label="Input (split)"),
        mpatches.Patch(color=COLORS["output"], label="Output (join)"),
        mpatches.Patch(color=COLORS["active"], label="Active compute tile"),
        mpatches.Patch(color=COLORS["inactive"], label="Unused (routing limit)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=7,
              framealpha=0.9)

    # DDR label
    ax.text(4.5, -1.2, "Host DDR Memory",
            ha="center", fontsize=10, fontweight="bold", color=COLORS["shim"])
    ax.annotate("", xy=(4.5, -0.1), xytext=(4.5, -0.8),
                arrowprops=dict(arrowstyle="<->", color=COLORS["shim"], lw=1.5))

    ax.set_title("Recurrent MLP Mapped to XDNA 2 — 24 Tiles (3×8)\n"
                 "Weight held in SRAM, hardware loop, ping-pong buffers",
                 fontsize=13, fontweight="bold", pad=15)

    fig.tight_layout()
    path = OUTPUT_DIR / "recurrent_mlp.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


# ── Figure 3: Performance Scaling ─────────────────────────────────────────

def draw_performance():
    """Draw throughput scaling chart: tiles vs TFLOPS with CPU baseline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    # Data from benchmarks
    tiles = np.array([8, 16, 24])
    tflops = np.array([2.89, 5.74, 8.95])
    speedup = np.array([12.2, 16.2, 20.4])
    gflops_per_tile = tflops * 1000 / tiles

    # Left plot: TFLOPS vs tiles
    ax1.bar(tiles, tflops, width=4, color=COLORS["active"], alpha=0.85,
            edgecolor="#333", linewidth=0.8, label="NPU throughput")
    ax1.axhline(y=25.0, color=COLORS["weight"], linestyle="--",
                linewidth=1.5, alpha=0.6, label="Theoretical peak (25 TFLOPS)")

    # Linear scaling reference
    ideal = tflops[0] / 8 * np.array([8, 16, 24, 32])
    ax1.plot([8, 16, 24, 32], ideal, "o--", color="#999",
             markersize=4, label="Linear scaling from 8 tiles")

    for i, (t, tf) in enumerate(zip(tiles, tflops)):
        ax1.text(t, tf + 0.3, f"{tf:.2f}", ha="center", fontsize=9,
                 fontweight="bold")

    ax1.set_xlabel("Number of Compute Tiles", fontsize=11)
    ax1.set_ylabel("Throughput (TFLOPS, bf16)", fontsize=11)
    ax1.set_title("NPU Throughput Scaling", fontsize=12, fontweight="bold")
    ax1.set_xticks([8, 16, 24, 32])
    ax1.set_ylim(0, 28)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Right plot: speedup vs tiles
    bars = ax2.bar(tiles, speedup, width=4, color=COLORS["input"], alpha=0.85,
                   edgecolor="#333", linewidth=0.8)
    ax2.axhline(y=1.0, color="#999", linestyle="-", linewidth=0.8)

    for i, (t, s) in enumerate(zip(tiles, speedup)):
        ax2.text(t, s + 0.5, f"{s:.1f}×", ha="center", fontsize=9,
                 fontweight="bold")

    ax2.set_xlabel("Number of Compute Tiles", fontsize=11)
    ax2.set_ylabel("Speedup vs CPU (PyTorch bf16)", fontsize=11)
    ax2.set_title("NPU vs CPU Speedup", fontsize=12, fontweight="bold")
    ax2.set_xticks([8, 16, 24])
    ax2.set_ylim(0, 25)
    ax2.grid(axis="y", alpha=0.3)

    # Add per-tile efficiency annotation
    ax2.text(0.98, 0.02,
             f"Per-tile: ~{gflops_per_tile.mean():.0f} GFLOPS\n"
             f"({gflops_per_tile.mean()/768*100:.0f}% of ~768 GFLOPS peak)",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=8, color="#666",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                       edgecolor="#ddd"))

    fig.suptitle("TileFlow: Recurrent MLP on AMD XDNA 2 NPU\n"
                 "H=128, B=16/tile, depth=2000, bfloat16",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUTPUT_DIR / "performance.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating TileFlow diagrams...")
    draw_hardware_architecture()
    draw_recurrent_mlp()
    draw_performance()
    print("Done.")
