# TileFlow: Spatial Neural Networks on AMD XDNA 2 NPU

TileFlow is an experimental project for hardware-software co-design on the
AMD Ryzen AI NPU (XDNA 2 / Strix Point architecture). It uses the
[IRON/MLIR-AIE](https://github.com/amd/IRON) toolchain to program the NPU
at the individual tile level — explicitly mapping neural network layers to
physical compute tiles and wiring them together with hardware data streams.

The goal: design a neural network architecture that maps **exactly** to the
NPU's 2D tile array and demonstrate inference throughput approaching the
chip's theoretical **50 TOPS** (INT8) peak — orders of magnitude faster than
CPU execution of the same network.

## The Hardware

The XDNA 2 NPU in the Ryzen AI 9 HX 370 is a **spatial dataflow computer**:

```
        Col 0    Col 1    Col 2    Col 3    Col 4    Col 5    Col 6    Col 7
       ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
Row 3  │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │
       │ (0,3)  │ (1,3)  │ (2,3)  │ (3,3)  │ (4,3)  │ (5,3)  │ (6,3)  │ (7,3)  │
       ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
Row 2  │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │
       │ (0,2)  │ (1,2)  │ (2,2)  │ (3,2)  │ (4,2)  │ (5,2)  │ (6,2)  │ (7,2)  │
       ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
Row 1  │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │
       │ (0,1)  │ (1,1)  │ (2,1)  │ (3,1)  │ (4,1)  │ (5,1)  │ (6,1)  │ (7,1)  │
       ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
Row 0  │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │
       │ (0,0)  │ (1,0)  │ (2,0)  │ (3,0)  │ (4,0)  │ (5,0)  │ (6,0)  │ (7,0)  │
       └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
         Mem 0    Mem 1    Mem 2    Mem 3    Mem 4    Mem 5    Mem 6    Mem 7
                                   (L2 Memory Tiles — 4 MB total)
```

| Property | Value |
|---|---|
| Compute tiles | 32 (8 columns × 4 rows) |
| Per-tile SRAM | ~48 KB |
| Per-tile compute | 512 INT8 MACs/cycle (VLIW+SIMD) |
| Clock | ~1.5 GHz |
| Peak throughput | **50 TOPS** (INT8), ~25 TFLOPS (BF16) |
| Achieved (GEMM) | ~38 TOPS INT8 (76% efficiency) |
| L2 memory | 4 MB (8 memory tiles, one per column) |
| Interconnect | N/S/E/W nearest-neighbor stream switches |
| Power | ~6 W |

Each tile is an independent processor with its own local memory and direct
hardware data links to its four neighbors. Data moves tile-to-tile through
programmable stream switches — no shared bus, no cache hierarchy, no
contention.

## The Concept: Spatial Pipelining

Instead of running a neural network layer-by-layer on a single processor
(temporal computing), we **spread the network across the physical tile array**
so all tiles compute simultaneously:

```
Input → [Col 0: Layer 1] → [Col 1: Layer 2] → ... → [Col 7: Layer 8] → Output
          (4 tiles)          (4 tiles)                  (4 tiles)

         Each column: 4 tiles cooperate on one layer (data-parallel split)
         Between columns: ObjectFIFO hardware streams carry activations
```

- **8 pipeline stages** = 8 columns, one neural network layer per column
- **4-way data parallelism** = 4 rows per column, each tile processes a
  slice of the layer's computation
- **Weights in tile-local SRAM** — no DDR access during inference
- **INT8 arithmetic** — maximizes throughput at 512 MACs/cycle/tile
- **ObjectFIFOs** — hardware ring buffers for zero-copy inter-tile streaming

Once the pipeline is full, one inference result exits **every pipeline
cycle** — all 32 tiles are active, all interconnects are carrying data.

## Toolchain

This project uses the **IRON** Python API — the close-to-metal programming
model for AMD AIE tiles. IRON compiles to MLIR-AIE, then to per-tile ELF
binaries, and loads them onto the NPU via XRT.

**This is NOT the Vitis AI / ONNX Runtime path.** Vitis AI treats the NPU as
a black box and doesn't give tile-level control. IRON gives full explicit
control over tile placement, data movement, buffer sizes, and kernel code —
which is what we need for true spatial pipelining.

| Component | Role |
|---|---|
| [IRON](https://github.com/amd/IRON) | Python API for tile layout + dataflow |
| [MLIR-AIE](https://github.com/Xilinx/mlir-aie) | MLIR dialect → hardware compilation |
| [Peano/LLVM-AIE](https://github.com/Xilinx/llvm-aie) | C++ compiler for per-tile kernels |
| [XRT](https://github.com/amd/xdna-driver) | Runtime for loading/executing on NPU |

## Project Phases

- [ ] **Phase 0 — Toolchain Setup**: Install IRON/MLIR-AIE, verify NPU access,
  run a minimal passthrough example end-to-end.
- [ ] **Phase 1 — Peak Throughput**: Run whole-array INT8 matrix multiplication
  on all 32 tiles. Measure TOPS vs theoretical peak.
- [ ] **Phase 2 — Spatial Neural Network**: Design and implement an 8-layer
  pipelined MLP mapped to the 8×4 tile grid.
- [ ] **Phase 3 — Benchmark**: Compare NPU inference latency/throughput against
  CPU (and GPU). Quantify speedup.
- [ ] **Phase 4 — Training & Applications**: Explore backpropagation on NPU
  tiles. Pick a real ML task (time-series, FEM, signal processing).

## Hardware Requirements

- **Processor**: AMD Ryzen AI 9 HX 370 (or any XDNA 2 / Strix Point APU)
- **OS**: Linux, kernel 6.11+ with `amdxdna` driver
- **NPU device**: `/dev/accel/accel0` must be accessible
- **NPU firmware**: `/lib/firmware/amdnpu/`
- **Runtime**: XRT (built from [xdna-driver](https://github.com/amd/xdna-driver))

See [fastflowlm-docker](https://github.com/hpenedones/fastflowlm-docker) for
a working Docker-based NPU setup on this hardware.

## References

- [IRON repo](https://github.com/amd/IRON) — close-to-metal NPU programming
- [MLIR-AIE programming guide](https://github.com/Xilinx/mlir-aie/tree/main/programming_guide)
- [Whole-array matmul example](https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/matrix_multiplication/whole_array)
- [GEMM optimization on XDNA (arXiv)](https://arxiv.org/html/2512.13282v1) — achieved ~38 TOPS
- [NPU training (arXiv)](https://arxiv.org/html/2504.03083v1) — backprop on AIE tiles
- [Linux kernel NPU docs](https://docs.kernel.org/accel/amdxdna/amdnpu.html)
- [IRON tutorial (IPDPS 2025)](https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/iron-for-ryzen-ai-tutorial-ipdps-2025.pdf)
