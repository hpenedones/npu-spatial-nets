# NPU-Native Neural Networks

Neural networks co-designed with the AMD XDNA 2 NPU — a 32-layer residual MLP,
with one layer per tile across 32 NPU tiles, programmed with
[IRON](https://github.com/amd/IRON).

## resmlp: 32-Layer Residual MLP on MNIST

Each NPU tile holds one weight matrix and computes `y = relu(x @ W) + x`.
Data flows through all 32 tiles in a serpentine path. The repo now supports
both the original hybrid trainer (CPU embed/head + NPU residual stack) and a
full-NPU training mode that uses the 32 compute tiles as
`embed + 30 residual + head`.

```
Input (784) → Linear → H=160
  → [relu(x @ W_i) + x] × 32 tiles    ← NPU snake pipeline
  → Linear → 10 classes
```

### Results

| Metric | Value |
|--------|-------|
| Parameters | 946K |
| MNIST test accuracy | 97.2% |
| NPU throughput | 24K images/sec |
| NPU latency | 0.33 ms / batch of 8 |

### Quick Start

```bash
# Train on CPU (~45 seconds)
python -m resmlp.train

# Train with the hybrid path (CPU embed/head, NPU residual stack)
python -m resmlp.train_npu --epochs 10

# Train with the full-NPU path (1 embed tile + 30 residual tiles + 1 head tile)
# This mode currently requires all 8 columns / 32 compute tiles.
# Note: current kernels use a fixed NPU SGD LR of 0.01.
# Current limitation: weights stay resident on-device, so host checkpoint export
# and CPU-side evaluation are not implemented for this mode yet.
python -m resmlp.train_npu --pipeline full-npu --epochs 10

# Run MNIST inference on NPU
python -m resmlp.infer resmlp/checkpoints/resmlp_hybrid_epoch009.pt --bench

# Test NPU pipeline correctness
python -m tests.test_training --cols 1
python -m tests.test_inference --cols 1
```

## Project Structure

```
resmlp/
├── __init__.py          # Tiled layout utilities (to_tiled / from_tiled)
├── model.py             # PyTorch model: embed → residual stack → head
├── train.py             # CPU-only MNIST training
├── train_npu.py         # Hybrid and full-NPU MNIST training entry points
├── infer.py             # NPU inference with trained weights
├── design.py            # IRON snake pipeline (inference: 32 tiles)
├── op.py                # IRON operator wrapper (inference)
├── training_design.py   # Hybrid training pipeline (32 residual tiles)
├── training_op.py       # Hybrid training operator wrapper
├── training_full_design.py # Full-NPU pipeline (embed + 30 residual + head)
└── training_full_op.py  # Full-NPU operator wrapper

aie_kernels/
├── matmul_relu_skip.cc  # Fused fwd kernel: c = relu(a @ w) + a
├── residual_backward.cc # Fused bwd kernel: grad + in-place SGD update
└── copy_activation.cc   # SRAM block copy utility

tests/
├── test_inference.py    # Snake pipeline correctness + benchmark
├── test_backward.py     # Single-layer backward validation
├── test_checkpoint.py   # Forward checkpoint probe
└── test_training.py     # Full training pipeline validation (fwd+bwd+SGD)
```

## Requirements

- **NPU**: AMD Ryzen AI (XDNA 2 / Strix Point) — e.g. Ryzen AI 9 HX 370
- **OS**: Linux, kernel 6.11+ with `amdxdna` driver
- **Toolchain**: [IRON](https://github.com/amd/IRON) /
  [XRT](https://github.com/amd/xdna-driver)

## References

- [IRON](https://github.com/amd/IRON) — close-to-metal NPU programming
- [MLIR-AIE programming guide](https://github.com/Xilinx/mlir-aie/tree/main/programming_guide)
- [Development logbook](logbook.md) — full history and technical notes
