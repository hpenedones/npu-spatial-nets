# NPU-Native Neural Networks

Neural networks co-designed with the AMD XDNA 2 NPU — 32 layers on 32 tiles,
programmed with [IRON](https://github.com/amd/IRON).

## resmlp: 32-Layer Residual MLP on MNIST

Each NPU tile holds one weight matrix and computes `y = relu(x @ W) + x`.
Data flows through all 32 tiles in a serpentine path. The model trains on CPU
in PyTorch and runs inference on the NPU.

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

# Test NPU pipeline correctness
python -m resmlp.test

# Run MNIST inference on NPU
python -m resmlp.infer resmlp/checkpoints/resmlp_epoch009.pt --bench
```

## Project Structure

```
resmlp/
├── model.py       # PyTorch model: embed → 32 × ResidualLinear → head
├── train.py       # MNIST training
├── infer.py       # NPU inference with trained weights
├── design.py      # IRON snake pipeline (32 tiles, 8 cols × 4 rows)
├── op.py          # IRON operator wrapper
└── test.py        # Correctness tests + benchmark

aie_kernels/
└── matmul_relu_skip.cc   # Fused kernel: c = relu(a @ w) + a
```

See [logbook.md](logbook.md) for earlier experiments (peak throughput
benchmarking, character language model, hardware constraints discovered).

## Requirements

- **NPU**: AMD Ryzen AI (XDNA 2 / Strix Point) — e.g. Ryzen AI 9 HX 370
- **OS**: Linux, kernel 6.11+ with `amdxdna` driver
- **Toolchain**: [IRON](https://github.com/amd/IRON) /
  [XRT](https://github.com/amd/xdna-driver)

## References

- [IRON](https://github.com/amd/IRON) — close-to-metal NPU programming
- [MLIR-AIE programming guide](https://github.com/Xilinx/mlir-aie/tree/main/programming_guide)
- [Development logbook](logbook.md) — full history and technical notes
