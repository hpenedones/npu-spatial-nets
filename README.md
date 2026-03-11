# TileFlow: Neural Networks Co-Designed with NPU Hardware

A character-level language model whose architecture maps **exactly** to the
AMD Ryzen AI NPU — 32 layers on 32 tiles, no approximations.

```
for each character:
    for each block g = 0..7:
        CPU:  h_b = RMSNorm(h) + embed(char) + bias_g
        NPU:  h_b = ReLU(ReLU(ReLU(ReLU(h_b @ W[4g]) @ W[4g+1]) @ W[4g+2]) @ W[4g+3])
        CPU:  h = h + h_b   (residual connection)
    CPU:  logits = h @ W_out + b_out
```

## Results

| Model | Params | Val Loss | Perplexity | Device |
|-------|--------|----------|------------|--------|
| **TileFlow block-RNN** | 542K | 2.42 | 11.2 | NPU (32 tiles) |
| Transformer baseline | 818K | 1.89 | 6.6 | GPU |

| Metric | Value |
|--------|-------|
| Throughput (384 parallel sequences) | **89,600 chars/s** |
| Throughput (single sequence) | 233 chars/s |
| NPU tiles used | 32 (8 columns × 4 rows) |
| Latency per NPU call | 0.14 ms |

## Quick Start

```bash
# Train on GPU (~18 min for 10 epochs)
HSA_OVERRIDE_GFX_VERSION=11.0.0 python -m char_lm.train --epochs 10

# Generate text on CPU
python -m char_lm.generate --device cpu --prompt "KING RICHARD"

# Generate text on NPU (32 tiles, 384 parallel sequences)
python -m char_lm.generate --device npu --prompt "KING RICHARD"
```

## How It Works

The NPU is a spatial-dataflow processor: 32 small compute tiles, each with
64 KB of SRAM and a bfloat16 matrix-multiply unit, connected by hardware
FIFOs. Instead of compiling an arbitrary model to this hardware, we design
the model to match it:

- **32 layers = 32 tiles.** 8 blocks of 4 layers, one block per pipeline call.
- **4 rows = 4-stage pipeline.** Within a block, data flows tile-to-tile
  through ObjectFIFOs — no DDR traffic.
- **8 columns = 8 batch slices.** 48 samples per column, 384 total.
- **H=128 fits SRAM.** Weight (32 KB) + activations (2×12 KB) + stack = 57 KB < 64 KB.

Between blocks, the CPU does what the pipeline can't: RMSNorm, embedding
injection, bias, and residual connections. The model trains with this exact
structure — what you train is what you deploy.

See the [whitepaper](docs/tileflow_whitepaper.pdf) for full details including
hardware background and NPU programming concepts.

## Project Structure

```
char_lm/
├── model.py               # Block-recurrent char LM (RecurrentCharLM)
├── train.py               # GPU training loop
├── generate.py            # Text generation on CPU or NPU
├── transformer_baseline.py # Quality reference
└── data.py                # Shakespeare dataset

spatial_mlp/
├── pipeline_design.py     # IRON/MLIR-AIE: 32-tile pipeline (8×4)
├── pipeline_op.py         # Operator: compilation + runtime
└── pipeline_test.py       # NPU benchmark

aie_kernels/
├── matmul_relu.cc         # Fused C = ReLU(A × B)
└── mlp_kernels.cc         # copy_bf16 support kernel
```

## Requirements

- **NPU**: AMD Ryzen AI 9 HX 370 (or any XDNA 2 / Strix Point APU)
- **OS**: Linux, kernel 6.11+ with `amdxdna` driver
- **Training**: AMD iGPU (ROCm) or NVIDIA GPU (CUDA)
- **Toolchain**: [IRON](https://github.com/amd/IRON) /
  [MLIR-AIE](https://github.com/Xilinx/mlir-aie) /
  [XRT](https://github.com/amd/xdna-driver)

## References

- [IRON](https://github.com/amd/IRON) — close-to-metal NPU programming
- [MLIR-AIE programming guide](https://github.com/Xilinx/mlir-aie/tree/main/programming_guide)
- [Linux kernel NPU docs](https://docs.kernel.org/accel/amdxdna/amdnpu.html)
- [Development logbook](logbook.md) — historical benchmarks and technical notes
