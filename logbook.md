# Development Logbook

This file records the project's development history — earlier architectures,
benchmark results, and technical discoveries made along the way. The current
system (resmlp) is described in the README and whitepaper. Earlier experimental
code (char_lm, spatial_mlp) lives on the `experimental` branch.

---

## Phase 1: Single Large GEMM (baseline)

**Goal:** Understand the NPU's raw matmul throughput.

Mapped a single large GEMM across all 32 tiles using IRON/MLIR-AIE.

| Metric | Value |
|--------|-------|
| TFLOPS | 2.49 |
| Peak % | 10% |
| CPU speedup | 1.4× |

**Lesson:** The NPU is memory-bandwidth-limited when data streams from DDR.
A single matmul loads weights, computes, and writes back — most time is spent
on DMA, not arithmetic. The NPU wins only when data **stays on-chip**.

---

## Phase 2: Pipeline MLP (4-stage feedforward)

**Goal:** Chain multiple matmuls in a pipeline (data flows tile-to-tile).

Built a 4-stage pipeline where each tile holds a different weight matrix.
Input enters at stage 1, flows through `ReLU(x @ W₁) → ReLU(· @ W₂) → ...`,
and exits at stage 4. Used all 32 tiles (8 columns × 4 rows).

| Metric | Value |
|--------|-------|
| TFLOPS | 0.13 |
| Peak % | 0.5% |
| CPU speedup | 0.87× (slower than CPU!) |

**Lesson:** The pipeline processes each input only once through 4 matmuls.
With H=128, that's 4 × 128² × 2 = 131K FLOPs per input — completed in
microseconds, but the ~120 μs driver overhead per NPU call dwarfs the compute.
Throughput is bottlenecked by invocation overhead, not arithmetic.

---

## Phase 3: Recurrent MLP (same-W hardware loop) — Peak Throughput

**Goal:** Amortise invocation overhead by looping on-chip.

A single weight matrix W (128×128, bfloat16) is loaded once into each tile's
SRAM and applied repeatedly in a tight hardware loop:

```
x = input                     # loaded from DDR, 48×128 bf16
for i in range(depth):        # depth=1000
    y = ReLU(x @ W)           # fused matmul+ReLU
    x = ReLU(y @ W)           # ping-pong buffers
output = x                    # drained to DDR
```

Mapped to 24 tiles (3 rows × 8 columns). Row 5 was unused due to MemTile
routing constraints (~6 northward master ports per MemTile).

### Optimisation journey

| Configuration | TFLOPS | Peak % | GFLOPS/tile | CPU Speedup |
|---------------|--------|--------|-------------|-------------|
| B=16, separate kernels | 8.98 | 35.9% | 374 | 18.5× |
| B=32, separate kernels | 13.47 | 53.9% | 561 | 19.5× |
| B=48, separate kernels | 15.98 | 63.9% | 666 | 17.6× |
| **B=48, fused matmul+ReLU** | **23.93** | **95.7%** | **997** | **25.9×** |

All measurements: 24 tiles, H=128, depth=1000, bfloat16.

### Key optimisation insights

**Batch size (B=16 → B=48):** The MMUL unit processes 8×8 blocks with a 2×2
expansion pattern. At B=16 the outer loop has only 1 iteration — too few for
the chess compiler to pipeline effectively. At B=48 (3 iterations), the
compiler overlaps loads, multiplications, and stores. +78% throughput.

**Fused matmul+ReLU kernel:** Replaced 3 separate kernel calls (zero output
buffer, accumulate C += A×B, apply ReLU in-place) with a single fused call
that zero-initialises accumulators in registers, accumulates, and applies ReLU
during the store phase. +50% throughput at B=48.

**Non-linear interaction:** The fused kernel was 10% *slower* at B=16 (8.03 vs
8.98 TFLOPS) but 50% *faster* at B=48 (23.93 vs 15.98). The compiler needs
enough outer-loop iterations to schedule the fused pipeline.

### Remaining gap to peak (4.3%)

- **BFP16 emulation:** MMUL uses block-floating-point emulation for bf16 with
  r=s=t=8 tile factors. Reformatting exponents costs a few cycles per block.
- **Array utilisation:** 24 of 32 tiles (75%) due to MemTile routing.
- **Invocation overhead:** ~120 μs fixed per NPU call, ~8% at depth=1000.

---

## Phase 4: Block-Recurrent Character LM

This phase traded peak TFLOPS for a richer architecture with 32 distinct weight
matrices, RMSNorm, residual connections, and character embeddings — an actual
language model trained on Shakespeare.

### Architecture

```
for each character:
    for each block g = 0..7:
        CPU:  h_b = h + embed(char) + bias_g
        NPU:  for stage j = 0..3:
                  h_b = ReLU(RMSNorm(h_b) @ W[4g+j])   ← fused on-chip
        CPU:  h = h + h_b   (residual connection)
    CPU:  logits = h @ W_out + b_out
```

- **32 layers = 32 tiles.** 8 blocks of 4 layers, one block per pipeline call.
- **4 rows = 4-stage pipeline.** Within a block, data flows tile-to-tile
  through ObjectFIFOs — no DDR traffic.
- **Each stage fuses RMSNorm + matmul + ReLU.** Per-layer normalisation
  prevents activation explosion.
- **8 columns = 8 batch slices.** 48 samples per column, 384 total.
- **H=128 fits SRAM.** Weight+scale (32.25 KB) + activations (2×12 KB) = 58 KB < 64 KB.

### Results

| Model | Params | Val Loss | Perplexity | Device |
|-------|--------|----------|------------|--------|
| **TileFlow block-RNN** | 542K | 2.03 | 7.6 | NPU (32 tiles) |
| Transformer baseline | 818K | 1.89 | 6.6 | GPU |

| Metric | Value |
|--------|-------|
| Throughput (384 parallel sequences) | **89,600 chars/s** |
| Throughput (single sequence) | 233 chars/s |
| NPU tiles used | 32 (8 columns × 4 rows) |
| Latency per NPU call | 0.14 ms |

### Quality progression

| Model variant | Val Loss | Perplexity | Notes |
|---------------|----------|------------|-------|
| No bias, no residual | 3.54 | 34.5 | ReLU threshold stuck at zero |
| With bias | 3.35 | 28.5 | Shifts ReLU threshold |
| + Residual + post-norm | 3.08 | 21.8 | Gradient flow through 32 layers |
| + Pre-norm + input injection (per-layer) | **1.94** | **6.9** | Each layer sees embedding |
| Blocked (4-layer groups, NPU-compatible) | **2.42** | **11.2** | Trade-off for NPU mapping |
| Transformer baseline (818K params) | 1.89 | 6.6 | Reference |

### NaN debugging

The 32-layer residual model (`h = h + ReLU(h @ W + b)`) produced NaN during
training. Root cause: hidden state norm grew ~9× per timestep through the 64-
character BPTT window (0.22 → 81 → NaN by step 3 of the sequence). Fix:
RMSNorm after the recurrence (post-norm) prevented cross-timestep explosion.
Pre-norm before each layer prevented within-timestep explosion.

### Code (archived)

```
char_lm/
├── model.py               # Block-recurrent char LM (RecurrentCharLM)
├── train.py               # GPU training loop
├── generate.py            # Text generation on CPU or NPU
├── transformer_baseline.py # Quality reference
└── data.py                # Shakespeare dataset
```

A whitepaper (`docs/tileflow_whitepaper.pdf`) was also produced for this phase,
generated by `docs/generate_whitepaper.py`. It covers the hardware background,
IRON programming model, and the block-recurrent architecture in detail.

---

## Hardware Constraints Discovered

These non-obvious constraints shaped the NPU designs:

1. **No FIFO ops inside loops:** Placing `acquire()` / `release()` inside
   `range_()` (which compiles to `scf.for`) causes DMA deadlock. All FIFO
   operations must happen outside the loop.

2. **DMA BD 10-bit size limit:** Shim DMA buffer-descriptor sizes are 10-bit
   (max 1024). For B=48, H=128, B×H=6144 exceeds this, so tensor access
   patterns must factor dimensions as [B, H] = [48, 128] instead of [6144].

3. **Fused kernel needs ≥3 loop iterations:** The chess compiler pipelines the
   fused matmul+ReLU efficiently only when M/(2r) ≥ 3, meaning B ≥ 48 for
   r=8. At B=16, the fused kernel is 10% slower than separate kernels.

4. **MemTile routing limit:** ~6 northward master ports per MemTile. At 3 data
   streams per compute row (weight + input + output), 3 rows = 9 streams fits;
   4 rows = 12 streams causes the MLIR-AIE router to fail.

5. **ROCm stderr escapes:** ROCm/HIP emits escape sequences to stderr during
   CUDA init that some terminals interpret as bell characters. Fix: redirect
   stderr with `2>/dev/null` on GPU-using commands.

---

## Multi-Row Data Routing (Phase 3)

When using multiple compute rows, data passes through MemTiles (row 1):

- **Weights** are *broadcast* via `forward()`: one DDR→MemTile transfer, then
  MemTile fans out to all rows.
- **Inputs** are *split* via `split()`: host buffer partitioned per row.
- **Outputs** are *joined* via `join()`: per-row results aggregated back.

The pipeline design (Phase 4) uses a different routing pattern: 4 rows per
column form a pipeline chain (stage 1 → 2 → 3 → 4), with data flowing
north-to-south through ObjectFIFOs.

---

## Phase 5: Clean 32-Layer Residual MLP (current)

**Goal:** Start fresh with the simplest possible NPU-native network — one
weight matrix per tile, no RMSNorm, no embeddings inside the NPU loop.

Rewrote everything from scratch as the `resmlp/` package. Key simplification:
each tile computes `y = relu(x @ W) + x` (residual skip connection). Data flows
through all 32 tiles in a serpentine ("snake") path. Trained on MNIST as a
didactic end-to-end example.

### Results

| Metric | Value |
|--------|-------|
| Model parameters | 946K (32 × 160 × 160 + embed + head) |
| MNIST test accuracy (CPU) | 97.1% |
| MNIST test accuracy (NPU) | 97.16% |
| NPU latency per batch | 0.33 ms (batch of 8) |
| NPU throughput | 24,114 images/sec |
| NPU GFLOPS (32-tile pipeline) | 55 |

### Design choices

- **H=160** maximises SRAM usage: 160×160×2 = 50 KB weight + 2×8×160×2 = 5 KB
  activations = 55 KB of 64 KB per tile.
- **B=8** is the minimum batch that fills the 8×8×8 MMUL blocks (1 block in the
  M dimension). Keeps the model simple and latency low.
- **Snake routing** through all 32 tiles: col 0 bottom→top, col 1 top→bottom,
  etc. The IRON compiler handles MemTile routing automatically with plain
  ObjectFifo (no explicit forward/join).
- **BFP16 emulation** (8×8×8 blocks) gives uniform tiling for A, B, C matrices.
  Over 32 layers the quantisation error compounds (~20-30% element-wise vs f32)
  but doesn't affect classification accuracy.

### Lessons learned (vs Phase 4 snake attempt)

The old `spatial_mlp/snake_*.py` code produced all-zeros at B=8 and was never
fixed. The new `resmlp/` code works because:

1. Simpler kernel (no RMSNorm/scale vector → smaller weight buffer).
2. Plain ObjectFifo instead of explicit `forward()`/`join()` — the compiler
   routes through MemTile automatically.
3. Single task group (AXPY pattern) instead of separate fill/drain groups.
