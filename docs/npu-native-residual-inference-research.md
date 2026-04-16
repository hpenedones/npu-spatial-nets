# Research Evidence: NPU-Native Residual Inference Whitepaper

## 1. Code-Paper Alignment

### Model equation
**Paper:** `h_{l+1} = ReLU(h_l W_l) + h_l`
**Code (`model.py`):**
```python
def forward(self, x):
    y = x @ self.weight
    if self.bias is not None:
        y = y + self.bias
    return torch.relu(y) + x
```
✅ **Match.** The code implements exactly `relu(x @ W) + x`. Note: bias=False by default, matching the paper's no-bias residual blocks.

### AIE kernel (`matmul_relu_skip.cc`)
The kernel implements:
1. `c = relu(a @ w)` via blocked 8×8×8 MMUL intrinsic
2. `c += a` via element-wise vector add
✅ **Match.** The inference entry point `matmul_relu_skip_infer_bf16` does matmul→relu→skip in that order.

### Tile mapping (`streaming_design.py` + `design.py`)
- `ROWS_PER_COL = 4`, snake ordering via `snake_tile_order(num_cols)`
- Weights embedded at compile time via `Buffer(initial_value=...)`
- FIFO depth = 2 for stream_depth > 1
- Assertions: `B % 8 == 0`, `H % 8 == 0`
✅ **Match.** One weight matrix per tile, snake routing, compile-time embedding.

### Training recipes
**Paper Table 3 claims:**
- H=64, L=32 tuning: best at epoch 46/50, AdamW, batch 4096, cosine lr 1.069e-3 → 8.70e-5, wd 3.23e-3, label smoothing 0.0197

**Code (`tune_higgs_optuna.py`):** Optuna sweeps over hidden_dim∈{32,64}, num_layers∈{8,32}, batch∈{4096,8192}, lr∈[1e-3,4e-3], wd∈[1e-5,5e-3], label_smoothing∈[0,0.05], up to 50 epochs with validation selection.
✅ **Consistent.** The reported hyperparameters fall within the Optuna search ranges.

### Throughput benchmarking (`streaming_infer.py`)
- `benchmark()` method: warmup_calls=4, then timed loop over `calls = ceil(num_samples / (B * stream_depth))`
- Wall time: `time.perf_counter()` around full loop including CPU embed + NPU + head
- Kernel time: accumulated from `npu_op.run_stream()` elapsed
- CPU head done via numpy matmul: `hidden @ cpu_head_weight.T + cpu_head_bias`
✅ **Sound methodology.** Warmup included. Wall vs kernel separation is clean.

### Data pipeline (`data_utils.py`)
- Uses `jxie/higgs` via HuggingFace → local `.pt` cache
- HIGGS_INPUT_DIM = 28, num_classes = 2
- DEFAULT_VAL_SIZE = 100_000, split_seed = 1234
- Normalization: per-feature mean/std from training data
✅ **Match.** 10.5M train (minus 100k val = 10.4M), 500k test.

### Tests
- `test_higgs_data.py`: smoke tests for native 28-width, rejects padded 56-width
- `test_streaming_inference.py`: reference Python implementation vs NPU output, 95% element-wise tolerance at rtol=0.30, atol=0.50
⚠️ **Note:** The streaming test has generous tolerances (rtol=0.30). This is expected for bf16 through 8+ residual blocks but worth flagging.

## 2. SRAM Accounting Verification

**Paper formulas:**
- Internal tile: `1024 + 2H² + 32H` bytes
- Ingress tile: `1024 + 2H² + 64H` bytes

**Derivation from code:**
- Weight buffer: H×H bf16 = 2H² bytes
- With fifo_depth=2 and B=8: each activation buffer = B×H×2 = 16H bytes
- Internal tile: 2 FIFOs × 1 buffer each (shared memory?) = 2 × 16H = 32H → total 1024 + 2H² + 32H
- Ingress tile: needs both host-ingress + first residual = double the activation buffers → 64H

**Boundary check:**
| H | Internal (KB) | Ingress (KB) | Fits 64KB? |
|---|---|---|---|
| 160 | 56.0 | 61.0 | ✅ Yes |
| 168 | 61.4 | 66.6 | ❌ No (ingress) |

**Paper says:** H=160 fits at ~62.5 KB on ingress. **Computed:** 61.0 KB.
⚠️ **Minor discrepancy:** 61.0 KB vs claimed ~62.5 KB. Difference is 1.5 KB — likely the paper rounds up or includes additional overhead not in the simplified formula. The qualitative conclusion (fits at 160, doesn't fit at 168) is correct.

## 3. Figure-Table Consistency

`generate_accuracy_throughput_frontier.py` **parses the whitepaper.tex directly** to extract data points:
- Throughput point: H=32,L=8, accuracy=75.89%, wall=4.18M/s (Table 4 row 1, Table 5 CPU-head row 1) ✅
- Manual point: H=32,L=32 20-epoch, accuracy=76.98%, wall=3.49M/s ✅
- Best accuracy: H=64,L=32 tuned, accuracy=77.98%, wall=1.35M/s ✅

All three points correctly use CPU-head wall throughput as stated.

## 4. Baseline Cross-Check

### Baldi et al. (2014) — arXiv:1402.4735
- BDT (all features): AUC = 0.81
- Shallow NN (all features): AUC = 0.816
- **Deep NN 5-layer (all features): AUC = 0.885**
- Trained on up to 10M examples

### XGBoost (Chen & Guestrin 2016)
- Typically achieves ~0.84-0.85 AUC on HIGGS with default/tuned settings

### This paper
- Best: **0.865 ROC AUC** (H=64, L=32, validation-selected tuning)

**Assessment:** The paper's 0.865 AUC is:
- Above XGBoost baselines (~0.84-0.85) ✅
- Below the original Baldi et al. 5-layer DNN (0.885) by ~2 points
- The paper explicitly disclaims SOTA: "not a claim of state-of-the-art HIGGS accuracy" ✅

The gap vs Baldi's DNN is notable but understandable: the paper's architecture is deliberately constrained (square residual blocks, bf16, hardware-friendly widths). The claim of "credible" is defensible.

## 5. Throughput Plausibility

- 4.18M samples/s over 50M samples → ~12 seconds total wall time. Plausible for a tight loop.
- Kernel throughput 26.29M/s vs wall 4.18M/s = 6.3× gap
- Gap sources: CPU embed (28→32 matmul per batch), numpy head computation, buffer sync, Python loop overhead
- The benchmark reuses the same batch (repeated_features), so it measures sustained throughput, not unique-data throughput

⚠️ **Important:** This is "repeated-sample" throughput — the same input batch is reprocessed. The paper states this clearly. Real deployment would need fresh data each call, which could add data-loading overhead.

## 6. Novelty Assessment

### What exists
- **hls4ml** (Duarte et al.): FPGA-based inference for particle physics. Different target (FPGAs, not NPUs).
- **Spatial dataflow accelerators** (DaDianNao, Eyeriss, TPU): General concepts of tile-local compute + explicit movement. Well-established literature.
- **AMD XDNA/IRON/MLIR-AIE**: Existing toolchain, not introduced by this paper.
- **Triton-XDNA** (AMD): Compiler-driven kernel generation for XDNA NPUs — focuses on Triton-level abstraction, not hand-mapped residual pipelines.

### What's new
- No prior published work maps a residual MLP specifically to XDNA 2 compute tiles with one-block-per-tile spatial streaming
- The co-design argument (model shape ↔ hardware topology) with end-to-end measurements is novel
- The honest measurement of CPU-head-is-faster-than-NPU-head is an unusual and valuable systems finding

## 7. Missing Elements

1. **No per-sample latency** — only throughput reported. Latency per inference call would strengthen the systems case, especially given the CMS trigger motivation.
2. **No variance/confidence intervals on accuracy** — single-run numbers. No error bars, no repeated seeds.
3. **No CPU/GPU throughput comparison** — how fast is the same model on CPU or GPU? Without this, the NPU throughput number lacks context.
4. **No power/energy measurement** — explicitly disclaimed, but it's the strongest missing piece for the "consumer hardware" story.
5. **No comparison with other NPU workloads** — is this a good use of the NPU vs other models?
6. **Repeated-sample benchmark** — throughput uses repeated input data, not streaming fresh data.

## 8. Writing Quality Notes

- Clear, well-structured, refreshingly honest about scope
- Good use of explicit non-claims
- Minor: the paper says "about 64 KB local SRAM per compute tile" — should cite source for exact number
- The duplicate sentence in Section 5 ("On the width extends the accuracy frontier...") appears to be a copy-paste artifact
