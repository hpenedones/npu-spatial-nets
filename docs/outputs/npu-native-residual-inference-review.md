# Peer Review: NPU-Native Neural Networks on AMD XDNA 2

**Artifact:** Whitepaper + codebase — "NPU-Native Neural Networks on AMD XDNA 2: High-throughput residual inference for HIGGS"
**Author:** Hugo Penedones
**Reviewer:** Feynman (automated review)
**Date:** 2026-04-10

---

## Summary

This paper presents a co-designed residual MLP whose hidden body maps one residual block per compute tile on the AMD XDNA 2 NPU (32-tile array). The model is evaluated on the public HIGGS collider-event benchmark, yielding two retained configurations: a speed-optimized H=32, L=8 variant at ~4.18M samples/s wall-clock throughput, and an accuracy-tuned H=64, L=32 variant at 77.98% test accuracy / 0.865 ROC AUC. The paper includes a complete open-source implementation spanning training, hyperparameter search, AIE kernel code, and a spatial-streaming deployment path via IRON/MLIR-AIE.

**Overall assessment:** This is a well-scoped, honest systems paper that makes a narrow but defensible contribution. The code-paper alignment is strong, the non-claims are genuine guardrails (not hedging), and the co-design argument is novel in combining model shape, hardware topology, and end-to-end measurements on real silicon. The main weaknesses are missing baselines (no CPU/GPU throughput comparison) and the absence of latency and variance reporting.

**Recommendation:** Accept with minor revisions.

---

## Detailed Review

### 1. Novelty — ADEQUATE

**Strengths:**
- The one-block-per-tile spatial mapping of a residual MLP to XDNA 2 is, to my knowledge, unpublished. Prior spatial dataflow work (DaDianNao, Eyeriss, TPU) establishes the general paradigm, but none targets this specific hardware with this specific model family.
- The hls4ml line of work targets FPGAs for collider inference; this paper targets a consumer NPU — a genuinely different hardware regime with different constraints (tile-local SRAM vs. FPGA fabric).
- AMD's own Triton-XDNA project takes a compiler-driven approach; this paper takes a hand-mapped, model-shape-aware approach. The two are complementary.
- The finding that the fused NPU head is *slower* than CPU head is a valuable honest systems result that wouldn't emerge from a purely algorithmic study.

**Weaknesses:**
- The paper correctly notes that IRON, MLIR-AIE, and XRT are pre-existing. The novel content is the mapping + measurements, which is narrower than a new architecture or compiler. This is acceptable for the claimed scope but limits the contribution's breadth.

### 2. Empirical Rigor — ADEQUATE WITH GAPS

#### 2a. Accuracy results
The reported 0.865 ROC AUC (H=64, L=32) is contextually reasonable:
- **Above** typical XGBoost baselines on HIGGS (~0.84–0.85 AUC)
- **Below** Baldi et al.'s original 5-layer DNN result (0.885 AUC) by ~2 points

The paper's self-assessment — "not a claim of state-of-the-art HIGGS accuracy" but "strong enough that the main contribution is no longer accuracy-indifferent" — is honest and well-calibrated.

⚠️ **Missing: variance reporting.** All accuracy numbers appear to be single-run results. Even one repeated seed would strengthen confidence. The Optuna sweep partially mitigates this (validation selection from multiple trials), but the final test-set evaluation has no error bars.

#### 2b. Throughput results
The benchmarking methodology is sound:
- 4 warmup calls before timing ✅
- 50M processed samples for stable measurement ✅
- Clear separation of wall time vs. kernel time ✅
- `time.perf_counter()` for high-resolution timing ✅

The 4.18M samples/s figure is plausible: 50M samples / 4.18M/s ≈ 12 seconds, which is a reasonable tight-loop benchmark duration.

⚠️ **"Repeated-sample" caveat.** The benchmark reuses the same input batch across all iterations. The paper states this clearly in Table 5 ("repeated-sample throughput"), which is honest. However, real deployment would need fresh data each call. The benchmark thus measures the *sustained compute throughput ceiling*, not end-to-end application throughput. This distinction deserves a sentence in the discussion.

⚠️ **Wall-vs-kernel gap.** The 6.3× gap between kernel throughput (26.29M/s) and wall throughput (4.18M/s) is large. The paper attributes this to host overhead (CPU embed, head, buffer sync, Python loop). This is plausible but unexplored — a breakdown of where the 6.3× goes (CPU embed time, sync time, Python overhead) would significantly strengthen the systems analysis.

#### 2c. SRAM accounting
I verified the paper's tile-SRAM formulas:
- Internal tile: `1024 + 2H² + 32H` bytes — **structurally correct**
- Ingress tile: `1024 + 2H² + 64H` bytes — **structurally correct**
- H=160 ingress: computed 61.0 KB; paper says "about 62.5 KB" — **minor discrepancy (~1.5 KB)**. Likely from additional overhead not in the simplified formula. The qualitative conclusion (fits at H=160, fails at H=168) is correct in both cases.

### 3. Baselines — WEAK POINT

This is the review's most substantive concern.

⚠️ **No CPU/GPU throughput baseline.** The paper reports 4.18M samples/s on the NPU but never measures the same H=32, L=8 model running on the host CPU or on the integrated GPU. For a systems paper arguing that the NPU delivers "unusually high throughput," this comparison is essential. A simple PyTorch benchmark of the same model on CPU/GPU would take minutes to implement and could either validate the NPU advantage or reveal that the host is competitive for this tiny model.

The HIGGS accuracy baselines are handled well — the paper cites XGBoost, LightGBM, CatBoost, TabNet, NODE, and the original Baldi et al. results, and positions itself honestly. But the accuracy baselines are not the main contribution; the throughput story is, and it lacks the most obvious comparison point.

**Recommendation:** Add a CPU and GPU throughput measurement for the same model architectures. Even a single-line table row would suffice.

### 4. Reproducibility — STRONG

This is a standout dimension of the paper:
- Complete training code with documented hyperparameters ✅
- Optuna/MLflow sweep infrastructure for the tuning run ✅
- Public dataset with documented splits (`jxie/higgs`, 10.5M/500k) ✅
- AIE kernel source code with clear compile flags ✅
- Streaming design that generates MLIR from Python ✅
- Correctness tests with reference implementation ✅
- Figure generation script that parses the paper's own tables ✅

The only barrier to full reproduction is the hardware requirement (AMD XDNA 2), which is inherent and not a paper flaw. The training and accuracy results are reproducible on any GPU.

### 5. Scope Honesty — EXCELLENT

The paper's explicit non-claims are genuine and well-placed:
- "Not a production trigger deployment" ✅
- "Not a state-of-the-art HIGGS accuracy paper" ✅
- "Not a fully on-array inference pipeline" ✅
- "Not an energy-efficiency study" ✅

This is refreshingly honest. The abstract states limits clearly, the discussion reinforces them, and the paper doesn't over-reach. The CMS trigger motivation is introduced carefully ("explain why multi-megasample-per-second measurements are interesting here, but they are not used to imply deployment equivalence").

### 6. Code-Paper Alignment — STRONG

I performed a line-by-line audit:

| Claim | Code location | Verdict |
|---|---|---|
| `h = relu(h @ W) + h` | `model.py:ResidualLinear.forward()` | ✅ Exact match |
| One weight per tile | `streaming_design.py:Buffer(initial_value=...)` | ✅ |
| Snake routing | `design.py:snake_tile_order()` | ✅ |
| 8×8 MMUL intrinsic | `matmul_relu_skip.cc` | ✅ |
| Compile-time weights | `streaming_op.py:_store_embedded_weights()` | ✅ |
| CPU head path | `streaming_infer.py:process_hidden_chunk()` | ✅ numpy matmul |
| 50M sample benchmark | `streaming_infer.py:benchmark()` | ✅ |
| 28-dim native input | `data_utils.py:HIGGS_INPUT_DIM=28` | ✅ |
| 100k val holdout | `data_utils.py:DEFAULT_VAL_SIZE=100_000` | ✅ |

No discrepancies found between paper claims and code implementation.

### 7. Writing Quality — GOOD

- Structure is clear and logical
- Notation is consistent (H, L, B defined upfront)
- The hardware background appendix is well-calibrated for a mixed ML/systems audience
- The HIGGS physics appendix adds useful context without over-explaining

**Bugs found:**
- ~~**Duplicate sentence in Section 5:** "On the width extends the accuracy frontier rather than the throughput frontier." appears twice in consecutive paragraphs. This is a copy-paste error.~~ **Fixed.**
- ~~The `model.py` docstring still says "image classification" in places (`ResMLP` class docstring: "Residual MLP for image classification"), suggesting legacy from an earlier MNIST/CIFAR iteration mentioned in the README's experimental branch.~~ **Fixed.**

### 8. Figures and Tables

- **Table 4 ↔ Table 5 ↔ Figure 3:** The frontier plot generation script (`generate_accuracy_throughput_frontier.py`) parses the LaTeX source tables directly, ensuring the figure is always consistent with the tables. This is an unusually strong consistency guarantee. ✅
- **Figure 1 (hardware layout):** Informative but not independently verifiable.
- **Figure 2 (model-hardware mapping):** Clear and matches the code.

---

## Issues Summary

| # | Severity | Issue |
|---|---|---|
| 1 | **Major** | No CPU/GPU throughput baseline for the same model. The NPU throughput claim ("unusually high") lacks the most obvious comparison point. |
| 2 | Minor | No variance/confidence intervals on accuracy results. Single-run numbers only. |
| 3 | Minor | No per-sample latency measurement. Only throughput is reported, but the CMS trigger motivation emphasizes latency budgets. |
| 4 | Minor | Wall-vs-kernel throughput gap (6.3×) is reported but not broken down. A profiling breakdown would strengthen the systems analysis. |
| 5 | Minor | SRAM formula gives 61.0 KB at H=160; paper says "about 62.5 KB." Small discrepancy suggests the formula is approximate. |
| 6 | ~~Nit~~ | ~~Duplicate sentence in Section 5 (copy-paste artifact).~~ **Fixed.** |
| 7 | ~~Nit~~ | ~~Legacy "image classification" docstring in `model.py`.~~ **Fixed.** |
| 8 | ~~Nit~~ | ~~"About 64 KB local SRAM per compute tile" — should cite the exact source for this number.~~ **Fixed.** |

---

## Verdict

**Accept with minor revisions.** The paper makes a narrow, well-defended contribution: a co-designed residual MLP that maps cleanly to a consumer NPU and delivers credible results on a real benchmark. The code quality and reproducibility are above average. The scope honesty is exemplary. The main gap — no CPU/GPU throughput comparison — is straightforward to fill and would meaningfully strengthen the central systems claim. The accuracy positioning is honest and well-contextualized against the relevant literature.

---

## Sources

All external sources consulted during this review:

1. **Baldi, Sadowski, Whiteson (2014)** — "Searching for Exotic Particles in High-Energy Physics with Deep Learning" — https://arxiv.org/abs/1402.4735
2. **UCI HIGGS Dataset** — https://archive.ics.uci.edu/dataset/280/higgs
3. **jxie/higgs HuggingFace mirror** — https://huggingface.co/datasets/jxie/higgs
4. **AMD XDNA Architecture overview** — https://amd.com/en/technologies/xdna.html
5. **AMD NPU Linux kernel documentation** — https://docs.kernel.org/next/accel/amdxdna/amdnpu.html
6. **AMD xdna-driver documentation** — https://github.com/amd/xdna-driver/blob/main/src/driver/doc/amdnpu.rst
7. **AMD IRON repository** — https://github.com/amd/IRON
8. **MLIR-AIE programming guide** — https://github.com/Xilinx/mlir-aie/tree/main/programming_guide
9. **AMD Triton-XDNA** — https://github.com/amd/Triton-XDNA
10. **Gorishniy et al. (2021)** — "Revisiting Deep Learning Models for Tabular Data" — https://arxiv.org/abs/2106.11959
11. **Grinsztajn et al. (2022)** — "Why do tree-based models still outperform deep learning on tabular data?" — https://arxiv.org/abs/2207.08815
12. **Chen & Guestrin (2016)** — "XGBoost: A Scalable Tree Boosting System" — https://arxiv.org/abs/1603.02754
13. **Duarte et al. (2018)** — "Fast inference of deep neural networks in FPGAs for particle physics" — https://doi.org/10.1088/1748-0221/13/07/P07027
14. **Jouppi et al. (2017)** — "In-Datacenter Performance Analysis of a Tensor Processing Unit" — https://arxiv.org/abs/1704.04760
