# TODO

Mock ML4Jets 2026 reviewer feedback suggested the following improvements before
submission. The goal of this file is to keep those follow-ups visible and easy
to prioritize.

## Paper and experiment follow-ups

- [ ] Expand the latency analysis
  - Measure single-sample latency for the conveyor-belt NPU path.
  - Measure tail latency (for example, p99) instead of reporting throughput
    alone.
  - Add either a latency distribution plot or a compact latency table to the
    paper.

- [ ] Add a standard HEP baseline comparison
  - Compare accuracy and throughput against a strong tabular baseline such as
    XGBoost or LightGBM on CPU.
  - Optionally include a small MLP baseline on a mobile GPU.
  - If the numbers support it, frame the comparison in terms of
    efficiency-per-dollar or efficiency-per-watt.

- [ ] Discuss scaling to wider feature sets
  - Explain how the conveyor-belt design changes as the input becomes wider.
  - State more explicitly when the 64 KB per-tile SRAM becomes the hard
    bottleneck.
  - Briefly connect this to more realistic jet-tagging or richer HEP feature
    pipelines.

- [ ] Quantify the current CPU embed overhead
  - Measure how much of the current wall-clock time is spent in the host-side
    `28 -> 32` embed.
  - Estimate how much headroom could be recovered by moving that step on-array.
  - Use that number to sharpen the motivation for the next systems step.

## Suggested order

1. Quantify the CPU embed overhead.
2. Expand the latency analysis.
3. Add the HEP baseline comparison.
4. Add the wider-input / SRAM-scaling discussion to the paper.
