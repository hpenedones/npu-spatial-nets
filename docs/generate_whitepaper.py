#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate the TileFlow white paper as a PDF.

Combines explanatory text with the architecture diagrams into a
single document suitable for reading as an introduction to the project.

Usage::

    python docs/generate_whitepaper.py

Requires: weasyprint, matplotlib (both in the project .venv).
"""

from pathlib import Path
from weasyprint import HTML

DOCS_DIR = Path(__file__).parent

HTML_CONTENT = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {
    size: A4;
    margin: 2.5cm 2cm;
    @bottom-center { content: counter(page); font-size: 9pt; color: #888; }
  }
  body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #222;
    max-width: 100%;
  }
  h1 {
    font-size: 22pt;
    color: #1a1a2e;
    border-bottom: 3px solid #4A90D9;
    padding-bottom: 8px;
    margin-top: 0;
  }
  h2 {
    font-size: 15pt;
    color: #2c3e50;
    margin-top: 1.5em;
    border-bottom: 1px solid #ddd;
    padding-bottom: 4px;
  }
  h3 {
    font-size: 12pt;
    color: #34495e;
    margin-top: 1.2em;
  }
  .subtitle {
    font-size: 13pt;
    color: #555;
    margin-top: -0.5em;
    margin-bottom: 1.5em;
  }
  .authors {
    font-size: 10pt;
    color: #777;
    margin-bottom: 2em;
  }
  code {
    font-family: 'Courier New', monospace;
    font-size: 9.5pt;
    background: #f5f5f5;
    padding: 1px 4px;
    border-radius: 3px;
  }
  pre {
    background: #f8f8f8;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 12px;
    font-size: 9pt;
    line-height: 1.4;
    overflow-x: auto;
  }
  .figure {
    text-align: center;
    margin: 1.5em 0;
    page-break-inside: avoid;
  }
  .figure img {
    max-width: 100%;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
  }
  .figure .caption {
    font-size: 9.5pt;
    color: #555;
    margin-top: 0.5em;
    font-style: italic;
  }
  table {
    border-collapse: collapse;
    margin: 1em 0;
    width: 100%;
    font-size: 10pt;
  }
  th, td {
    border: 1px solid #ddd;
    padding: 6px 10px;
    text-align: left;
  }
  th {
    background: #4A90D9;
    color: white;
    font-weight: 600;
  }
  tr:nth-child(even) { background: #f9f9f9; }
  .highlight { background: #fff3cd; padding: 10px; border-radius: 4px;
               border-left: 4px solid #F5A623; margin: 1em 0; }
  .key-insight { background: #d4edda; padding: 10px; border-radius: 4px;
                 border-left: 4px solid #2ECC71; margin: 1em 0; }
  .warning { background: #f8d7da; padding: 10px; border-radius: 4px;
             border-left: 4px solid #E74C3C; margin: 1em 0; }
</style>
</head>
<body>

<h1>TileFlow: Spatial Neural Networks on AMD XDNA&nbsp;2 NPU</h1>
<p class="subtitle">
  Hardware-software co-design for close-to-metal neural network inference
</p>
<p class="authors">
  Built with <a href="https://github.com/amd/IRON">IRON/MLIR-AIE</a> toolchain
  &mdash; Target: AMD Ryzen AI 9 HX 370 (Strix Point)
</p>

<h2>1. Introduction</h2>

<p>
Modern neural processing units (NPUs) are spatial dataflow computers: instead of
a single CPU with caches and a register file, they expose a 2D array of small
compute tiles, each with its own SRAM and SIMD units, connected by a
programmable interconnect. This architecture is extremely efficient for
<em>on-chip</em> computation &mdash; but only if the software maps the
algorithm directly to the physical hardware.
</p>

<p>
<strong>TileFlow</strong> takes this literally. We design a neural network whose
architecture is dictated by the physical tile layout of the AMD XDNA&nbsp;2 NPU. The
network has learnable parameters (a shared weight matrix) and non-linearities
(ReLU), making it a valid machine-learning model &mdash; but its structure
(number of parallel paths, loop depth, buffer sizes) matches the hardware
exactly.
</p>

<div class="key-insight">
<strong>Key principle:</strong> We design the network to match the hardware,
not the other way around. Any architecture with learnable parameters and
non-linearities can learn &mdash; so we choose the one that maximizes
hardware utilization.
</div>

<h2>2. The Hardware</h2>

<p>
The AMD XDNA&nbsp;2 NPU in the Ryzen AI 9 HX 370 (codename Strix Point) is a
tiled spatial-dataflow processor with the following structure:
</p>

<div class="figure">
  <img src="xdna2_hardware.png" alt="XDNA 2 tile array">
  <div class="caption">
    Figure 1: Physical tile array of the AMD XDNA&nbsp;2 NPU. 32 compute tiles
    (rows 2&ndash;5) each contain ~64 KB SRAM and a bf16 MMUL unit. 8 memory tiles
    (row 1, 512 KB each) serve as on-chip L2 buffers and routing hubs.
    8 shim tiles (row 0) provide DMA access to host DDR memory.
  </div>
</div>

<table>
  <tr><th>Property</th><th>Value</th></tr>
  <tr><td>Compute tiles</td><td>32 (8 columns &times; 4 rows)</td></tr>
  <tr><td>Memory tiles</td><td>8 (512 KB each, 4 MB total)</td></tr>
  <tr><td>Per-tile SRAM</td><td>~64 KB data memory</td></tr>
  <tr><td>Per-tile compute</td><td>bf16 MMUL unit, VLIW+SIMD core</td></tr>
  <tr><td>Clock frequency</td><td>~1.5 GHz</td></tr>
  <tr><td>Peak throughput</td><td><strong>25 TFLOPS</strong> (bfloat16)</td></tr>
  <tr><td>Interconnect</td><td>ObjectFIFOs (tile-to-tile double-buffered streams)</td></tr>
  <tr><td>Power envelope</td><td>~6 W</td></tr>
</table>

<h3>2.1 Why spatial dataflow matters</h3>

<p>
On a CPU, a matrix multiply reads data from DRAM through multiple cache levels.
Each layer of a neural network bounces activations through L1&rarr;L2&rarr;L3&rarr;DRAM
and back. On the NPU, data stays in 64 KB tile SRAM between operations &mdash;
zero cache misses, zero bus contention. This is the source of the NPU's advantage
for deep, narrow computations.
</p>

<div class="highlight">
<strong>Lesson from Phase 1:</strong> A single large GEMM achieves only 2.49 TFLOPS
on the NPU (10% of peak) because it is <em>memory-bandwidth limited</em> &mdash;
data must stream from DDR. The NPU wins when data <strong>stays on-chip</strong>.
</div>

<h2>3. The Architecture: Recurrent MLP</h2>

<p>
We choose a recurrent MLP: a single weight matrix <code>W</code> (128&times;128,
bfloat16) is loaded once into each tile's SRAM and applied repeatedly in a tight
hardware loop:
</p>

<pre>
x = input                     # loaded from DDR, 16&times;128 bf16
for i in range(num_iters):
    y = ReLU(x @ W)           # matmul + activation
    x = ReLU(y @ W)           # ping-pong: result goes back to x
output = x                    # drained to DDR
</pre>

<p>
This is mapped to 24 compute tiles (3 rows &times; 8 columns), each running the
same loop independently on different input samples:
</p>

<div class="figure">
  <img src="recurrent_mlp.png" alt="Recurrent MLP mapped to NPU">
  <div class="caption">
    Figure 2: Recurrent MLP mapped to the XDNA&nbsp;2 tile array. 24 tiles across
    3 compute rows run the same hardware loop in parallel. Row 5 is unused due to
    MemTile routing constraints (~6 northward master ports). The detail box shows
    the per-tile computation: acquire buffers, loop, copy result.
  </div>
</div>

<h3>3.1 Why this architecture</h3>

<ul>
  <li><strong>Maximizes on-chip time:</strong> Weight is loaded once (32 KB) and
      reused for thousands of matmul operations. DDR I/O happens only at
      start and end.</li>
  <li><strong>Amortizes overhead:</strong> Each NPU invocation has ~120 &mu;s of
      driver/DMA overhead. With depth=2000, compute time (~1.3 ms) dominates
      overhead by 10&times;.</li>
  <li><strong>Fits SRAM budget:</strong> W (32 KB) + 2 activation buffers (4 KB
      each) + stack (1 KB) = 41 KB, well within the 64 KB tile limit.</li>
  <li><strong>Linear tile scaling:</strong> Each tile is independent &mdash;
      doubling tiles doubles throughput with no communication overhead.</li>
</ul>

<h3>3.2 Multi-row data routing</h3>

<p>
When using more than 8 tiles (i.e., more than one compute row), data must pass
through the MemTiles (row 1) which act as routing hubs:
</p>

<ul>
  <li><strong>Weights</strong> are <em>broadcast</em> via <code>forward()</code>:
      one DDR&rarr;MemTile transfer, then the MemTile fans out to all compute
      rows in the column.</li>
  <li><strong>Inputs</strong> are <em>split</em> via <code>split()</code>: the
      host buffer is partitioned so each row gets its own batch slice.</li>
  <li><strong>Outputs</strong> are <em>joined</em> via <code>join()</code>: per-row
      results are aggregated back through the MemTile to DDR.</li>
</ul>

<div class="warning">
<strong>Routing limit:</strong> Each MemTile has approximately 6 northward
master ports. Our design requires 3 data streams per row (weight + input + output).
At 3 compute rows = 9 streams, this fits; at 4 rows = 12 streams, the MLIR-AIE
router fails. This caps us at <strong>24 tiles</strong> (3 rows &times; 8 columns).
</div>

<h3>3.3 Critical implementation constraints</h3>

<p>
Several non-obvious hardware constraints shaped the design:
</p>

<ul>
  <li><strong>No FIFO ops inside loops:</strong> Placing <code>acquire()</code> /
      <code>release()</code> inside <code>range_()</code> (which compiles to
      <code>scf.for</code>) causes DMA deadlock. All FIFO operations must happen
      <em>outside</em> the loop.</li>
  <li><strong>DMA BD 10-bit size limit:</strong> Shim DMA buffer-descriptor sizes
      are 10-bit (max 1024). For B=16, H=128, the product B&times;H=2048 exceeds
      this, so tensor access patterns must factor dimensions as [B, H] = [16, 128]
      instead of [B&times;H] = [2048].</li>
  <li><strong>Accumulating matmul:</strong> IRON's <code>mm.cc</code> kernel
      computes C += A&times;B (not C = A&times;B). We must explicitly zero the
      output buffer before each matmul, which wastes ~12% of cycle time.</li>
</ul>

<h2>4. Results</h2>

<div class="figure">
  <img src="performance.png" alt="Performance scaling">
  <div class="caption">
    Figure 3: NPU throughput scaling (left) and speedup over CPU (right).
    Throughput scales near-linearly with tile count at ~360 GFLOPS/tile.
    At 24 tiles, we achieve 8.95 TFLOPS and 20&times; CPU speedup.
  </div>
</div>

<table>
  <tr><th>Tiles</th><th>Depth</th><th>NPU Latency</th><th>NPU TFLOPS</th>
      <th>CPU GFLOPS</th><th>Speedup</th></tr>
  <tr><td>8 (1 row)</td><td>1,000</td><td>1.45 ms</td><td><strong>2.89</strong></td>
      <td>237</td><td><strong>12.2&times;</strong></td></tr>
  <tr><td>16 (2 rows)</td><td>1,000</td><td>1.46 ms</td><td><strong>5.74</strong></td>
      <td>354</td><td><strong>16.2&times;</strong></td></tr>
  <tr><td>24 (3 rows)</td><td>1,000</td><td>1.46 ms</td><td><strong>8.63</strong></td>
      <td>429</td><td><strong>20.1&times;</strong></td></tr>
  <tr><td>24 (3 rows)</td><td>10,000</td><td>14.05 ms</td><td><strong>8.95</strong></td>
      <td>439</td><td><strong>20.4&times;</strong></td></tr>
</table>

<h3>4.1 Scaling analysis</h3>

<p>
Per-tile throughput is remarkably consistent at ~360 GFLOPS regardless of tile
count, confirming that the tiles operate independently with no contention:
</p>

<pre>
 8 tiles &times; 360 GFLOPS/tile =  2.9 TFLOPS  &check;
16 tiles &times; 360 GFLOPS/tile =  5.7 TFLOPS  &check;  (near-linear)
24 tiles &times; 360 GFLOPS/tile =  8.6 TFLOPS  &check;  (near-linear)
</pre>

<h3>4.2 Gap to theoretical peak</h3>

<p>
We achieve 8.95 of 25 TFLOPS (35.8%). The remaining gap is well-understood:
</p>

<table>
  <tr><th>Factor</th><th>Impact</th><th>Potential fix</th></tr>
  <tr><td>Per-tile utilization</td><td>360/768 = 47%</td>
      <td>Fused C=A&times;B kernel</td></tr>
  <tr><td>zero_bf16 overhead</td><td>~12% of step time</td>
      <td>Fused kernel eliminates this</td></tr>
  <tr><td>Array utilization</td><td>24/32 = 75%</td>
      <td>4-row routing (needs HW/compiler support)</td></tr>
  <tr><td>Combined theoretical max</td><td>~18 TFLOPS</td>
      <td>With fused kernel + 24 tiles</td></tr>
</table>

<h2>5. The Toolchain</h2>

<table>
  <tr><th>Component</th><th>Role</th></tr>
  <tr><td><a href="https://github.com/amd/IRON">IRON</a></td>
      <td>Python API for tile layout, ObjectFIFOs, and dataflow</td></tr>
  <tr><td><a href="https://github.com/Xilinx/mlir-aie">MLIR-AIE</a></td>
      <td>MLIR dialect &rarr; hardware bitstream compilation</td></tr>
  <tr><td><a href="https://github.com/Xilinx/llvm-aie">Peano/LLVM-AIE</a></td>
      <td>C++ compiler for per-tile kernels</td></tr>
  <tr><td><a href="https://github.com/amd/xdna-driver">XRT</a></td>
      <td>Runtime for loading and executing on the NPU</td></tr>
</table>

<h3>5.1 Compilation pipeline</h3>

<pre>
design.py  &xrarr;  MLIR  &xrarr;  aiecc  &xrarr;  .xclbin (bitstream)
                          &xrarr;  .bin   (instruction sequence)

mm.cc          &xrarr;  mlp_mm.o    &xrarr;  mlp_kernels.a
mlp_kernels.cc &xrarr;  mlp_relu.o  &nearr;
</pre>

<h2>6. Code Structure</h2>

<p>
The project is intentionally minimal &mdash; four Python files and one C++ file:
</p>

<table>
  <tr><th>File</th><th>Lines</th><th>Purpose</th></tr>
  <tr><td><code>spatial_mlp/__init__.py</code></td><td>~55</td>
      <td>Tiling utilities (<code>to_tiled</code>, <code>from_tiled</code>)</td></tr>
  <tr><td><code>spatial_mlp/design.py</code></td><td>~300</td>
      <td>IRON design: tile topology, FIFOs, workers, DMA</td></tr>
  <tr><td><code>spatial_mlp/op.py</code></td><td>~140</td>
      <td>IRON operator: compilation artifacts, runtime buffers</td></tr>
  <tr><td><code>spatial_mlp/test.py</code></td><td>~240</td>
      <td>Benchmark: NPU vs CPU execution and reporting</td></tr>
  <tr><td><code>aie_kernels/mlp_kernels.cc</code></td><td>~55</td>
      <td>Custom AIE2P kernels: ReLU, copy (bf16, SIMD)</td></tr>
</table>

<p>
Each module has a single, well-defined responsibility. The design module is
decomposed into small functions that each handle one aspect of the hardware
mapping: validation, kernel definition, FIFO topology, worker bodies, tensor
access patterns, and DMA sequences.
</p>

<h2>7. Future Work</h2>

<ul>
  <li><strong>Fused matmul kernel:</strong> A C=A&times;B kernel (instead of
      C+=A&times;B + separate zero) would eliminate ~12% overhead and push
      per-tile throughput from 360 to ~700 GFLOPS.</li>
  <li><strong>INT8 mode:</strong> The NPU's peak is 50 TOPS for int8.
      With H=256 (fitting larger weights), this could double throughput.</li>
  <li><strong>Training:</strong> Research NPU backpropagation
      (see <a href="https://arxiv.org/html/2504.03083v1">arXiv:2504.03083</a>).</li>
  <li><strong>Real ML task:</strong> Apply the architecture to a concrete
      sequence modeling or time-series task where deep recurrence is natural.</li>
</ul>

<h2>References</h2>

<ol>
  <li>AMD IRON repository: <a href="https://github.com/amd/IRON">github.com/amd/IRON</a></li>
  <li>MLIR-AIE programming guide:
      <a href="https://github.com/Xilinx/mlir-aie/tree/main/programming_guide">
      github.com/Xilinx/mlir-aie</a></li>
  <li>IRON tutorial (IPDPS 2025):
      <a href="https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/iron-for-ryzen-ai-tutorial-ipdps-2025.pdf">
      AMD Technical Paper</a></li>
  <li>NPU training: <a href="https://arxiv.org/html/2504.03083v1">arXiv:2504.03083</a></li>
  <li>Linux kernel NPU docs:
      <a href="https://docs.kernel.org/accel/amdxdna/amdnpu.html">kernel.org</a></li>
</ol>

</body>
</html>
"""


def generate():
    """Render the white paper HTML to PDF."""
    output_path = DOCS_DIR / "tileflow_whitepaper.pdf"
    html = HTML(string=HTML_CONTENT, base_url=str(DOCS_DIR))
    html.write_pdf(output_path)
    print(f"  ✓ {output_path}")
    return output_path


if __name__ == "__main__":
    print("Generating TileFlow white paper...")
    generate()
    print("Done.")
