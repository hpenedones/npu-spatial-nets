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
  a.gref {
    color: #2c3e50;
    text-decoration: none;
    border-bottom: 1px dotted #4A90D9;
  }
  a.gref:hover {
    color: #4A90D9;
    border-bottom: 1px solid #4A90D9;
  }
  .vocab-term {
    font-weight: 600;
    color: #2c3e50;
  }
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
architecture is dictated by the physical tile layout of the AMD XDNA&nbsp;2
<a href="#g-npu" class="gref">NPU</a>. The network has learnable parameters
and non-linearities (<a href="#g-relu" class="gref">ReLU</a>), but its structure
(number of layers, block grouping, buffer sizes) matches the hardware exactly.
</p>

<p>
Our main result is a <strong>block-recurrent character language model</strong>
that generates Shakespeare-like text entirely on the NPU. The model has 32
layers grouped into 8 blocks of 4. Each block maps to a 4-stage pipeline
on the NPU (one layer per compute row, 8 columns in parallel). Between
blocks, the CPU applies normalisation, character-embedding injection, and
residual connections &mdash; the same operations the model uses during training.
The result: <strong>89,600 characters/second</strong> on 32 NPU tiles, with
the trained weights producing <em>identical</em> computation on hardware
&mdash; no approximations, no graph compilation, no operator fallback.
</p>

<div class="key-insight">
<strong>Key principle:</strong> We design the network to match the hardware,
not the other way around. Any architecture with learnable parameters and
non-linearities can learn &mdash; so we choose the one that maximizes
hardware utilization.
</div>

<h2>2. Background</h2>

<p>
This section provides the hardware and systems context that most ML engineers
never need to think about &mdash; until they want to understand <em>why</em>
certain hardware is fast (or slow) for their models. If you&rsquo;ve trained
models with PyTorch or JAX and think of hardware as &ldquo;a GPU with some
VRAM,&rdquo; this section fills in the gap between that abstraction and the
physical reality of a spatial processor like the XDNA&nbsp;2 NPU.
</p>

<h3>2.1 Glossary of acronyms</h3>

<p>
The table below defines every acronym used in this paper. Each entry has an
<code>id</code> so that occurrences in the text link back here. We group them
by domain so you can revisit this section as a reference while reading.
</p>

<table>
  <tr><th>Acronym</th><th>Stands for</th><th>What it means</th></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Hardware &amp; memory</td></tr>
  <tr id="g-npu"><td>NPU</td><td>Neural Processing Unit</td>
      <td>A dedicated accelerator for neural-network inference (and
          sometimes training), built into a laptop or phone chip.</td></tr>
  <tr id="g-apu"><td>APU</td><td>Accelerated Processing Unit</td>
      <td>AMD&rsquo;s name for a single die that integrates CPU + GPU + NPU.</td></tr>
  <tr id="g-xdna"><td>XDNA</td><td>(AMD brand name)</td>
      <td>AMD&rsquo;s NPU architecture family. XDNA&nbsp;2 is the second
          generation, found in &ldquo;Strix Point&rdquo; Ryzen AI chips.</td></tr>
  <tr id="g-aie"><td>AIE</td><td>AI Engine</td>
      <td>The individual tile processor IP inside the XDNA NPU, originally
          designed by Xilinx (acquired by AMD).</td></tr>
  <tr id="g-sram"><td>SRAM</td><td>Static Random-Access Memory</td>
      <td>Fast, on-chip memory (~1&ndash;2 ns access). Each compute tile has
          ~64 KB of SRAM. Expensive per bit, but extremely fast because it
          sits right next to the compute logic.</td></tr>
  <tr id="g-dram"><td>DRAM</td><td>Dynamic Random-Access Memory</td>
      <td>The main system memory (8&ndash;64 GB). Much slower than SRAM
          (~50&ndash;100 ns access) but vastly cheaper per bit.</td></tr>
  <tr id="g-ddr"><td>DDR</td><td>Double Data Rate</td>
      <td>The interface standard for DRAM modules. &ldquo;DDR memory&rdquo; is
          the system RAM your laptop uses. In this paper, &ldquo;DDR&rdquo;
          means &ldquo;host-side main memory.&rdquo;</td></tr>
  <tr id="g-dma"><td>DMA</td><td>Direct Memory Access</td>
      <td>A hardware mechanism that copies data between memory regions
          <em>without using the CPU</em>. The NPU&rsquo;s shim tiles contain
          DMA engines that move data between DDR and tile SRAM.</td></tr>
  <tr id="g-pcie"><td>PCIe</td><td>Peripheral Component Interconnect Express</td>
      <td>The high-speed bus connecting the NPU to the rest of the system.</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Compute concepts</td></tr>
  <tr id="g-simd"><td>SIMD</td><td>Single Instruction, Multiple Data</td>
      <td>A processor executes one instruction on a <em>vector</em> of values
          simultaneously. For example, multiplying 32 numbers in one clock
          cycle instead of one at a time. This is how GPUs and NPUs achieve
          massive throughput.</td></tr>
  <tr id="g-vliw"><td>VLIW</td><td>Very Long Instruction Word</td>
      <td>A processor design where each instruction encodes <em>multiple
          operations</em> to execute in parallel (e.g., a multiply, an add,
          and a load all in one cycle). Each AIE tile uses a VLIW core.</td></tr>
  <tr id="g-mmul"><td>MMUL</td><td>Matrix Multiply (unit)</td>
      <td>A hardware block dedicated to multiplying small matrices (e.g.,
          8&times;8 blocks of bfloat16). This is the workhorse of each AIE
          tile.</td></tr>
  <tr id="g-fifo"><td>FIFO</td><td>First In, First Out</td>
      <td>A queue where data is read in the same order it was written. The
          NPU uses hardware FIFOs (&ldquo;ObjectFIFOs&rdquo;) to stream data
          between tiles, like Unix pipes between processes.</td></tr>
  <tr id="g-gemm"><td>GEMM</td><td>General Matrix Multiply</td>
      <td>The standard linear algebra operation C = A &times; B + C. Neural
          network layers are essentially sequences of GEMMs with
          non-linearities in between.</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Numeric formats</td></tr>
  <tr id="g-bf16"><td>bf16 / bfloat16</td><td>Brain Floating Point, 16-bit</td>
      <td>A 16-bit floating-point format with the same exponent range as
          float32 (8 bits) but less precision (7-bit mantissa vs 23-bit).
          Invented at Google Brain for ML training where range matters more
          than precision.</td></tr>
  <tr id="g-bfp16"><td>BFP16</td><td>Block Floating Point, 16-bit</td>
      <td>An emulation mode on AIE2P that groups bf16 values into blocks
          sharing a common exponent. Enables efficient SIMD matmul in the
          MMUL unit with tile factor r = s = t = 8.</td></tr>
  <tr id="g-int8"><td>INT8</td><td>8-bit Integer</td>
      <td>8-bit integer arithmetic, used for quantized inference. The
          NPU&rsquo;s peak in INT8 mode is 50 TOPS (double the bf16
          rate).</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Performance metrics</td></tr>
  <tr id="g-flops"><td>FLOPS</td><td>Floating-point Operations Per Second</td>
      <td>The standard measure of compute throughput. One multiply-add on
          two numbers counts as 2 FLOPs.</td></tr>
  <tr id="g-gflops"><td>GFLOPS</td><td>Giga&nbsp;FLOPS (10<sup>9</sup>)</td>
      <td>Billions of floating-point operations per second.</td></tr>
  <tr id="g-tflops"><td>TFLOPS</td><td>Tera&nbsp;FLOPS (10<sup>12</sup>)</td>
      <td>Trillions of floating-point operations per second. The NPU&rsquo;s
          peak is 25 TFLOPS in bfloat16.</td></tr>
  <tr id="g-tops"><td>TOPS</td><td>Tera Operations Per Second</td>
      <td>Like TFLOPS but for integer operations. Used for INT8 peak specs
          (50 TOPS for this NPU).</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Machine learning</td></tr>
  <tr id="g-mlp"><td>MLP</td><td>Multi-Layer Perceptron</td>
      <td>A neural network made of fully-connected (dense) layers. Each
          layer computes y = activation(x &times; W + b).</td></tr>
  <tr id="g-relu"><td>ReLU</td><td>Rectified Linear Unit</td>
      <td>The activation function max(x, 0). Simple, cheap to compute, and
          widely used.</td></tr>
  <tr id="g-rmsnorm"><td>RMSNorm</td><td>Root Mean Square Normalisation</td>
      <td>A layer-normalisation variant that divides each vector by its RMS
          value: x / sqrt(mean(x&sup2;) + &epsilon;). Cheaper than LayerNorm
          because it skips mean subtraction.</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Toolchain &amp; software</td></tr>
  <tr id="g-iron"><td>IRON</td><td>(name, not an acronym)</td>
      <td>AMD&rsquo;s Python API for programming AIE tiles at a high level:
          defining kernels, ObjectFIFOs, and tile placements.</td></tr>
  <tr id="g-mlir"><td>MLIR</td><td>Multi-Level Intermediate Representation</td>
      <td>A compiler framework (from the LLVM project) that represents
          programs at multiple abstraction levels. MLIR-AIE is the dialect
          that targets AIE hardware.</td></tr>
  <tr id="g-xrt"><td>XRT</td><td>Xilinx Runtime</td>
      <td>The userspace library and kernel driver that loads bitstreams onto
          the NPU and manages execution.</td></tr>
  <tr id="g-xclbin"><td>XCLBIN</td><td>Xilinx Container for Linux Binary</td>
      <td>The compiled binary file that contains the NPU bitstream (tile
          configuration, routing, kernel code).</td></tr>
  <tr id="g-llvm"><td>LLVM</td><td>Low Level Virtual Machine</td>
      <td>A widely-used compiler infrastructure. Peano/LLVM-AIE is a fork
          that compiles C++ to AIE tile machine code.</td></tr>
  <tr id="g-bd"><td>BD</td><td>Buffer Descriptor</td>
      <td>A hardware structure in the DMA engine that describes one data
          transfer: source address, size, stride pattern. The 10-bit size
          field (max 1024) is a constraint discussed in Section 3.6.</td></tr>
</table>

<h3>2.2 Key vocabulary (non-acronyms)</h3>

<p>
Several important terms in this paper are not acronyms but may be unfamiliar
to readers who have worked above the hardware abstraction layer:
</p>

<table>
  <tr><th style="width:120px;">Term</th><th>What it means</th></tr>
  <tr id="v-shim"><td class="vocab-term">Shim tile</td>
      <td>The row of interface tiles at the bottom of the NPU array (row 0).
          &ldquo;Shim&rdquo; means a thin adapter layer &mdash; these tiles
          <em>bridge</em> two different worlds: the system&rsquo;s DDR memory
          (accessed via PCIe/AXI bus) and the NPU&rsquo;s internal tile
          interconnect. Each shim tile contains DMA engines that translate
          between host memory addresses and on-chip tile coordinates.
          They do no computation; they are pure data movers.</td></tr>
  <tr id="v-tile"><td class="vocab-term">Tile</td>
      <td>A self-contained processing unit with its own local memory (SRAM),
          instruction memory, and compute logic. Unlike CPU cores that share
          caches, each tile is an <em>independent computer</em> that communicates
          with neighbours through explicit data channels (FIFOs), not shared
          memory.</td></tr>
  <tr id="v-bitstream"><td class="vocab-term">Bitstream</td>
      <td>The compiled binary that programs the NPU&rsquo;s hardware
          configuration: which tiles run which code, how the interconnect routes
          data, and what DMA transfers to perform. Analogous to a GPU shader
          binary, but configuring a full spatial processor rather than a single
          shader core. Packaged in an XCLBIN file.</td></tr>
  <tr id="v-kernel"><td class="vocab-term">Kernel (NPU)</td>
      <td>A C++ function compiled to run on a single tile. Not to be confused
          with an OS kernel or a CUDA kernel (which runs on thousands of GPU
          threads). An NPU kernel runs on exactly <em>one</em> tile&rsquo;s
          VLIW+SIMD core and has direct access to that tile&rsquo;s SRAM.</td></tr>
  <tr id="v-pipeline"><td class="vocab-term">Pipeline</td>
      <td>A technique where multiple stages of a task overlap in time, like
          an assembly line. While stage N processes item <em>i</em>, stage N+1
          processes item <em>i&minus;1</em>. The chess compiler pipelines the
          inner loops of our matmul kernel so that memory loads, multiplications,
          and stores overlap across iterations.</td></tr>
  <tr id="v-register"><td class="vocab-term">Register</td>
      <td>The smallest, fastest storage inside a processor core &mdash; a few
          hundred bytes at most, accessible in a single clock cycle. The MMUL
          unit accumulates partial matrix products in registers before writing
          back to SRAM. Our fused kernel zero-initialises these accumulators
          in registers, avoiding a separate SRAM clear pass.</td></tr>
  <tr id="v-memwall"><td class="vocab-term">Memory wall</td>
      <td>The growing gap between processor speed and memory speed. CPUs can
          execute arithmetic ~100&times; faster than DRAM can supply data. This
          is <em>the</em> fundamental reason NPUs (with data-local SRAM) can be
          dramatically faster than CPUs for the right workloads.</td></tr>
  <tr id="v-memtile"><td class="vocab-term">MemTile</td>
      <td>A memory-only tile (no compute core) in the NPU array, sitting between
          the shim row and the compute rows. Contains 512&nbsp;KB of SRAM and
          powerful DMA engines that can split, forward, or join data streams,
          routing data from the shim to multiple compute rows.</td></tr>
  <tr id="v-interconnect"><td class="vocab-term">Interconnect</td>
      <td>The on-chip wiring and switches that connect tiles to each other and
          to memory tiles. Unlike a CPU&rsquo;s shared bus, the NPU interconnect
          is <em>circuit-switched</em>: routes are configured at compile time,
          giving dedicated bandwidth with no contention.</td></tr>
  <tr id="v-throughput"><td class="vocab-term">Throughput vs Latency</td>
      <td><strong>Latency</strong> is how long one operation takes (e.g., one
          matmul = 1 &mu;s). <strong>Throughput</strong> is how many operations
          complete per second (e.g., 24 tiles &times; 1000 matmuls = 24M matmuls/s).
          The NPU excels at throughput through massive parallelism, even if
          single-tile latency is similar to a CPU.</td></tr>
  <tr id="v-doublebuf"><td class="vocab-term">Double buffering</td>
      <td>Using two buffers that swap roles each iteration: one is being read
          by the compute unit while the other is being filled by DMA.
          Also called &ldquo;ping-pong buffering.&rdquo; This hides data
          transfer latency behind computation.</td></tr>
</table>

<h3>2.3 How CPUs execute neural networks (and why it&rsquo;s slow)</h3>

<p>
When you call <code>y = torch.relu(x @ W)</code> on a CPU, here is what
actually happens at the hardware level:
</p>

<ol>
  <li>The weight matrix <code>W</code> lives in <a href="#g-dram" class="gref">DRAM</a>.
      The CPU issues a load instruction, and the data travels through the
      <strong><a href="#v-memwall" class="gref">memory hierarchy</a></strong>:
      DRAM &rarr; L3 cache &rarr; L2 cache &rarr; L1 cache &rarr;
      <a href="#v-register" class="gref">registers</a>.</li>
  <li>The CPU&rsquo;s <a href="#g-simd" class="gref">SIMD</a> units multiply a
      few rows at a time. The result goes back through the cache hierarchy to
      DRAM.</li>
  <li>For the next layer, the process repeats: read the activation from DRAM,
      read the next weight matrix from DRAM, compute, write back.</li>
</ol>

<p>
The fundamental problem is the <strong><a href="#v-memwall" class="gref">memory
wall</a></strong>: modern compute units can perform arithmetic far faster than
memory can supply data. A typical CPU can do ~1
<a href="#g-tflops" class="gref">TFLOPS</a> of
<a href="#g-bf16" class="gref">bf16</a> math, but its
<a href="#g-ddr" class="gref">DDR</a> bandwidth is ~50 GB/s. To keep the ALUs
busy doing a 128&times;128 matmul, you need 128&times;128&times;2 = 32 KB of
weight data delivered every ~1 &mu;s &mdash; which is 32 GB/s just for one
matrix. If you&rsquo;re running 4 layers, that&rsquo;s 128 GB/s, already
exceeding the memory bandwidth. The CPU stalls waiting for data.
</p>

<div class="highlight">
<strong>The memory wall in one sentence:</strong> Arithmetic is cheap; moving
data is expensive. The deeper you go in a neural network, the more time is
spent shuffling data between memory levels, not doing useful math.
</div>

<h3>2.4 The spatial dataflow alternative</h3>

<p>
A <strong>spatial architecture</strong> takes a radically different approach.
Instead of one fast processor with a deep cache hierarchy, it uses <em>many
small processors</em> (<a href="#v-tile" class="gref">tiles</a>), each with a
tiny but <em>extremely fast</em> local memory
(<a href="#g-sram" class="gref">SRAM</a>). Data moves between tiles through
dedicated hardware channels (<a href="#g-fifo" class="gref">FIFOs</a>), not
through shared caches.
</p>

<p>
Think of it like a factory assembly line versus a single master craftsman:
</p>

<ul>
  <li><strong>CPU (craftsman):</strong> One highly skilled worker goes to the
      warehouse (<a href="#g-dram" class="gref">DRAM</a>) for each part, brings
      it to the workbench (registers), processes it, takes the result back to the
      warehouse, gets the next part. Most time is spent walking, not working.</li>
  <li><strong><a href="#g-npu" class="gref">NPU</a> (assembly line):</strong>
      Many workers sit at their stations with all their parts already on their
      desk (SRAM). A conveyor belt (FIFO) moves the workpiece from one station
      to the next. Nobody walks anywhere.</li>
</ul>

<p>
The key property is <strong>data locality</strong>: once data is loaded into a
tile&rsquo;s 64 KB SRAM, it stays there for as many operations as you can do
on it. There is no cache to get evicted from, no bus to contend for. If a
128&times;128 weight matrix (32 KB) fits in SRAM, you can multiply against it
thousands of times at full speed &mdash; which is exactly what our recurrent
<a href="#g-mlp" class="gref">MLP</a> does.
</p>

<h3>2.5 Key hardware concepts used in this project</h3>

<p>
These concepts appear throughout the paper. You don&rsquo;t need to understand
every transistor, but knowing these ideas will make the design choices clear:
</p>

<h4><a href="#v-doublebuf" class="gref">Double buffering</a> (ping-pong)</h4>
<p>
If a tile needs to <em>receive</em> new data while <em>computing</em> on data
it already has, you need two buffers: one being filled by
<a href="#g-dma" class="gref">DMA</a> while the other is being read by the
compute unit. They swap roles each iteration. We use this for the activation
buffers: buffer A holds the input, the tile computes into buffer B, then B
becomes the input and A becomes the output.
</p>

<h4>Tiled matrix layout</h4>
<p>
A &ldquo;row-major&rdquo; matrix stores elements left-to-right, top-to-bottom:
<code>[[a, b], [c, d]]</code> becomes <code>[a, b, c, d]</code>. The
<a href="#g-aie" class="gref">AIE</a> matmul unit instead expects a
<strong>blocked (tiled) layout</strong>: the matrix is divided into 8&times;8
sub-matrices, and each block is stored contiguously. This matches the
<a href="#g-mmul" class="gref">MMUL</a> hardware which multiplies 8&times;8
blocks in one operation. The <code>to_tiled()</code> function handles this
conversion.
</p>

<h4><a href="#g-fifo" class="gref">ObjectFIFOs</a> and data movement</h4>
<p>
On a CPU, data movement is implicit &mdash; you <code>load</code> from an
address and the cache hierarchy handles the rest. On the NPU, you must
<em>explicitly</em> program every data transfer: &ldquo;move 4 KB from DDR
address X to tile (3, 2)&rsquo;s input buffer.&rdquo;
<a href="#g-iron" class="gref">IRON</a>&rsquo;s <code>ObjectFifo</code>
abstraction makes this manageable: you declare a typed channel between a
producer and a consumer, and the compiler generates the DMA configurations.
</p>

<h4>The invocation overhead problem</h4>
<p>
Every time the host CPU tells the NPU to run, there is a fixed overhead of
~120 &mu;s for driver calls, instruction dispatch, and DMA setup. This is
analogous to the overhead of launching a CUDA kernel on a GPU. If your actual
compute takes only 1 &mu;s (as it does for a single small matmul), you are
spending 99% of the time on overhead. The solution is to do <em>lots of work
per invocation</em> &mdash; hence the hardware loop that repeats thousands of
matmuls before returning to the host.
</p>

<h3>2.6 How to read the rest of this paper</h3>

<p>
With this background, the rest of the paper should be accessible:
</p>

<ul>
  <li><strong>Section 3</strong> is the NPU primer: physical hardware,
      tile microarchitecture, the C++ kernel API, the IRON Python
      programming model, design patterns, and hardware constraints.</li>
  <li><strong>Section 5</strong> presents the neural network architecture:
      block-recurrent character language model mapped to the NPU pipeline.</li>
  <li><strong>Section 6</strong> shows quality and throughput results.</li>
  <li><strong>Section 7</strong> describes the software toolchain.</li>
  <li><strong>Section 8</strong> maps the code structure to the concepts.</li>
</ul>

<h3 id="s-nn-fundamentals">2.7 Neural network fundamentals (for hardware engineers)</h3>

<p>
If you work in hardware or systems and have heard &ldquo;neural network&rdquo;
but never implemented one, this section gives you just enough to follow the
architecture decisions in this paper.
</p>

<h4>What is a neural network?</h4>
<p>
A neural network is a function built by stacking <strong>layers</strong>.
Each layer applies a linear transformation (matrix multiply) followed by a
<strong>non-linearity</strong> (a simple element-wise function). For example:
</p>

<pre>
y = ReLU(x @ W + b)      # one layer
</pre>

<p>
Here <code>x</code> is the input vector, <code>W</code> is a learnable
<strong>weight matrix</strong>, <code>b</code> is a learnable
<strong>bias vector</strong>, and
<a href="#g-relu" class="gref">ReLU</a>(z)&nbsp;=&nbsp;max(z,&nbsp;0) is the
non-linearity. Without the non-linearity, stacking layers would collapse to
a single matrix multiply &mdash; the non-linearity is what gives depth its
power. Stacking many such layers creates a &ldquo;deep&rdquo; network that
can learn increasingly abstract representations of its input.
</p>

<h4>Language models: predicting the next character</h4>
<p>
A <strong>language model</strong> assigns probabilities to sequences of text.
The simplest version predicts the next character (or word) given everything
that came before. If the model has read &ldquo;KING RICHAR&rdquo;, it should
assign high probability to &ldquo;D&rdquo;. At generation time, we sample
from this probability distribution to produce text one character at a time.
</p>

<p>
The quality of a language model is measured by <strong>cross-entropy loss</strong>
&mdash; roughly, how surprised the model is by each character in a held-out
test set. Lower is better. The related metric
<strong>perplexity</strong>&nbsp;=&nbsp;e<sup>loss</sup> can be interpreted
as &ldquo;the model is as confused as if choosing uniformly among <em>N</em>
options.&rdquo; A perplexity of 11 means the model is, on average, as uncertain
as if it were picking among 11 equally likely characters (out of ~65 in our
vocabulary).
</p>

<h4>Recurrence: memory across time</h4>
<p>
A <strong>recurrent</strong> network maintains a <strong>hidden state</strong>
<code>h</code> that carries information from one time step to the next:
</p>

<pre>
h = f(h, embed(char))     # update hidden state with new character
logits = h @ W_out + b    # predict next character from hidden state
</pre>

<p>
The function <code>f</code> reads the previous hidden state and the current
character&rsquo;s <strong>embedding</strong> (a learned vector that represents
each character as a point in continuous space), and produces a new hidden state.
This is fundamentally different from feedforward networks that process each
input independently: the hidden state acts as the model&rsquo;s
&ldquo;memory&rdquo; of everything it has read so far.
</p>

<h4>Making deep networks trainable</h4>
<p>
Simply stacking 32 layers of <code>h = ReLU(h @ W + b)</code> does not work:
the hidden state either explodes to infinity or collapses to zero as it
passes through many layers. Three techniques solve this:
</p>

<ul>
  <li><strong>Residual connections:</strong> Instead of <code>h = f(h)</code>,
      compute <code>h = h + f(h)</code>. The identity &ldquo;shortcut&rdquo;
      ensures that gradients can flow backwards through the network even if
      <code>f</code> has very small gradients. This was the key innovation
      behind deep learning&rsquo;s leap from ~10 to ~100+ layer networks.</li>
  <li><strong>Normalisation (RMSNorm):</strong> Before each layer, rescale the
      hidden state to have unit root-mean-square:
      <code>h / sqrt(mean(h&sup2;) + &epsilon;)</code>. This prevents the
      activations from growing exponentially across layers or time steps.</li>
  <li><strong>Input injection:</strong> Add the character embedding at every
      layer (not just the first), so each layer can directly access the current
      input rather than relying on information that has been progressively
      transformed and potentially diluted by many preceding layers.</li>
</ul>

<h4 id="s-bptt">Training: learning by gradient descent</h4>
<p>
The weight matrices start with random values. <strong>Training</strong> means
repeatedly: (1) run the model on a batch of text, (2) compute how wrong the
predictions are (the loss), (3) compute the gradient of the loss with respect
to every weight (using <strong>backpropagation</strong> &mdash; the chain
rule applied systematically through the network), and (4) nudge each weight
in the direction that reduces the loss. For recurrent networks, this is called
<strong>backpropagation through time (BPTT)</strong> because gradients flow
backwards through both layers and time steps.
</p>

<p>
Training typically happens on a GPU because GPUs have massive memory bandwidth
and can process large batches in parallel. The NPU enters at
<strong>inference</strong> time: once the weights are learned, we deploy the
trained model to the NPU for fast, efficient text generation.
</p>

<h2>3. The Hardware</h2>

<p>
The AMD <a href="#g-xdna" class="gref">XDNA</a>&nbsp;2
<a href="#g-npu" class="gref">NPU</a> in the Ryzen AI 9 HX 370 (codename
Strix Point) is a tiled spatial-dataflow processor with the following structure:
</p>

<div class="figure">
  <img src="xdna2_hardware.png" alt="XDNA 2 tile array">
  <div class="caption">
    Figure 1: Physical tile array of the AMD XDNA&nbsp;2 NPU. 32 compute
    <a href="#v-tile" class="gref">tiles</a> (rows 2&ndash;5) each contain
    ~64 KB <a href="#g-sram" class="gref">SRAM</a> and a
    <a href="#g-bf16" class="gref">bf16</a>
    <a href="#g-mmul" class="gref">MMUL</a> unit. 8 memory tiles (row 1,
    512 KB each) serve as on-chip L2 buffers and routing hubs. 8
    <a href="#v-shim" class="gref">shim tiles</a> (row 0) provide
    <a href="#g-dma" class="gref">DMA</a> access to host
    <a href="#g-ddr" class="gref">DDR</a> memory.
  </div>
</div>

<table>
  <tr><th>Property</th><th>Value</th></tr>
  <tr><td>Compute tiles</td><td>32 (8 columns &times; 4 rows)</td></tr>
  <tr><td>Memory tiles</td><td>8 (512 KB each, 4 MB total)</td></tr>
  <tr><td>Per-tile <a href="#g-sram" class="gref">SRAM</a></td><td>~64 KB data memory</td></tr>
  <tr><td>Per-tile compute</td><td><a href="#g-bf16" class="gref">bf16</a> <a href="#g-mmul" class="gref">MMUL</a> unit, <a href="#g-vliw" class="gref">VLIW</a>+<a href="#g-simd" class="gref">SIMD</a> core</td></tr>
  <tr><td>Clock frequency</td><td>~1.5 GHz</td></tr>
  <tr><td>Peak <a href="#v-throughput" class="gref">throughput</a></td><td><strong>25 <a href="#g-tflops" class="gref">TFLOPS</a></strong> (bfloat16)</td></tr>
  <tr><td><a href="#v-interconnect" class="gref">Interconnect</a></td><td>ObjectFIFOs (tile-to-tile <a href="#v-doublebuf" class="gref">double-buffered</a> streams)</td></tr>
  <tr><td>Power envelope</td><td>~6 W</td></tr>
</table>

<h3>3.1 Why spatial dataflow matters</h3>

<p>
On a CPU, a matrix multiply reads data from DRAM through multiple cache levels.
Each layer of a neural network bounces activations through L1&rarr;L2&rarr;L3&rarr;DRAM
and back. On the NPU, data stays in 64 KB tile SRAM between operations &mdash;
zero cache misses, zero bus contention. This is the source of the NPU's advantage
for deep, narrow computations.
</p>

<div class="highlight">
<strong>Why on-chip matters:</strong> A single large <a href="#g-gemm" class="gref">GEMM</a>
achieves only 2.49 TFLOPS on the NPU (10% of peak) because it is
<em>memory-bandwidth limited</em> &mdash; data must stream from DDR. The NPU
wins when data <strong>stays on-chip</strong>. Our architecture keeps weights
in tile SRAM and passes activations tile-to-tile through FIFOs.
</div>

<h3>3.2 Inside an AIE tile</h3>

<p>
Each of the 32 compute tiles is a self-contained processor with four main
subsystems.  Understanding what lives inside a tile is essential for knowing
what computation is practical at the single-tile level.
</p>

<h4>VLIW+SIMD core</h4>
<p>
The heart of each tile is a <a href="#g-vliw" class="gref">VLIW</a> processor
that can issue multiple operations per clock cycle: a vector multiply, a vector
add, a scalar operation, and a memory load/store &mdash; all in the same cycle.
The vector unit operates on <strong>256-bit vectors</strong>: 16&nbsp;elements
of <a href="#g-bf16" class="gref">bfloat16</a>, 32&nbsp;elements of
<a href="#g-int8" class="gref">int8</a>, or 8&nbsp;elements of float32.
This <a href="#g-simd" class="gref">SIMD</a> width determines the natural
&ldquo;chunk size&rdquo; for all tile-level programming: data buffers must
be aligned to 64&nbsp;bytes and sized in multiples of 32&nbsp;bfloat16
elements (64&nbsp;bytes) for efficient access.
</p>

<h4>Matrix multiply (MMUL) unit</h4>
<p>
Dedicated hardware multiplies small matrix tiles in a single instruction.
The <a href="#g-mmul" class="gref">MMUL</a> unit operates on blocks of size
<code>r&times;s&times;t</code>, meaning it multiplies an
<code>r&times;s</code> sub-matrix of A by an <code>s&times;t</code>
sub-matrix of B and accumulates into an <code>r&times;t</code> result.
For <a href="#g-bf16" class="gref">bfloat16</a> on XDNA&nbsp;2:
</p>

<ul>
  <li><strong>Native bf16:</strong> (r,&thinsp;s,&thinsp;t) = (4,&thinsp;8,&thinsp;8)
      &mdash; 256 multiply-accumulate operations per instruction.</li>
  <li><strong><a href="#g-bfp16" class="gref">BFP16</a> emulation:</strong>
      (r,&thinsp;s,&thinsp;t) = (8,&thinsp;8,&thinsp;8) &mdash;
      512 MACs per instruction, double the throughput. This groups bf16 values
      into blocks sharing an exponent, trading negligible precision for 2&times;
      throughput. We use this mode throughout.</li>
</ul>

<p>
At ~1.5&nbsp;GHz, a single tile can sustain
512&nbsp;MACs &times; 2&nbsp;FLOPs/MAC &times; 1.5&nbsp;GHz &asymp;
<strong>~768 <a href="#g-gflops" class="gref">GFLOPS</a></strong> peak in
<a href="#g-bfp16" class="gref">BFP16</a> mode.  Multiplied by 32&nbsp;tiles,
this gives the advertised 25&nbsp;<a href="#g-tflops" class="gref">TFLOPS</a>
system peak.
</p>

<h4>Local SRAM</h4>
<p>
Each tile has <strong>~64 KB</strong> of data
<a href="#g-sram" class="gref">SRAM</a> &mdash; <em>all</em> the memory
the core can access directly.  There are no caches; every byte is explicitly
managed by the programmer.  A separate, smaller instruction memory holds the
compiled kernel.  The 64&nbsp;KB budget must fit: (1) weight data,
(2) input and output activation buffers, and (3) stack and local variables.
This is the single most important constraint in NPU architecture co-design.
</p>

<h4>Per-tile DMA engine</h4>
<p>
Each tile has its own <a href="#g-dma" class="gref">DMA</a> engine that moves
data between tile SRAM and neighbouring tiles or the
<a href="#v-memtile" class="gref">MemTile</a>.  The programmer does not call
DMA functions directly; instead, <a href="#g-iron" class="gref">IRON</a>&rsquo;s
<a href="#g-fifo" class="gref">ObjectFIFOs</a> abstract these transfers as typed
channels.  The compiler generates <a href="#g-bd" class="gref">buffer descriptors</a>
that program the DMA with source addresses, sizes, and stride patterns.
</p>

<div class="key-insight">
<strong>Mental model:</strong> Think of each tile as a tiny computer with
64&nbsp;KB of RAM, a fast matrix-multiply co-processor, and a mail slot
(DMA/FIFO) for sending and receiving data from neighbours.  There is no
shared memory and no operating system &mdash; just your compiled C++ function
running in an infinite loop, acquiring data, computing, and releasing results.
</div>

<h3>3.3 Writing AIE kernels (the C++ API)</h3>

<p>
Each tile runs a C++ function compiled by the AIE-specific
<a href="#g-llvm" class="gref">Peano/LLVM-AIE</a> compiler.  The
<code>aie_api/aie.hpp</code> header provides intrinsics that map directly
to hardware instructions.  This section walks through the key patterns using
our actual <code>norm_matmul_relu.cc</code> kernel as an example.
</p>

<h4>Vector types</h4>
<p>
The fundamental data type is a fixed-width SIMD vector:
</p>
<pre>
aie::vector&lt;bfloat16, 16&gt;   // 16 × bf16 = 256 bits (one vector register)
aie::vector&lt;int8, 32&gt;       // 32 × int8  = 256 bits
aie::vector&lt;float, 8&gt;       //  8 × f32   = 256 bits
</pre>
<p>
Load and store 16 bfloat16 elements at once with
<code>aie::load_v&lt;16&gt;(ptr)</code> and
<code>aie::store_v(ptr,&thinsp;vec)</code>.  Element-wise operations like
<code>aie::max(vec,&thinsp;zeros)</code> (ReLU) operate on full vectors.
</p>

<h4>Matrix multiply: <code>aie::mmul</code></h4>
<p>
The core workhorse is the <code>aie::mmul&lt;r,&thinsp;s,&thinsp;t&gt;</code>
class, which wraps the MMUL hardware unit:
</p>
<pre>
using MMUL = aie::mmul&lt;8, 8, 8, bfloat16, bfloat16, accauto&gt;;

MMUL acc(zeros);                      // zero-initialise accumulator in registers
auto A = aie::load_v&lt;MMUL::size_A&gt;(pA);   // load 8×8 A tile (64 bf16)
auto B = aie::load_v&lt;MMUL::size_B&gt;(pB);   // load 8×8 B tile (64 bf16)
acc.mac(A, B);                        // acc += A × B  (hardware MAC instruction)
auto result = acc.to_vector&lt;bfloat16&gt;();    // extract 8×8 result as bf16
aie::store_v(pC, aie::max(result, zeros));  // store with fused ReLU
</pre>
<p>
The <code>accauto</code> type uses higher-precision accumulators internally
(typically 48-bit or fp32), converting back to bfloat16 only at
<code>to_vector</code>.  This preserves numerical quality during large
reductions.
</p>

<h4>2&times;2 tile expansion</h4>
<p>
Processing one 8&times;8 block at a time under-utilises registers.  Our
kernels process <strong>four</strong> output blocks per inner-loop iteration
(a 2&times;2 grid of MMUL accumulators).  This keeps 4 accumulator registers
active simultaneously, allowing the VLIW scheduler to interleave loads and
multiplies:
</p>
<pre>
// Four accumulators for a 2×2 grid of output blocks
MMUL C00(zeros), C01(zeros), C10(zeros), C11(zeros);

for (i = 0; i &lt; K/8; ++i) {
    A0 = load(pA1);  A1 = load(pA2);     // two rows of A
    B0 = load(pB1);  B1 = load(pB2);     // two columns of B
    C00.mac(A0, B0);  C01.mac(A0, B1);   // top row of output
    C10.mac(A1, B0);  C11.mac(A1, B1);   // bottom row of output
}
// Store with fused ReLU: max(result, 0)
store(pC, max(C00.to_vector(), zeros));
// ... C01, C10, C11 similarly
</pre>
<p>
This pattern yields roughly 2&times; throughput over naive single-block
processing, because the compiler can pipeline loads with multiplies
across the four independent accumulations.
</p>

<h4>Compiler hints</h4>
<p>
Two pragmas are critical for performance:
</p>
<ul>
  <li><code>chess_prepare_for_pipelining</code> &mdash; tells the
      &ldquo;chess&rdquo; scheduler (the AIE backend compiler) to overlap
      iterations of the loop body, like a CPU&rsquo;s instruction pipeline
      but at the loop level.  Without this, each iteration starts only after
      the previous one fully completes.</li>
  <li><code>chess_flatten_loop</code> &mdash; fully unrolls the inner loop
      so the compiler can schedule all operations as one large block.  Used
      on tight inner loops where the iteration count is small and known at
      compile time.</li>
</ul>

<h4>The fused norm+matmul+ReLU kernel</h4>
<p>
Our <code>norm_matmul_relu.cc</code> kernel performs three operations in a
single tile invocation:
</p>
<ol>
  <li><strong>RMSNorm in-place</strong> (scalar float32): For each of the
      B rows, accumulate the sum of squares, compute the inverse RMS using
      8 Babylonian iterations (avoiding library dependencies), and scale
      each element by the learned scale vector.  Cost: ~650 scalar float
      ops per row &times; B rows &asymp; 25&nbsp;&mu;s at M=48, K=128.</li>
  <li><strong>Matmul + ReLU</strong> (vectorised bf16): The 2&times;2 tile
      expansion described above, with ReLU fused into the store step.
      At the end: <code>store(max(acc.to_vector(), zeros))</code>.</li>
</ol>

<p>
The key implementation detail: weights and scale share one buffer.
The weight matrix (H&times;H = 32&nbsp;KB in bf16 tiled layout) occupies
the first H&times;H elements, and the scale vector (H elements = 256 bytes)
is appended after it.  The kernel extracts the scale pointer:
</p>
<pre>
const bfloat16 *scale = w_and_scale + DIM_K * DIM_N;
rms_norm_inplace(input, scale);    // step 1: normalise in-place
matmul_relu_2x2(input, w, output); // step 2: matmul + ReLU
</pre>

<h3>3.4 IRON: the Python programming model</h3>

<p>
<a href="#g-iron" class="gref">IRON</a> is a Python API that lets you describe
<em>what</em> each tile computes and <em>how</em> data moves between tiles,
without writing any MLIR or assembly by hand.  The IRON compiler translates
your Python program into MLIR-AIE, then into the binary
<a href="#v-bitstream" class="gref">bitstream</a> (<code>.xclbin</code>) and
instruction sequence (<code>.bin</code>) that configure the hardware.
</p>

<h4><code>ObjectFifo</code>: explicit data channels</h4>
<p>
The most important abstraction.  An ObjectFifo is a hardware queue between
a producer tile and a consumer tile:
</p>
<pre>
fifo = ObjectFifo(buffer_type, name="my_fifo", depth=1)
</pre>
<p>
The <code>depth</code> is how many buffers are in the queue (depth=1 means
single-buffering; depth=2 gives
<a href="#v-doublebuf" class="gref">double buffering</a>).
Each end has two operations:
</p>
<ul>
  <li><code>fifo.prod()</code> / <code>fifo.cons()</code> &mdash;
      get the producer or consumer endpoint.</li>
  <li><code>acquire(n)</code> / <code>release(n)</code> &mdash;
      <strong>blocking</strong> acquire (wait until n buffers are available)
      and release (signal the other end that n buffers are done).
      This is the <em>only</em> synchronisation mechanism between tiles.</li>
</ul>

<p>
Three routing operations handle the MemTile as an on-chip switch:
</p>
<ul>
  <li><code>.forward()</code> &mdash; pass data through MemTile unchanged
      (DDR input &rarr; MemTile &rarr; compute tile).</li>
  <li><code>.split(offsets, types)</code> &mdash; one large buffer arriving
      at MemTile is sliced into N smaller buffers distributed to N tiles.
      Used for distributing per-stage weights from a single DDR transfer.</li>
  <li><code>.join(offsets, types)</code> &mdash; the reverse of split:
      N tiles&rsquo; outputs are gathered into one large buffer for DDR.</li>
</ul>

<div class="highlight">
<strong>ObjectFIFO &rlarr; hardware mapping:</strong> Each ObjectFifo compiles
to a pair of <a href="#g-bd" class="gref">buffer descriptors</a> (BDs) in
the DMA engines of the producer and consumer tiles.  The BDs describe the
memory addresses, sizes, and stride patterns.  The <code>acquire/release</code>
calls become hardware lock operations: a tile stalls until the BD signals data
is ready.  There is zero software overhead at runtime.
</div>

<h4><code>Kernel</code>: compiled C++ function reference</h4>
<pre>
mm_relu = Kernel(
    "norm_matmul_relu_bf16_bf16",   # C function name (extern "C")
    "mlp_kernels.a",                # compiled library
    [act_type, weight_type, act_type]  # argument types
)
</pre>
<p>
A Kernel object is just a <em>reference</em> to a compiled C++ function.
It tells IRON what to link onto each tile.  The argument types must match the
ObjectFifo buffer types.
</p>

<h4><code>Worker</code>: one function per tile</h4>
<pre>
def stage_body(of_in, of_out, of_w, mm_relu_fn):
    x = of_in.acquire(1)       # wait for input
    y = of_out.acquire(1)      # wait for output buffer
    w = of_w.acquire(1)        # wait for weights
    mm_relu_fn(x, w, y)        # compute: y = ReLU(RMSNorm(x) @ W)
    of_w.release(1)
    of_in.release(1)
    of_out.release(1)          # signal output ready

Worker(stage_body,
       fn_args=[in_fifo.cons(), out_fifo.prod(), wt_fifo.cons(), mm_relu],
       placement=Tile(col=0, row=2))
</pre>
<p>
A Worker binds a Python function to a specific physical tile.  The function
describes the tile&rsquo;s behaviour: acquire inputs from FIFOs, call the
kernel, release outputs.  IRON compiles this into the tile&rsquo;s instruction
sequence.  Each tile gets exactly one Worker.
</p>

<h4><code>Runtime</code>: host&ndash;device orchestration</h4>
<p>
The Runtime describes what the host CPU does: load data into the NPU,
start the workers, and drain results back:
</p>
<pre>
rt = Runtime()
with rt.sequence(input_type, weight_type, output_type) as (inp, wts, out):
    rt.start(*workers)          # launch all 32 workers (they block on FIFOs)
    tg = rt.task_group()
    for col in range(8):
        rt.fill(ddr_in[col].prod(), inp, tap, task_group=tg)   # DMA: host &rarr; tile
    rt.finish_task_group(tg)
    # ... fill weights, drain outputs similarly
</pre>
<p>
<code>task_group()</code> groups multiple DMA transfers that execute in
parallel (e.g., all 8 columns fill simultaneously).
<code>TensorAccessPattern</code> (TAP) objects describe how to slice the
host buffer into per-column chunks:
</p>
<pre>
tap = TensorAccessPattern(
    (1, total_size),    # shape of the host buffer
    col * B * H,        # offset: start of this column's slice
    [1, 1, B, H],       # logical shape for DMA iteration
    [0, 0, H, 1],       # strides
)
</pre>

<h4><code>Program</code>: putting it all together</h4>
<pre>
program = Program(NPU2(), rt)
mlir_module = program.resolve_program(SequentialPlacer())
</pre>
<p>
<code>NPU2()</code> selects the XDNA&nbsp;2 hardware target (8&nbsp;columns,
4&nbsp;compute rows).  <code>SequentialPlacer()</code> assigns Workers to
tiles in order.  The <code>resolve_program</code> call generates the complete
MLIR-AIE module, which is then compiled to the final
<code>.xclbin</code> bitstream and <code>.bin</code> instruction file.
</p>

<h3>3.5 Design patterns for the NPU</h3>

<p>
With the hardware and programming model understood, here are the key patterns
for mapping computation graphs onto the tile array.
</p>

<h4>Pattern 1: Column parallelism (data-parallel batching)</h4>
<p>
All 8 columns are identical and independent: they run the same kernel on
different slices of the input batch.  If you have B&nbsp;samples per column,
the total batch is 8&times;B.  This is pure data parallelism &mdash; no
communication between columns.  Each column has its own set of ObjectFIFOs
and its own weight copy in tile SRAM.
</p>
<p>
<strong>When to use:</strong> Always, for throughput.  8 columns &times; B
samples per column amortises the ~120&nbsp;&mu;s XRT invocation overhead
across 8&times;B results.
</p>

<h4>Pattern 2: Row pipelining (sequential stages)</h4>
<p>
Within each column, the 4 compute rows form a pipeline.  Data flows from
row&nbsp;2 &rarr; row&nbsp;3 &rarr; row&nbsp;4 &rarr; row&nbsp;5 through
tile-to-tile ObjectFIFOs (no MemTile hop, zero-copy).  Each row applies a
different transformation (different weight matrix).  The pipeline behaviour
emerges naturally from FIFO blocking: row&nbsp;3&rsquo;s
<code>acquire()</code> stalls until row&nbsp;2&rsquo;s
<code>release()</code> signals output is ready.
</p>
<p>
<strong>When to use:</strong> For deep, sequential computation (stacked layers).
Each pipeline stage adds one more learnable transformation without any DDR
round-trip.
</p>

<h4>Pattern 3: Weight-stationary execution</h4>
<p>
The weight matrix stays in tile SRAM for the entire NPU invocation.  Only
activations stream through the tile (one input buffer in, one output buffer
out).  This is the optimal strategy when:
</p>
<ul>
  <li>The weight matrix fits in SRAM (our 128&times;128 matrix = 32&nbsp;KB
      &lt; 64&nbsp;KB per tile).</li>
  <li>The same weights are applied to many input samples (batch processing).</li>
</ul>
<p>
The alternative (activation-stationary, streaming weights) is needed when
weights exceed tile SRAM &mdash; but then the operation becomes memory-bandwidth
limited and the NPU loses its advantage over the CPU.
</p>

<h4>Pattern 4: SRAM budget planning</h4>
<p>
Every design starts with this calculation.  For our architecture (H=128,
B=48, BFP16 mode):
</p>
<table>
  <tr><th>Component</th><th>Size (bytes)</th><th>Notes</th></tr>
  <tr><td>Weight matrix W</td><td>128&times;128&times;2 = 32,768</td>
      <td>bf16, 8&times;8 tiled layout</td></tr>
  <tr><td>Scale vector</td><td>128&times;2 = 256</td>
      <td>bf16, flat (appended to W)</td></tr>
  <tr><td>Input activation buffer</td><td>48&times;128&times;2 = 12,288</td>
      <td>bf16, one batch</td></tr>
  <tr><td>Output activation buffer</td><td>48&times;128&times;2 = 12,288</td>
      <td>bf16, one batch</td></tr>
  <tr><td>Stack + code locals</td><td>~1,500</td><td>Estimated</td></tr>
  <tr style="font-weight:600;">
      <td>Total</td><td>~59,100 (57.7 KB)</td>
      <td>Fits in 64 KB &check;</td></tr>
</table>
<p>
This budget leaves ~6&nbsp;KB of headroom.  Increasing H to 192 would need
192&times;192&times;2 = 72&nbsp;KB for weights alone &mdash; over the 64&nbsp;KB
limit.  This is why H=128 is the maximum hidden dimension for single-tile
weight-stationary execution in bf16.
</p>

<h4>Pattern 5: Kernel fusion</h4>
<p>
Unfused operations (separate kernels for norm, matmul, and relu) require
three DMA round-trips per stage, three kernel invocations, and three sets
of buffers.  Fusing them into one kernel:
</p>
<ul>
  <li>Eliminates intermediate buffers (input is normalised in-place).</li>
  <li>Keeps data in registers between operations.</li>
  <li>Reduces the number of Worker <code>acquire/release</code> cycles.</li>
  <li>Allows the compiler to pipeline the norm reduction with the matmul
      loads.</li>
</ul>
<p>
In our benchmarks, the fused <code>matmul_relu</code> kernel achieved
<strong>23.93 <a href="#g-tflops" class="gref">TFLOPS</a></strong> (95.7%
of peak) versus 15.98 TFLOPS unfused at B=48 &mdash; a 50% throughput
improvement from fusion alone.
</p>

<h4>Pattern 6: Amortising invocation overhead</h4>
<p>
Every NPU call has ~120&nbsp;&mu;s of <a href="#g-xrt" class="gref">XRT</a>
driver overhead.  If each call does only one matmul (~1&nbsp;&mu;s of
compute), you waste 99% of time on overhead.  Solutions:
</p>
<ul>
  <li><strong>Large batches:</strong> B=48 means 48 rows of 128 multiplied
      per tile, 24&times; more work per call than B=2.</li>
  <li><strong>Deep pipelines:</strong> 4 stages per call means 4 matmuls
      of compute for one overhead.</li>
  <li><strong>Hardware loops:</strong> For throughput benchmarks (not
      autoregressive generation), a <code>range_(N)</code> loop repeats
      the computation N times <em>on-chip</em> before returning to the host.
      At N=1000, compute completely dominates overhead.</li>
</ul>

<h3>3.6 Hardware constraints and gotchas</h3>

<p>
The NPU has several constraints that are not obvious from high-level
descriptions.  We discovered each of these through trial and error during
development.
</p>

<h4>Buffer descriptor size limit (10-bit fields)</h4>
<p>
The <a href="#g-dma" class="gref">DMA</a> engines in
<a href="#v-shim" class="gref">shim tiles</a> use
<a href="#g-bd" class="gref">buffer descriptors</a> with 10-bit size fields,
meaning each dimension of a
<code>TensorAccessPattern</code> must be &le;&nbsp;1024.  If your tensor has
a dimension larger than 1024 (e.g., B&times;H = 48&times;128 = 6144), you must
factor it: use shape <code>[tiles_per_col,&thinsp;B,&thinsp;H]</code> instead
of <code>[tiles_per_col,&thinsp;B*H]</code>.  The compiler will reject designs
that violate this, with the cryptic error
<code>&ldquo;Size 0 exceeds [0:1023] range&rdquo;</code>.
</p>

<h4>MemTile routing limits</h4>
<p>
Each MemTile has approximately 6 master ports going northward (toward compute
rows).  A multi-row design that uses <code>split()</code> + <code>forward()</code>
+ <code>join()</code> consumes 3 FIFOs per row of compute tiles.  This means:
</p>
<ul>
  <li><strong>3 compute rows</strong> (24 tiles) works reliably:
      3&times;3 = 9 FIFOs, within limits.</li>
  <li><strong>4 compute rows</strong> (32 tiles) sometimes fails with
      <code>&ldquo;Unable to find a legal routing&rdquo;</code>,
      depending on the FIFO topology.</li>
</ul>
<p>
Our design uses all 4 compute rows by routing inter-tile activations
<em>directly tile-to-tile</em> (not through MemTile), reserving MemTile
ports for DDR-facing traffic only (weight split, activation forward/join).
</p>

<h4>ShimDMA channel limits</h4>
<p>
There are 16 ShimDMA channels total (2 per column).  Each DDR-facing
ObjectFifo consumes one channel.  With 8 columns, our design uses 3
DDR FIFOs per column (input, weight, output) = 24, but the compiler
shares channels across task groups that don&rsquo;t overlap in time.
Exceeding 16 <em>simultaneous</em> channels causes compilation failure.
</p>

<h4>ObjectFIFO acquire/release inside loops</h4>
<p>
Placing <code>acquire()</code> and <code>release()</code> calls inside an
IRON <code>range_(N)</code> hardware loop can cause deadlocks: the tile&rsquo;s
DMA and compute core enter a circular dependency where each waits for the
other.  The fix: acquire all FIFOs <em>before</em> the loop, put only kernel
calls inside, and release <em>after</em> the loop.  This was the most subtle
bug we encountered and caused silent hardware timeouts
(<code>ERT_CMD_STATE_TIMEOUT</code>) with no diagnostic output.
</p>

<h4>Tiled memory layout</h4>
<p>
The MMUL hardware expects weight matrices in <strong>blocked (tiled)
format</strong>: the matrix is divided into 8&times;8 sub-matrices stored
contiguously, rather than row-major order.  A 128&times;128 matrix becomes
16&times;16 blocks of 64 elements each.  Host-side code must convert with
<code>to_tiled()</code> before sending weights to the NPU, and
<code>from_tiled()</code> to interpret results.  Forgetting this conversion
produces silently wrong results, not errors.
</p>

<h3>3.7 Putting it all together</h3>

<p>
Let&rsquo;s trace how these pieces combine to build a complete NPU design,
starting small and scaling up.
</p>

<h4>Step 1: Single-tile matmul</h4>
<p>
The simplest possible design: one tile, one weight matrix, one input, one
output.  Two ObjectFIFOs (in, out), one Kernel, one Worker.  The host fills
the input FIFO, the tile computes <code>y&thinsp;=&thinsp;ReLU(x&thinsp;@&thinsp;W)</code>,
and the host drains the output.  This uses 1 of 32 tiles and achieves ~1/32
of peak throughput, but it validates the kernel and data layout.
</p>

<h4>Step 2: Column parallelism (8 tiles)</h4>
<p>
Replicate the single-tile design across 8 columns. Each column gets its own
ObjectFIFOs and Worker, processing a different batch slice.  The Runtime uses
<code>task_group()</code> to issue 8 parallel DMA fills.  Throughput scales
linearly: 8&times; the single-tile result.
</p>

<h4>Step 3: Row pipelining (4&times;8 = 32 tiles)</h4>
<p>
Add 3 more rows per column.  Inter-tile FIFOs chain the stages:
row&nbsp;2&rsquo;s output feeds row&nbsp;3&rsquo;s input, and so on.
Weights are distributed via <code>.split()</code> through the MemTile.
Now each NPU call applies 4 sequential transformations &mdash; equivalent
to 4 layers of a neural network.
</p>

<h4>Step 4: Host-orchestrated depth (8 blocks &times; 4 stages = 32 layers)</h4>
<p>
The host calls the NPU 8 times, each time loading a different set of 4 weight
matrices.  Between calls, the CPU applies normalisation and residual
connections.  The result is a 32-layer model running on 32 tiles, with the
architecture described in the next section.
</p>

<div class="key-insight">
<strong>Design methodology:</strong> Start with one tile, get it working,
then scale horizontally (columns) for throughput and vertically (rows) for
depth.  Each scaling step adds hardware resources without changing the
per-tile kernel.  The host orchestration layer (IRON Runtime) manages the
increasing complexity.
</div>

<h2>5. The Architecture: Block-Recurrent Character LM</h2>

<p>
We build a character-level language model whose architecture is dictated by the
NPU&rsquo;s physical tile array. The model has 32 layers (one per compute tile),
grouped into 8 blocks of 4 layers (one block per column). Each block maps to
a single NPU pipeline call. Between blocks, the CPU applies the operations
that the pipeline cannot: normalisation, embedding injection, and residual
connections.
</p>

<h3>5.1 The block-recurrent computation</h3>

<p>
For each character in the input sequence, the model updates a hidden state
<code>h</code> (a 128-dimensional <a href="#g-bf16" class="gref">bfloat16</a>
vector) through 8 blocks:
</p>

<pre>
for each character:
    for each block g = 0..7:
        CPU:  h_b = h + embed(char) + bias_g              &larr; inject input
        NPU:  for stage j = 0..3:
                  h_b = ReLU(RMSNorm(h_b) @ W[4g+j])      &larr; fused on-chip
        CPU:  h = h + h_b                                  &larr; residual connection
    CPU:  h = RMSNorm(h)                                   &larr; post-norm
    CPU:  logits = h @ W_out + b_out                       &larr; predict next char
</pre>

<p>
Each of the 8 NPU calls runs a 4-stage pipeline where each stage fuses
RMSNorm&nbsp;+&nbsp;matmul&nbsp;+&nbsp;ReLU:
</p>

<ol>
  <li>The CPU prepares the block input: add the character embedding and a
      per-block learned bias to the hidden state (two vector additions).</li>
  <li>The prepared vector is sent to the NPU, where it passes through 4 tiles
      in sequence. Each tile normalises its input (RMSNorm), multiplies by its
      own weight matrix W<sub>i</sub> (128&times;128), and applies
      <a href="#g-relu" class="gref">ReLU</a> &mdash; all in a single fused
      <code>norm_matmul_relu</code>
      <a href="#v-kernel" class="gref">kernel</a>.</li>
  <li>The NPU returns the result to the CPU, which adds it back to the hidden
      state (residual connection).</li>
</ol>

<p>
After all 8 blocks, a final RMSNorm and a linear readout produce a probability
distribution over the ~65-character vocabulary.
</p>

<h3>5.2 NPU tile mapping</h3>

<p>
The model uses all 32 compute tiles in an 8-column &times; 4-row layout:
</p>

<pre>
Column 0        Column 1        ...  Column 7
(samples 0-47)  (samples 48-95)      (samples 336-383)

  Row 2: W&sub0;       Row 2: W&sub0;    ...    Row 2: W&sub0;      Stage 1
  &darr;              &darr;                  &darr;
  Row 3: W&sub1;       Row 3: W&sub1;    ...    Row 3: W&sub1;      Stage 2
  &darr;              &darr;                  &darr;
  Row 4: W&sub2;       Row 4: W&sub2;    ...    Row 4: W&sub2;      Stage 3
  &darr;              &darr;                  &darr;
  Row 5: W&sub3;       Row 5: W&sub3;    ...    Row 5: W&sub3;      Stage 4
</pre>

<ul>
  <li><strong>8 columns</strong> process 8 independent batch slices in parallel,
      48 samples each = 384 total.</li>
  <li><strong>4 rows</strong> per column form a pipeline: data flows from row 2
      to row 5 through <a href="#g-fifo" class="gref">ObjectFIFOs</a>,
      tile-to-tile, with no <a href="#g-ddr" class="gref">DDR</a> traffic
      between stages.</li>
  <li>All 8 columns share the same 4 weight matrices for a given block. The
      CPU loads the appropriate block&rsquo;s weights before each NPU call.</li>
</ul>

<h3>5.3 Why this architecture</h3>

<p>
Every architectural choice follows from a hardware constraint:
</p>

<ul>
  <li><strong>H=128:</strong> A 128&times;128 weight matrix plus a 128-element
      RMSNorm scale vector occupies 32.25 KB in
      <a href="#g-bf16" class="gref">bf16</a>. With two activation buffers
      (48&times;128 = 12 KB each, <a href="#v-doublebuf" class="gref">double-buffered</a>)
      and ~1.5 KB of stack+code, the total is ~58 KB &mdash; within the 64 KB per-tile
      <a href="#g-sram" class="gref">SRAM</a> limit.</li>
  <li><strong>4 layers per block:</strong> The NPU has 4 compute rows.
      A 4-stage pipeline uses one row per stage, fully utilizing the vertical
      dimension.</li>
  <li><strong>8 blocks:</strong> With 4 layers per block and 32 total layers,
      8&nbsp;blocks = 32&nbsp;layers = 32&nbsp;tiles. This is the maximum depth
      that maps cleanly to the hardware.</li>
  <li><strong>B=48:</strong> The fused <code>matmul_relu</code> kernel
      pipelines efficiently only when the outer loop has &ge;3 iterations:
      M/(2r) &ge; 3, so B &ge; 48 for r&nbsp;=&nbsp;8.
      (See <code>logbook.md</code> for the kernel fusion analysis.)</li>
  <li><strong>384 parallel sequences:</strong> 8 columns &times; 48 samples
      per column. Processing many sequences in parallel amortises the ~120 &mu;s
      <a href="#g-xrt" class="gref">XRT</a> driver overhead per NPU call.</li>
</ul>

<h3>5.4 Fusing normalisation into the pipeline</h3>

<p>
The key innovation is a custom NPU kernel that fuses three operations &mdash;
<a href="#g-rmsnorm" class="gref">RMSNorm</a>, matrix multiply, and
<a href="#g-relu" class="gref">ReLU</a> &mdash; into a single per-tile function.
Without this fusion, normalisation would require CPU intervention between every
pipeline stage, defeating the purpose of the 4-stage streaming pipeline.
</p>

<p>
The fused <code>norm_matmul_relu</code> kernel for each pipeline stage:
</p>

<ol>
  <li><strong>RMSNorm in-place</strong> (scalar float32 for stability): compute
      the root-mean-square of the input row, divide each element by it, and
      multiply by the learned scale vector. This is a reduction over 128 elements
      per row, using 8 Babylonian iterations for the inverse square root to avoid
      library dependencies.</li>
  <li><strong>Fused matmul + ReLU</strong> (vectorised bf16): the normalised
      input feeds directly into the same 2&times;2 tile-expanded matrix multiply
      used previously, with ReLU applied during the store phase.</li>
</ol>

<p>
The scale vector (128&nbsp;&times;&nbsp;2 bytes = 256 bytes) is packed at the end
of each tile&rsquo;s weight buffer: [W (32 KB), scale (256 B)].  This adds only
0.25 KB per tile, well within the SRAM budget.
</p>

<p>
Moving per-layer normalisation into the NPU pipeline closed the quality gap
from val&nbsp;loss 2.42 (pure matmul+ReLU blocks) to 2.03, approaching the
per-layer CPU-only model&rsquo;s 1.94 &mdash; while keeping the full 32-layer
inference on the NPU.
</p>

<div class="key-insight">
<strong>Design principle:</strong> We train the model with exactly the same
block structure that runs on the NPU. There is no &ldquo;compilation&rdquo; step
that approximates a richer model for hardware. What you train is what you deploy.
</div>

<h3>5.5 Making it trainable</h3>

<p>
Simply stacking 32 layers of ReLU(h&nbsp;&times;&nbsp;W&nbsp;+&nbsp;b) does not
work: the hidden state either explodes to infinity or collapses to zero. We use
three techniques (explained in
<a href="#s-nn-fundamentals">Section 2.7</a>):
</p>

<ul>
  <li><strong>Per-stage RMSNorm:</strong> Each pipeline stage normalises its
      input to unit RMS <em>before</em> the matrix multiply. Because the norm
      is fused into the NPU kernel, this happens on-chip with no CPU
      intervention.  This prevents the activations from growing exponentially
      through the 4-layer chain and across the 64-character sequence
      during <a href="#s-bptt">backpropagation through time</a>.</li>
  <li><strong>Residual connections:</strong> <code>h = h + block(h)</code>
      after each block. Gradients flow backwards through the identity shortcut,
      keeping training stable even at 32 layers.</li>
  <li><strong>Input injection:</strong> The character embedding is added at
      the start of every block, not just the first. This ensures each block
      can directly access the current input, rather than relying on information
      that has been progressively transformed by preceding blocks.</li>
  <li><strong>Spectral norm clamping:</strong> During training, each weight
      matrix is periodically rescaled so its largest singular value &le; 1.
      This prevents any single layer from amplifying the hidden state, keeping
      the recurrent dynamics stable.</li>
</ul>

<h2>6. Results</h2>

<h3>6.1 Language model quality</h3>

<p>
The block-recurrent model was trained on the tiny Shakespeare dataset (1.1 MB,
~1.1M characters) for 10 epochs on a GPU, then deployed to the NPU:
</p>

<table>
  <tr><th>Model</th><th>Parameters</th><th>Val Loss</th>
      <th>Perplexity</th><th>Device</th></tr>
  <tr style="background:#d4edda;font-weight:600;">
      <td>Block-recurrent (fused norm+mm+relu)</td>
      <td>542K</td><td>2.03</td><td>7.6</td><td>NPU (32 tiles)</td></tr>
  <tr><td>Per-layer recurrent (norm every layer)</td>
      <td>542K</td><td>1.94</td><td>6.9</td><td>CPU/GPU only</td></tr>
  <tr><td>Transformer baseline (GPT-style)</td>
      <td>818K</td><td>1.89</td><td>6.6</td><td>GPU</td></tr>
</table>

<p>
The fused norm+matmul+ReLU kernel closes the quality gap: the NPU model
(perplexity 7.6) approaches the per-layer CPU model (6.9) and the transformer
baseline (6.6), while running entirely on the 32-tile NPU pipeline.
The model produces recognisable Shakespearean English with a 542K-parameter
model trained on 1 MB of text.
</p>

<h3>6.2 NPU inference performance</h3>

<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>NPU tiles used</td><td>32 (8 columns &times; 4 rows)</td></tr>
  <tr><td>NPU calls per character</td><td>8 (one per block)</td></tr>
  <tr><td>Latency per NPU call</td><td>~0.14 ms</td></tr>
  <tr><td>Total NPU time per character</td><td>~1.1 ms</td></tr>
  <tr><td>Total time per character (NPU + CPU)</td><td>~4.2 ms</td></tr>
  <tr><td>Throughput (1 sequence)</td><td>233 chars/s</td></tr>
  <tr><td>Throughput (384 parallel sequences)</td><td><strong>89,600 chars/s</strong></td></tr>
</table>

<p>
The 384-sequence throughput is the key figure: by processing 8&nbsp;columns
&times; 48&nbsp;samples per column in parallel, we amortise the per-call
overhead across hundreds of sequences. The CPU overhead between NPU calls
(embedding injection, bias addition, residual connection) is minimal since
normalisation is now handled on-chip by the fused kernel.
</p>

<h3>6.3 What limits throughput</h3>

<p>
Each NPU call performs only 4 matmuls of size 48&times;128&times;128 &mdash;
a tiny amount of arithmetic completed in microseconds. The ~0.14 ms measured
per call is dominated by <a href="#g-xrt" class="gref">XRT</a> driver
overhead (instruction dispatch, <a href="#g-dma" class="gref">DMA</a> setup),
not compute. With 8 calls per character, that overhead adds up.
</p>

<p>
This is fundamentally different from the throughput benchmark (see
<code>logbook.md</code>), where a single NPU call ran 1000 loop iterations
on-chip, making compute dominate overhead and achieving 95.7% of the 25
<a href="#g-tflops" class="gref">TFLOPS</a> peak. The character LM trades
peak utilisation for a richer, more useful architecture.
</p>

<div class="key-insight">
<strong>Insight:</strong> NPU utilisation is not the goal &mdash; building a
useful model that runs on the NPU is. The architecture is designed for
correctness of the hardware mapping; throughput optimisation (larger hidden
dimensions, fewer blocks, quantisation) is future work.
</div>

<h2>7. The Toolchain</h2>

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

<h3>7.1 Compilation pipeline</h3>

<pre>
design.py  &xrarr;  MLIR  &xrarr;  aiecc  &xrarr;  .xclbin (bitstream)
                          &xrarr;  .bin   (instruction sequence)

norm_matmul_relu.cc &xrarr;  mlp_norm_mm_relu.o &xrarr;  mlp_kernels.a
mlp_kernels.cc     &xrarr;  mlp_copy.o         &nearr;
</pre>

<h2>8. Code Structure</h2>

<p>
The project is intentionally minimal &mdash; four Python files and two C++ files:
</p>

<table>
  <tr><th>File</th><th>Lines</th><th>Purpose</th></tr>
  <tr><td><code>char_lm/model.py</code></td><td>~210</td>
      <td>Block-recurrent character LM (RecurrentCharLM)</td></tr>
  <tr><td><code>char_lm/train.py</code></td><td>~250</td>
      <td>GPU training loop (ROCm / CUDA), spectral norm clamping</td></tr>
  <tr><td><code>char_lm/generate.py</code></td><td>~215</td>
      <td>Text generation on CPU or NPU</td></tr>
  <tr><td><code>char_lm/data.py</code></td><td>~80</td>
      <td>Shakespeare dataset and vocabulary</td></tr>
  <tr><td><code>char_lm/transformer_baseline.py</code></td><td>~240</td>
      <td>GPT-style reference model for quality comparison</td></tr>
  <tr><td><code>spatial_mlp/__init__.py</code></td><td>~55</td>
      <td>Tiling utilities (<code>to_tiled</code>, <code>from_tiled</code>)</td></tr>
  <tr><td><code>spatial_mlp/pipeline_design.py</code></td><td>~290</td>
      <td><a href="#g-iron" class="gref">IRON</a> design: 32-tile pipeline
          (8 cols &times; 4 rows)</td></tr>
  <tr><td><code>spatial_mlp/pipeline_op.py</code></td><td>~130</td>
      <td>IRON operator: compilation + runtime buffers</td></tr>
  <tr><td><code>aie_kernels/norm_matmul_relu.cc</code></td><td>~170</td>
      <td>Fused C&nbsp;=&nbsp;ReLU(RMSNorm(A,&nbsp;scale)&nbsp;&times;&nbsp;W)
          <a href="#v-kernel" class="gref">kernel</a> for
          <a href="#g-aie" class="gref">AIE2P</a></td></tr>
  <tr><td><code>aie_kernels/mlp_kernels.cc</code></td><td>~55</td>
      <td>Support kernel: <code>copy_bf16</code> for result staging</td></tr>
</table>

<p>
The character LM module (<code>char_lm/</code>) handles training and generation.
The spatial MLP module (<code>spatial_mlp/</code>) provides the NPU pipeline
infrastructure used by the generator for on-device inference.
</p>

<h2>9. Future Work</h2>

<ul>
  <li><strong><a href="#g-int8" class="gref">INT8</a> mode:</strong> The
      NPU&rsquo;s peak is 50 <a href="#g-tops" class="gref">TOPS</a> for
      int8 &mdash; double the bf16 rate. With quantization-aware training, this
      could push effective throughput to ~48 TOPS and allow H=256 (fitting in
      the same SRAM budget).</li>
  <li><strong>Vectorised RMSNorm:</strong> The current kernel uses scalar
      float32 for the normalisation reduction. A vectorised implementation
      using v16float SIMD could reduce the per-stage norm overhead from
      ~25&nbsp;&mu;s to ~3&nbsp;&mu;s, improving pipeline throughput
      when compute (not driver overhead) is the bottleneck.</li>
  <li><strong>Training on NPU:</strong> Research NPU backpropagation
      (see <a href="https://arxiv.org/html/2504.03083v1">arXiv:2504.03083</a>).
      The block structure could allow per-block gradient computation on-chip,
      with CPU handling the cross-block gradient flow.</li>
  <li><strong>Larger tasks:</strong> Apply the block-recurrent architecture to
      longer text datasets, music generation, or time-series forecasting where
      deep recurrence is natural and the NPU throughput advantage compounds
      over long sequences.</li>
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
