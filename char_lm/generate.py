#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate text using the trained character LM — on CPU or NPU.

Usage::

    # CPU inference (baseline)
    python -m char_lm.generate --device cpu --prompt "To be"

    # NPU inference — recurrent design (same W on all tiles, deep loop)
    python -m char_lm.generate --device npu --prompt "To be"

    # NPU inference — pipeline design (different W per tile, 32 stages)
    python -m char_lm.generate --device npu-pipeline --prompt "To be"

The **recurrent** NPU path broadcasts one W to 24 tiles and loops many
iterations on-chip (good for deep single-weight models).

The **pipeline** NPU path uses all 32 tiles with different weights —
4 stages per column, 8 columns in parallel.  For a 32-layer model,
makes 8 NPU calls per character (4 layers each), achieving the same
total depth with 32× more parameters.
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from ml_dtypes import bfloat16

from char_lm.train import load_checkpoint
from char_lm.data import Vocabulary

DATA_DIR = Path(__file__).parent.parent / "data"

# ── NPU configuration ────────────────────────────────────────────────────
H = 128
B = 48

# Recurrent design (legacy)
NUM_TILES = 24
TOTAL_SAMPLES_RECURRENT = NUM_TILES * B  # 1152 parallel sequences

# Pipeline design
NUM_COLS = 8
STAGES_PER_COL = 4
TOTAL_SAMPLES_PIPELINE = NUM_COLS * B    # 384 parallel sequences


def _generate_cpu(
    model, vocab: Vocabulary, prompt: str, num_chars: int, temperature: float
) -> tuple[str, float]:
    """Generate text using pure CPU inference. Returns (text, elapsed_s)."""
    model.eval()
    prompt_ids = torch.tensor([vocab.encode(prompt)])

    t0 = time.perf_counter()
    generated = model.generate(
        prompt_ids, num_chars=num_chars, temperature=temperature
    )
    elapsed = time.perf_counter() - t0

    return vocab.decode(generated), elapsed


def _ensure_iron_env():
    """Set up IRON environment: chdir to IRON root, source XRT."""
    iron_dir = os.environ.get(
        "IRON_DIR", str(Path.home() / "source" / "IRON")
    )
    os.chdir(iron_dir)

    xrt_setup = "/opt/xilinx/xrt/setup.sh"
    if "XILINX_XRT" not in os.environ and Path(xrt_setup).exists():
        import subprocess
        env = subprocess.check_output(
            f"source {xrt_setup} && env", shell=True, text=True,
            executable="/bin/bash"
        )
        for line in env.strip().split("\n"):
            if "=" in line:
                k, _, v = line.partition("=")
                os.environ[k] = v


def _setup_npu_recurrent(num_iters: int):
    """Compile recurrent NPU operator (same W on all tiles)."""
    _ensure_iron_env()
    from iron.common.aie_context import AIEContext
    from spatial_mlp.op import AIERecurrentMLP

    ctx = AIEContext()
    op = AIERecurrentMLP(
        H=H, B=B, num_tiles=NUM_TILES, num_iters=num_iters, context=ctx
    )
    print("Compiling recurrent NPU design (cached if unchanged)...")
    ctx.compile_all()
    ctx.prepare_runtime()
    print("NPU ready (recurrent).")
    return op


def _setup_npu_pipeline():
    """Compile pipeline NPU operator (different W per tile, 4 stages/col)."""
    _ensure_iron_env()
    from iron.common.aie_context import AIEContext
    from spatial_mlp.pipeline_op import AIEPipelineMLP

    ctx = AIEContext()
    op = AIEPipelineMLP(H=H, B=B, num_cols=NUM_COLS, context=ctx)
    print("Compiling pipeline NPU design (cached if unchanged)...")
    ctx.compile_all()
    ctx.prepare_runtime()
    print("NPU ready (pipeline, 32 tiles).")
    return op


def _generate_npu_recurrent(
    model, vocab: Vocabulary, prompt: str, num_chars: int,
    temperature: float, npu_op
) -> tuple[str, float, float]:
    """Generate text using recurrent NPU (same W broadcast to all tiles).

    For multi-layer models, makes num_layers NPU calls per character,
    each with a different weight matrix.

    Returns (text, total_elapsed_s, npu_compute_s).
    """
    from spatial_mlp import to_tiled, from_tiled

    model.eval()
    total_samples = TOTAL_SAMPLES_RECURRENT

    W_tiled_list = []
    for W in model.weights:
        W_bf16 = W.data.numpy().astype(bfloat16)
        W_tiled_list.append(to_tiled(W_bf16))

    prompt_ids = vocab.encode(prompt)
    hidden = np.zeros((total_samples, H), dtype=bfloat16)
    embed_w = model.embed.weight.data.numpy().astype(bfloat16)
    readout_w = model.readout.weight.data.numpy().astype(np.float32)
    readout_b = model.readout.bias.data.numpy().astype(np.float32)
    output_zeros = np.zeros(total_samples * H, dtype=bfloat16)

    generated = list(prompt_ids)
    npu_total = 0.0
    t0 = time.perf_counter()

    def _run_recurrence(hidden):
        npu_time = 0.0
        for W_tiled in W_tiled_list:
            input_tiled = np.concatenate([
                to_tiled(hidden[i * B : (i + 1) * B])
                for i in range(NUM_TILES)
            ])
            npu_op.write_buffer("input", input_tiled)
            npu_op.write_buffer("weights", W_tiled)
            npu_op.write_buffer("output", output_zeros.copy())
            npu_time += npu_op.run_runlist()
            out_flat = npu_op.read_buffer(
                "output", (total_samples * H,), copy=True
            )
            hidden = np.concatenate([
                from_tiled(out_flat[i * B * H : (i + 1) * B * H], B, H)
                for i in range(NUM_TILES)
            ])
        return hidden, npu_time

    for char_idx in prompt_ids:
        hidden = hidden + embed_w[char_idx][np.newaxis, :]
        hidden, dt = _run_recurrence(hidden)
        npu_total += dt

    for step in range(num_chars):
        h_f32 = hidden[0].astype(np.float32)
        logits = (h_f32 @ readout_w.T + readout_b) / temperature
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        next_idx = np.random.choice(len(probs), p=probs)
        generated.append(next_idx)
        hidden = hidden + embed_w[next_idx][np.newaxis, :]
        hidden, dt = _run_recurrence(hidden)
        npu_total += dt

    total_elapsed = time.perf_counter() - t0
    return vocab.decode(generated), total_elapsed, npu_total


def _generate_npu_pipeline(
    model, vocab: Vocabulary, prompt: str, num_chars: int,
    temperature: float, npu_op
) -> tuple[str, float, float]:
    """Generate text using pipeline NPU (different W per tile, 32 stages).

    The model must have num_layers that is a multiple of STAGES_PER_COL (4).
    Each NPU call processes 4 layers through one column pipeline.
    For 32 layers: 8 NPU calls per character step.

    Returns (text, total_elapsed_s, npu_compute_s).
    """
    from spatial_mlp import to_tiled, from_tiled

    model.eval()
    num_layers = model.num_layers
    total_samples = TOTAL_SAMPLES_PIPELINE  # 384
    assert num_layers % STAGES_PER_COL == 0, (
        f"num_layers={num_layers} must be divisible by {STAGES_PER_COL}")
    num_npu_calls = num_layers // STAGES_PER_COL

    # Pre-tile all weight groups (each group = 4 consecutive W matrices)
    # Host layout: [col0_W0, col0_W1, col0_W2, col0_W3, col1_W0, ...]
    # All 8 columns get the SAME 4 weights per call
    W_tiled_groups = []
    for call_idx in range(num_npu_calls):
        parts = []
        for col in range(NUM_COLS):
            for stage in range(STAGES_PER_COL):
                layer_idx = call_idx * STAGES_PER_COL + stage
                W_bf16 = model.weights[layer_idx].data.numpy().astype(bfloat16)
                parts.append(to_tiled(W_bf16))
        W_tiled_groups.append(np.concatenate(parts))

    prompt_ids = vocab.encode(prompt)
    hidden = np.zeros((total_samples, H), dtype=bfloat16)
    embed_w = model.embed.weight.data.numpy().astype(bfloat16)
    readout_w = model.readout.weight.data.numpy().astype(np.float32)
    readout_b = model.readout.bias.data.numpy().astype(np.float32)
    output_zeros = np.zeros(total_samples * H, dtype=bfloat16)

    generated = list(prompt_ids)
    npu_total = 0.0
    t0 = time.perf_counter()

    def _run_pipeline(hidden):
        """Run all layers via pipeline NPU calls."""
        npu_time = 0.0
        for W_tiled in W_tiled_groups:
            # Tile activations: [col0_batch, col1_batch, ...]
            input_tiled = np.concatenate([
                to_tiled(hidden[c * B : (c + 1) * B])
                for c in range(NUM_COLS)
            ])
            npu_op.write_buffer("input", input_tiled)
            npu_op.write_buffer("weights", W_tiled)
            npu_op.write_buffer("output", output_zeros.copy())
            npu_time += npu_op.run_runlist()
            out_flat = npu_op.read_buffer(
                "output", (total_samples * H,), copy=True
            )
            hidden = np.concatenate([
                from_tiled(out_flat[c * B * H : (c + 1) * B * H], B, H)
                for c in range(NUM_COLS)
            ])
        return hidden, npu_time

    for char_idx in prompt_ids:
        hidden = hidden + embed_w[char_idx][np.newaxis, :]
        hidden, dt = _run_pipeline(hidden)
        npu_total += dt

    for step in range(num_chars):
        h_f32 = hidden[0].astype(np.float32)
        logits = (h_f32 @ readout_w.T + readout_b) / temperature
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        next_idx = np.random.choice(len(probs), p=probs)
        generated.append(next_idx)
        hidden = hidden + embed_w[next_idx][np.newaxis, :]
        hidden, dt = _run_pipeline(hidden)
        npu_total += dt

    total_elapsed = time.perf_counter() - t0
    return vocab.decode(generated), total_elapsed, npu_total


def main():
    parser = argparse.ArgumentParser(description="Generate text with char LM")
    parser.add_argument("--device",
                        choices=["cpu", "npu", "npu-pipeline"],
                        default="cpu",
                        help="Inference device (default: cpu)")
    parser.add_argument("--prompt", type=str, default="\nTo be, or not to be",
                        help="Text prompt to start generation")
    parser.add_argument("--num-chars", type=int, default=200,
                        help="Number of characters to generate (default: 200)")
    parser.add_argument("--depth", type=int, default=None,
                        help="Override recurrence depth (default: use checkpoint value)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--checkpoint", type=str,
                        default=str(DATA_DIR / "charlm_checkpoint.pt"),
                        help="Path to model checkpoint")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        print("Run `python -m char_lm.train` first.")
        sys.exit(1)

    model, vocab = load_checkpoint(ckpt_path)

    if args.depth is not None:
        model.depth = args.depth
        model.bptt_depth = min(model.bptt_depth, args.depth)

    params = model.count_parameters()
    device_label = args.device.upper().replace("-", " ")

    print("=" * 60)
    print(f"TileFlow Character LM — Generate ({device_label})")
    print("=" * 60)
    print(f"Parameters: {params['total']:,} "
          f"(W: {params['recurrent_W']:,} in {params['num_layers']} layer(s), "
          f"depth: {model.depth})")
    print(f"Prompt: {repr(args.prompt)}")
    print(f"Generating {args.num_chars} characters...")
    print()

    if args.device == "cpu":
        text, elapsed = _generate_cpu(
            model, vocab, args.prompt, args.num_chars, args.temperature
        )
        chars_per_sec = args.num_chars / elapsed
        print("─" * 60)
        print(text)
        print("─" * 60)
        print(f"\nCPU: {elapsed:.3f}s total, "
              f"{chars_per_sec:.0f} chars/s, "
              f"depth={model.depth}")

    elif args.device == "npu":
        num_iters = model.depth_per_layer // 2
        npu_op = _setup_npu_recurrent(num_iters)
        text, total_elapsed, npu_time = _generate_npu_recurrent(
            model, vocab, args.prompt, args.num_chars, args.temperature, npu_op
        )
        total_steps = len(args.prompt) + args.num_chars
        chars_per_sec = args.num_chars / total_elapsed
        total_samples = TOTAL_SAMPLES_RECURRENT
        print("─" * 60)
        print(text)
        print("─" * 60)
        print(f"\nNPU (recurrent): {total_elapsed:.3f}s total, "
              f"NPU compute: {npu_time:.3f}s "
              f"({npu_time/total_elapsed*100:.0f}%), "
              f"{chars_per_sec:.0f} chars/s (1 sequence)")
        print(f"NPU per step: {npu_time/total_steps*1000:.2f}ms, "
              f"overhead: {(total_elapsed-npu_time)/total_steps*1000:.2f}ms")
        print(f"Throughput: {total_samples} sequences × {args.num_chars} chars "
              f"= {total_samples * chars_per_sec:.0f} total chars/s")

    elif args.device == "npu-pipeline":
        npu_op = _setup_npu_pipeline()
        text, total_elapsed, npu_time = _generate_npu_pipeline(
            model, vocab, args.prompt, args.num_chars, args.temperature, npu_op
        )
        total_steps = len(args.prompt) + args.num_chars
        chars_per_sec = args.num_chars / total_elapsed
        total_samples = TOTAL_SAMPLES_PIPELINE
        num_calls = model.num_layers // STAGES_PER_COL
        print("─" * 60)
        print(text)
        print("─" * 60)
        print(f"\nNPU (pipeline, 32 tiles): {total_elapsed:.3f}s total, "
              f"NPU compute: {npu_time:.3f}s "
              f"({npu_time/total_elapsed*100:.0f}%), "
              f"{chars_per_sec:.0f} chars/s (1 sequence)")
        print(f"NPU per step: {npu_time/total_steps*1000:.2f}ms "
              f"({num_calls} calls × "
              f"{npu_time/total_steps/num_calls*1000:.2f}ms), "
              f"overhead: {(total_elapsed-npu_time)/total_steps*1000:.2f}ms")
        print(f"Throughput: {total_samples} sequences × {args.num_chars} chars "
              f"= {total_samples * chars_per_sec:.0f} total chars/s")


if __name__ == "__main__":
    main()
