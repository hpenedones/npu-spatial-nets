#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate text using the trained character LM — on CPU or NPU.

Usage::

    # CPU inference (baseline)
    python -m char_lm.generate --device cpu --prompt "To be"

    # NPU inference (requires compiled design)
    python -m char_lm.generate --device npu --prompt "To be"

The NPU path loads the trained weight matrix W onto all 24 tiles and runs
the recurrent core (h = ReLU(h @ W), depth iterations) in hardware,
achieving ~26× speedup over CPU for the matmul-dominated inner loop.
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
NUM_TILES = 24
TOTAL_SAMPLES = NUM_TILES * B  # 1152 parallel sequences


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


def _setup_npu(num_iters: int):
    """Compile and prepare the NPU operator. Returns the IRON operator."""
    # IRON requires the working directory to be the IRON repo root
    iron_dir = os.environ.get(
        "IRON_DIR", str(Path.home() / "source" / "IRON")
    )
    os.chdir(iron_dir)

    # Source XRT setup if not already done
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

    from iron.common.aie_context import AIEContext
    from spatial_mlp.op import AIERecurrentMLP

    ctx = AIEContext()
    op = AIERecurrentMLP(
        H=H, B=B, num_tiles=NUM_TILES, num_iters=num_iters, context=ctx
    )
    print("Compiling NPU design (cached if unchanged)...")
    ctx.compile_all()
    ctx.prepare_runtime()
    print("NPU ready.")
    return op


def _generate_npu(
    model, vocab: Vocabulary, prompt: str, num_chars: int,
    temperature: float, npu_op
) -> tuple[str, float, float]:
    """Generate text using NPU for the recurrent core.

    Returns (text, total_elapsed_s, npu_compute_s).
    """
    from spatial_mlp import to_tiled, from_tiled

    model.eval()
    depth = model.depth

    # Extract weights and convert to tiled bf16 layout
    W_f32 = model.W.data.numpy()
    W_bf16 = W_f32.astype(bfloat16)
    W_tiled = to_tiled(W_bf16)

    # We generate TOTAL_SAMPLES sequences in parallel, using the same prompt
    prompt_ids = vocab.encode(prompt)
    hidden = np.zeros((TOTAL_SAMPLES, H), dtype=bfloat16)
    embed_w = model.embed.weight.data.numpy().astype(bfloat16)
    readout_w = model.readout.weight.data.numpy().astype(np.float32)
    readout_b = model.readout.bias.data.numpy().astype(np.float32)

    generated = list(prompt_ids)
    npu_total = 0.0
    t0 = time.perf_counter()

    # Process prompt through the model
    for char_idx in prompt_ids:
        emb = embed_w[char_idx]  # (H,)
        hidden = hidden + emb[np.newaxis, :]  # broadcast add

        # Send to NPU: convert to tiled layout
        input_tiled = np.concatenate([
            to_tiled(hidden[i * B : (i + 1) * B])
            for i in range(NUM_TILES)
        ])
        npu_op.write_buffer("input", input_tiled)
        npu_op.write_buffer("weights", W_tiled)
        npu_op.write_buffer(
            "output", np.zeros(TOTAL_SAMPLES * H, dtype=bfloat16)
        )

        npu_elapsed = npu_op.run_runlist()
        npu_total += npu_elapsed

        # Read back
        out_flat = npu_op.read_buffer(
            "output", (TOTAL_SAMPLES * H,), copy=True
        )
        hidden = np.concatenate([
            from_tiled(out_flat[i * B * H : (i + 1) * B * H], B, H)
            for i in range(NUM_TILES)
        ])

    # Autoregressive generation
    for step in range(num_chars):
        # Compute logits on CPU from the first sequence
        h_f32 = hidden[0].astype(np.float32)
        logits = h_f32 @ readout_w.T + readout_b
        logits = logits / temperature

        # Sample
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        next_idx = np.random.choice(len(probs), p=probs)
        generated.append(next_idx)

        # Inject new character into all sequences
        emb = embed_w[next_idx]
        hidden = hidden + emb[np.newaxis, :]

        # NPU recurrence
        input_tiled = np.concatenate([
            to_tiled(hidden[i * B : (i + 1) * B])
            for i in range(NUM_TILES)
        ])
        npu_op.write_buffer("input", input_tiled)
        npu_op.write_buffer("weights", W_tiled)
        npu_op.write_buffer(
            "output", np.zeros(TOTAL_SAMPLES * H, dtype=bfloat16)
        )

        npu_elapsed = npu_op.run_runlist()
        npu_total += npu_elapsed

        out_flat = npu_op.read_buffer(
            "output", (TOTAL_SAMPLES * H,), copy=True
        )
        hidden = np.concatenate([
            from_tiled(out_flat[i * B * H : (i + 1) * B * H], B, H)
            for i in range(NUM_TILES)
        ])

    total_elapsed = time.perf_counter() - t0
    return vocab.decode(generated), total_elapsed, npu_total


def main():
    parser = argparse.ArgumentParser(description="Generate text with char LM")
    parser.add_argument("--device", choices=["cpu", "npu"], default="cpu",
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

    # Load model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        print("Run `python -m char_lm.train` first.")
        sys.exit(1)

    model, vocab = load_checkpoint(ckpt_path)

    # Override depth if requested
    if args.depth is not None:
        model.depth = args.depth
        model.bptt_depth = min(model.bptt_depth, args.depth)

    params = model.count_parameters()

    print("=" * 60)
    print(f"TileFlow Character LM — Generate ({args.device.upper()})")
    print("=" * 60)
    print(f"Parameters: {params['total']:,} "
          f"(W: {params['recurrent_W']:,}, depth: {model.depth})")
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
    else:
        num_iters = model.depth // 2
        npu_op = _setup_npu(num_iters)
        text, total_elapsed, npu_time = _generate_npu(
            model, vocab, args.prompt, args.num_chars, args.temperature, npu_op
        )
        total_steps = len(args.prompt) + args.num_chars
        chars_per_sec = args.num_chars / total_elapsed
        print("─" * 60)
        print(text)
        print("─" * 60)
        print(f"\nNPU: {total_elapsed:.3f}s total, "
              f"NPU compute: {npu_time:.3f}s "
              f"({npu_time/total_elapsed*100:.0f}%), "
              f"{chars_per_sec:.0f} chars/s (1 sequence)")
        print(f"NPU per step: {npu_time/total_steps*1000:.2f}ms, "
              f"overhead: {(total_elapsed-npu_time)/total_steps*1000:.2f}ms")
        print(f"Throughput: {TOTAL_SAMPLES} sequences × {args.num_chars} chars "
              f"= {TOTAL_SAMPLES * chars_per_sec:.0f} total chars/s")


if __name__ == "__main__":
    main()
