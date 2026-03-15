"""Compile and validate the logits-emitting streaming residual inference operator."""

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

from iron.common.aie_context import AIEContext

from resmlp import to_tiled
from resmlp.design import ROWS_PER_COL
from resmlp.streaming_logits_design import N_CLS_PADDED, NUM_CLASSES
from resmlp.streaming_logits_op import StreamingResMLPLogits


def reference_resmlp_logits(x, residual_weights, head_weight, head_bias):
    x_f32 = x.astype(np.float32)
    for w in residual_weights:
        w_f32 = w.astype(np.float32)
        matmul_out = (x_f32 @ w_f32).astype(bfloat16).astype(np.float32)
        relu_out = np.maximum(matmul_out, 0)
        x_f32 = (relu_out + x_f32).astype(bfloat16).astype(np.float32)
    logits = (x_f32 @ head_weight.astype(np.float32) + head_bias.astype(np.float32)).astype(bfloat16)
    return logits.astype(np.float32)


def compare(name, ref, got, rtol, atol):
    close = np.isclose(ref, got, rtol=rtol, atol=atol)
    pct = close.mean() * 100.0
    max_diff = np.max(np.abs(ref - got))
    status = "PASS" if pct > 95 else "FAIL"
    print(f"  [{status}] {name}: {pct:.1f}% close, max diff = {max_diff:.4f}")
    if pct <= 95:
        print(f"    ref [0,:5]: {ref[0, :5]}")
        print(f"    npu [0,:5]: {got[0, :5]}")
    return pct > 95


def run_test(H=160, B=8, cols=2, stream_depth=1, scale=0.01):
    rng = np.random.default_rng(123)
    num_tiles = cols * ROWS_PER_COL

    residual_weights = [
        (rng.standard_normal((H, H)).astype(np.float32) * scale).astype(bfloat16)
        for _ in range(num_tiles)
    ]
    head_weight = np.zeros((H, N_CLS_PADDED), dtype=np.float32)
    head_weight[:, :NUM_CLASSES] = rng.standard_normal((H, NUM_CLASSES)).astype(np.float32) * scale
    head_weight = head_weight.astype(bfloat16)
    head_bias = np.zeros((N_CLS_PADDED,), dtype=np.float32)
    head_bias[:NUM_CLASSES] = rng.standard_normal((NUM_CLASSES,)).astype(np.float32) * scale
    head_bias = head_bias.astype(bfloat16)

    inputs_round1 = [
        rng.standard_normal((B, H)).astype(np.float32).astype(bfloat16)
        for _ in range(stream_depth)
    ]
    inputs_round2 = [
        rng.standard_normal((B, H)).astype(np.float32).astype(bfloat16)
        for _ in range(stream_depth)
    ]

    refs_round1 = [reference_resmlp_logits(x, residual_weights, head_weight, head_bias) for x in inputs_round1]
    refs_round2 = [reference_resmlp_logits(x, residual_weights, head_weight, head_bias) for x in inputs_round2]

    packed_residual = np.stack([np.asarray(to_tiled(w), dtype=bfloat16) for w in residual_weights])
    packed_head_weight = np.asarray(to_tiled(head_weight), dtype=bfloat16)
    packed_head_bias = np.asarray(head_bias, dtype=bfloat16)

    ctx = AIEContext(use_runlist=False)
    op = StreamingResMLPLogits(
        packed_residual,
        packed_head_weight,
        packed_head_bias,
        H=H,
        B=B,
        num_cols=cols,
        stream_depth=stream_depth,
        context=ctx,
    )
    print(
        f"Compiling streaming logits operator ({num_tiles} tiles, B={B}, H={H}, "
        f"stream_depth={stream_depth})...",
        flush=True,
    )
    ctx.compile_all()
    ctx.prepare_runtime()

    ok = True
    for round_idx, (inputs_group, refs_group) in enumerate(
        ((inputs_round1, refs_round1), (inputs_round2, refs_round2)),
        start=1,
    ):
        op.write_buffer("input", np.concatenate([to_tiled(x) for x in inputs_group]))
        op.run_stream()
        logits_flat = op.read_buffer("output", (stream_depth * B * N_CLS_PADDED,), copy=True)

        for batch_idx, ref in enumerate(refs_group):
            start = batch_idx * B * N_CLS_PADDED
            stop = (batch_idx + 1) * B * N_CLS_PADDED
            got = logits_flat[start:stop].reshape(B, N_CLS_PADDED)[:, :NUM_CLASSES].astype(np.float32)
            ok &= compare(
                f"round {round_idx} batch {batch_idx}",
                ref[:, :NUM_CLASSES],
                got,
                rtol=0.30,
                atol=0.50,
            )

    return ok


def main():
    parser = argparse.ArgumentParser(description="Test resident streaming logits inference")
    parser.add_argument("--H", type=int, default=160)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--stream-depth", type=int, default=1)
    parser.add_argument("--scale", type=float, default=0.01)
    args = parser.parse_args()

    print(
        f"═══ Streaming logits inference test (cols={args.cols}, B={args.B}, "
        f"H={args.H}, stream_depth={args.stream_depth}) ═══\n"
    )
    ok = run_test(
        H=args.H,
        B=args.B,
        cols=args.cols,
        stream_depth=args.stream_depth,
        scale=args.scale,
    )
    print(f"\n{'═' * 50}")
    print("ALL PASSED" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
