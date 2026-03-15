import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

from iron.common.aie_context import AIEContext

from resmlp import from_tiled, to_tiled
from resmlp.design import ROWS_PER_COL
from resmlp.streaming_embed_op import StreamingEmbedResMLP
from resmlp.streaming_embed_infer import reference_embed_resmlp


def compare(name, ref, got, rtol, atol):
    ref_f32 = ref.astype(np.float32)
    got_f32 = got.astype(np.float32)
    close = np.isclose(ref_f32, got_f32, rtol=rtol, atol=atol)
    pct = close.mean() * 100.0
    max_diff = np.max(np.abs(ref_f32 - got_f32))
    status = "PASS" if pct > 95 else "FAIL"
    print(f"  [{status}] {name}: {pct:.1f}% close, max diff = {max_diff:.4f}")
    if pct <= 95:
        print(f"    ref [0,:5]: {ref_f32.reshape(ref.shape)[0, :5]}")
        print(f"    npu [0,:5]: {got_f32.reshape(got.shape)[0, :5]}")
    return pct > 95


def run_test(H=32, B=8, cols=2, stream_depth=1, scale=0.01):
    rng = np.random.default_rng(123)
    num_tiles = cols * ROWS_PER_COL
    num_layers = num_tiles - 1

    embed_weight = (
        rng.standard_normal((784, H)).astype(np.float32) * scale
    ).astype(bfloat16)
    residual_weights = [
        (rng.standard_normal((H, H)).astype(np.float32) * scale).astype(bfloat16)
        for _ in range(num_layers)
    ]
    inputs_round1 = [
        rng.standard_normal((B, 784)).astype(np.float32).astype(bfloat16)
        for _ in range(stream_depth)
    ]
    inputs_round2 = [
        rng.standard_normal((B, 784)).astype(np.float32).astype(bfloat16)
        for _ in range(stream_depth)
    ]

    refs_round1 = [reference_embed_resmlp(x, embed_weight, residual_weights) for x in inputs_round1]
    refs_round2 = [reference_embed_resmlp(x, embed_weight, residual_weights) for x in inputs_round2]

    packed_embed = np.asarray(to_tiled(embed_weight), dtype=bfloat16)
    packed_residual = np.stack([np.asarray(to_tiled(w), dtype=bfloat16) for w in residual_weights])

    ctx = AIEContext(use_runlist=False)
    op = StreamingEmbedResMLP(
        packed_embed,
        packed_residual,
        H=H,
        B=B,
        num_cols=cols,
        stream_depth=stream_depth,
        context=ctx,
    )
    print(
        f"Compiling embed-on-NPU streaming operator ({num_tiles} tiles, B={B}, H={H}, "
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
        y_flat = op.read_buffer("output", (stream_depth * B * H,), copy=True)

        for batch_idx, ref in enumerate(refs_group):
            start = batch_idx * B * H
            stop = (batch_idx + 1) * B * H
            got = from_tiled(y_flat[start:stop], B, H)
            ok &= compare(
                f"round {round_idx} batch {batch_idx}",
                ref,
                got,
                rtol=0.30,
                atol=0.50,
            )

    return ok


def main():
    parser = argparse.ArgumentParser(description="Test resident streaming embed-on-NPU inference")
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--stream-depth", type=int, default=1)
    parser.add_argument("--scale", type=float, default=0.01)
    args = parser.parse_args()

    print(
        f"═══ Streaming embed-on-NPU test (cols={args.cols}, B={args.B}, "
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
