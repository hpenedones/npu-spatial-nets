"""
Compile and validate the resident streaming residual inference operator.

The key property under test is not "one giant window", but a host-facing
continuous service shape:

  - compile once
  - embed weights once in the xclbin
  - run repeated host calls
  - feed new inputs each call
  - read outputs back correctly each call
"""

import argparse
import sys

import numpy as np
import pytest
from ml_dtypes import bfloat16

from resmlp.xrt_env import ensure_xrt_python_path

ensure_xrt_python_path()
pytest.importorskip("pyxrt")

from iron.common.aie_context import AIEContext

from resmlp import from_tiled, to_tiled
from resmlp.design import ROWS_PER_COL
from resmlp.streaming_op import StreamingResMLP


def reference_resmlp(x, weights):
    x_f32 = x.astype(np.float32)
    for w in weights:
        w_f32 = w.astype(np.float32)
        matmul_out = (x_f32 @ w_f32).astype(bfloat16).astype(np.float32)
        relu_out = np.maximum(matmul_out, 0)
        x_f32 = (relu_out + x_f32).astype(bfloat16).astype(np.float32)
    return x_f32.astype(bfloat16)


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

    weights = [
        (rng.standard_normal((H, H)).astype(np.float32) * scale).astype(bfloat16)
        for _ in range(num_tiles)
    ]
    inputs_round1 = [
        rng.standard_normal((B, H)).astype(np.float32).astype(bfloat16)
        for _ in range(stream_depth)
    ]
    inputs_round2 = [
        rng.standard_normal((B, H)).astype(np.float32).astype(bfloat16)
        for _ in range(stream_depth)
    ]

    refs_round1 = [reference_resmlp(x, weights) for x in inputs_round1]
    refs_round2 = [reference_resmlp(x, weights) for x in inputs_round2]

    packed_weights = np.stack([np.asarray(to_tiled(w), dtype=bfloat16) for w in weights])

    ctx = AIEContext(use_runlist=False)
    op = StreamingResMLP(
        packed_weights,
        H=H,
        B=B,
        num_cols=cols,
        stream_depth=stream_depth,
        context=ctx,
    )
    print(
        f"Compiling streaming inference operator ({num_tiles} tiles, B={B}, H={H}, "
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
        op.write_input_slot(0, np.concatenate([to_tiled(x) for x in inputs_group]))
        op.run_stream(slot=0)
        y_flat = op.read_output_slot(0, (stream_depth * B * H,), copy=True)

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


def test_streaming_inference_smoke():
    assert run_test(H=32, B=8, cols=2, stream_depth=1, scale=0.01)


def main():
    parser = argparse.ArgumentParser(description="Test resident streaming inference")
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--stream-depth", type=int, default=1)
    parser.add_argument("--scale", type=float, default=0.01)
    args = parser.parse_args()

    print(
        f"═══ Streaming inference test (cols={args.cols}, B={args.B}, "
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
