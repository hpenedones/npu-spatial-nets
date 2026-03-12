"""
Compile and validate the 4-tile forward checkpoint probe.
"""

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

from iron.common.aie_context import AIEContext

from resmlp import from_tiled, to_tiled
from resmlp.forward_checkpoint_op import ForwardCheckpointColumn, ROWS_PER_COL


def reference_forward_with_checkpoints(x, weights):
    current = x.astype(np.float32)
    checkpoints = []
    for w in weights:
        checkpoints.append(current.astype(bfloat16))
        matmul_out = (current @ w.astype(np.float32)).astype(bfloat16).astype(np.float32)
        current = (np.maximum(matmul_out, 0) + current).astype(bfloat16).astype(np.float32)
    return checkpoints, current.astype(bfloat16)


def compare(name, ref, got, rtol, atol):
    ref_f32 = ref.astype(np.float32)
    got_f32 = got.astype(np.float32)
    close = np.isclose(ref_f32, got_f32, rtol=rtol, atol=atol)
    pct = close.mean() * 100.0
    max_diff = np.max(np.abs(ref_f32 - got_f32))
    status = "PASS" if pct > 95 else "FAIL"
    print(f"  [{status}] {name}: {pct:.1f}% close, max diff = {max_diff:.4f}")
    print(f"    ref [0,:5]: {ref_f32.reshape(ref.shape)[0, :5]}")
    print(f"    npu [0,:5]: {got_f32.reshape(got.shape)[0, :5]}")
    return pct > 95


def run_test(H=160, B=8, scale=0.05):
    rng = np.random.default_rng(11)
    x = rng.standard_normal((B, H)).astype(np.float32).astype(bfloat16)
    weights = [
        (rng.standard_normal((H, H)).astype(np.float32) * scale).astype(bfloat16)
        for _ in range(ROWS_PER_COL)
    ]

    checkpoints_ref, y_ref = reference_forward_with_checkpoints(x, weights)

    ctx = AIEContext()
    op = ForwardCheckpointColumn(H=H, B=B, context=ctx)
    print(f"Compiling checkpoint probe (4 tiles, B={B}, H={H})...", flush=True)
    ctx.compile_all()
    ctx.prepare_runtime()

    op.write_buffer("input", to_tiled(x))
    op.write_buffer("weights", np.concatenate([to_tiled(w) for w in weights]))
    op.write_buffer("checkpoints", np.zeros(ROWS_PER_COL * B * H, dtype=bfloat16))
    op.write_buffer("output", np.zeros(B * H, dtype=bfloat16))
    op.run_runlist()

    y_npu = from_tiled(op.read_buffer("output", (B * H,), copy=True), B, H)
    ckpt_flat = op.read_buffer("checkpoints", (ROWS_PER_COL * B * H,), copy=True)
    checkpoints_npu = [
        from_tiled(ckpt_flat[i * B * H : (i + 1) * B * H], B, H)
        for i in range(ROWS_PER_COL)
    ]

    print("\nCheckpoint probe checks:")
    ok = compare("final output", y_ref, y_npu, rtol=0.10, atol=0.10)
    for i, (ref, got) in enumerate(zip(checkpoints_ref, checkpoints_npu)):
        ok &= compare(f"checkpoint x_{i}", ref, got, rtol=0.0, atol=0.0)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Test forward checkpoint probe")
    parser.add_argument("--H", type=int, default=160)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.05)
    args = parser.parse_args()

    print(f"═══ Forward checkpoint probe (B={args.B}, H={args.H}) ═══\n")
    ok = run_test(H=args.H, B=args.B, scale=args.scale)
    print(f"\n{'═' * 50}")
    print("ALL PASSED" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
