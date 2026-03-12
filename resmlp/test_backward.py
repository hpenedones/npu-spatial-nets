"""
Compile and validate the single-layer residual backward operators.

The NPU phase-0 path uses two operators:

    1. residual_grad_input_bf16  : gx = gy + (gy * mask) @ W^T
    2. residual_weight_grad_bf16 : dW = x^T @ (gy * mask)

The reference path uses bf16-style intermediate casts to match the NPU's
numerics reasonably closely.
"""

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

from iron.common.aie_context import AIEContext

from resmlp import from_tiled, to_tiled
from resmlp.backward_single_op import ResidualBackwardSingle


def reference_backward(x, w, gy):
    """Approximate the NPU backward pass with bf16 intermediate casts."""
    x_f32 = x.astype(np.float32)
    w_f32 = w.astype(np.float32)
    gy_f32 = gy.astype(np.float32)

    z = (x_f32 @ w_f32).astype(bfloat16).astype(np.float32)
    mask = (z > 0).astype(np.float32)
    gz = (gy_f32 * mask).astype(bfloat16).astype(np.float32)
    dw = (x_f32.T @ gz).astype(bfloat16).astype(np.float32)
    gx_mm = (gz @ w_f32.T).astype(bfloat16).astype(np.float32)
    gx = (gx_mm + gy_f32).astype(bfloat16).astype(np.float32)
    return mask.astype(bfloat16), gx.astype(bfloat16), dw.astype(bfloat16)


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
    rng = np.random.default_rng(7)
    x = rng.standard_normal((B, H)).astype(np.float32).astype(bfloat16)
    w = (rng.standard_normal((H, H)).astype(np.float32) * scale).astype(bfloat16)
    gy = rng.standard_normal((B, H)).astype(np.float32).astype(bfloat16)

    mask, gx_ref, dw_ref = reference_backward(x, w, gy)

    ctx = AIEContext()
    gx_op = ResidualBackwardSingle(H=H, B=B, mode="grad_input", context=ctx)
    dw_op = ResidualBackwardSingle(H=H, B=B, mode="weight_grad", context=ctx)
    print(f"Compiling backward kernels (B={B}, H={H})...", flush=True)
    ctx.compile_all()
    ctx.prepare_runtime()

    gx_op.write_buffer(
        "state", np.concatenate([to_tiled(gy), to_tiled(mask)])
    )
    gx_op.write_buffer("weights_t", to_tiled(w.T))
    gx_op.write_buffer("grad_in", np.zeros(B * H, dtype=bfloat16))
    gx_op.run_runlist()
    gx_npu = from_tiled(
        gx_op.read_buffer("grad_in", (B * H,), copy=True), B, H
    )

    dw_op.write_buffer(
        "state", np.concatenate([to_tiled(x), to_tiled(gy), to_tiled(mask)])
    )
    dw_op.write_buffer("dweights", np.zeros(H * H, dtype=bfloat16))
    dw_op.run_runlist()
    dw_npu = from_tiled(
        dw_op.read_buffer("dweights", (H * H,), copy=True), H, H
    )

    print("\nBackward checks:")
    ok_gx = compare("grad input", gx_ref, gx_npu, rtol=0.20, atol=0.20)
    ok_dw = compare("weight grad", dw_ref, dw_npu, rtol=0.20, atol=0.20)
    return ok_gx and ok_dw


def main():
    parser = argparse.ArgumentParser(description="Test residual backward kernel")
    parser.add_argument("--H", type=int, default=160)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.05)
    args = parser.parse_args()

    print(f"═══ Residual backward test (B={args.B}, H={args.H}) ═══\n")
    ok = run_test(H=args.H, B=args.B, scale=args.scale)
    print(f"\n{'═' * 50}")
    print("ALL PASSED" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
