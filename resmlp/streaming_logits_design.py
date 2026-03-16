"""Streaming residual inference that emits logits directly from the last tile."""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern

from resmlp.design import ROWS_PER_COL, snake_tile_order

NUM_CLASSES = 10
N_CLS_PADDED = 16


def padded_class_dim(num_classes: int) -> int:
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    return ((num_classes + 7) // 8) * 8


def _restore_bfloat16(array, shape):
    raw = np.asarray(array)
    if raw.itemsize != 2:
        raise ValueError(f"expected 2-byte packed weights, got dtype={raw.dtype}")
    return raw.view(bfloat16).reshape(shape)


def _load_embedded_weights(weights_path: str | Path, num_tiles: int, H: int, n_cls_padded: int):
    data = np.load(Path(weights_path), allow_pickle=False)
    return {
        "residual": _restore_bfloat16(data["residual"], (num_tiles, H * H)),
        "head_weight": _restore_bfloat16(data["head_weight"], (H * n_cls_padded,)),
        "head_bias": _restore_bfloat16(data["head_bias"], (n_cls_padded,)),
    }


def _tail_stack_size_bytes(B: int, H: int, n_cls_padded: int) -> int:
    scratch_bytes = 2 * B * H + 4 * B * n_cls_padded
    # The tail kernel keeps several tile-local scratch buffers on the worker
    # stack. Budgeting only the raw array footprint is too tight once the
    # compiler adds its own frame temporaries, which shows up as all-NaN logits
    # at larger B values even though the design still compiles.
    stack_bytes = max(0x2000, scratch_bytes + 0x1000)
    return ((stack_bytes + 0x7FF) // 0x800) * 0x800


def snake_streaming_logits_pipeline(
    H=160,
    B=8,
    num_cols=8,
    stream_depth=32,
    archive_name="resmlp_streaming_logits_kernel.a",
    weights_path=None,
    n_cls_padded=N_CLS_PADDED,
):
    assert 1 <= num_cols <= 8
    assert B % 8 == 0 and H % 8 == 0
    assert stream_depth >= 1
    assert n_cls_padded % 8 == 0
    if weights_path is None:
        raise ValueError("weights_path is required for embedded-weight streaming inference")

    num_tiles = num_cols * ROWS_PER_COL
    packed = _load_embedded_weights(weights_path, num_tiles=num_tiles, H=H, n_cls_padded=n_cls_padded)
    tile_order = snake_tile_order(num_cols)
    fifo_depth = 1 if stream_depth == 1 else 2

    act_ty = np.ndarray[(B * H,), np.dtype[bfloat16]]
    res_wt_ty = np.ndarray[(H * H,), np.dtype[bfloat16]]
    head_wt_ty = np.ndarray[(H * n_cls_padded,), np.dtype[bfloat16]]
    head_bias_ty = np.ndarray[(n_cls_padded,), np.dtype[bfloat16]]
    logits_ty = np.ndarray[(B * n_cls_padded,), np.dtype[bfloat16]]
    host_in_ty = np.ndarray[(stream_depth, B, H), np.dtype[bfloat16]]
    host_out_ty = np.ndarray[(stream_depth, B, n_cls_padded), np.dtype[bfloat16]]

    res_kernel = Kernel(
        "matmul_relu_skip_infer_bf16",
        archive_name,
        [act_ty, res_wt_ty, act_ty],
    )
    tail_kernel = Kernel(
        "residual_head_infer_bf16",
        archive_name,
        [act_ty, res_wt_ty, head_wt_ty, head_bias_ty, logits_ty],
    )

    act_in = ObjectFifo(act_ty, name="act_in", depth=fifo_depth)
    logits_out = ObjectFifo(logits_ty, name="logits_out", depth=fifo_depth)
    act_inter = [
        ObjectFifo(act_ty, name=f"act_{i}", depth=fifo_depth)
        for i in range(num_tiles - 1)
    ]

    def residual_worker(of_in, of_out, weight_buf, kern):
        stream_loop = range(1)
        if stream_depth > 1:
            stream_loop = range_(stream_depth)

        for _ in stream_loop:
            x = of_in.acquire(1)
            y = of_out.acquire(1)
            kern(x, weight_buf, y)
            of_in.release(1)
            of_out.release(1)

    def tail_worker(of_in, of_out, res_weight_buf, head_weight_buf, head_bias_buf, kern):
        stream_loop = range(1)
        if stream_depth > 1:
            stream_loop = range_(stream_depth)

        for _ in stream_loop:
            x = of_in.acquire(1)
            logits = of_out.acquire(1)
            kern(x, res_weight_buf, head_weight_buf, head_bias_buf, logits)
            of_in.release(1)
            of_out.release(1)

    workers = []
    for idx, (col, row) in enumerate(tile_order[:-1]):
        in_ep = act_in.cons() if idx == 0 else act_inter[idx - 1].cons()
        out_ep = act_inter[idx].prod()
        weight_buf = Buffer(
            initial_value=np.array(packed["residual"][idx], dtype=np.dtype("bfloat16"), copy=True),
            name=f"weights_{idx}",
        )
        workers.append(
            Worker(
                residual_worker,
                fn_args=[in_ep, out_ep, weight_buf, res_kernel],
                placement=Tile(col=col, row=row),
                allocation_scheme="basic-sequential",
            )
        )

    tail_col, tail_row = tile_order[-1]
    tail_res_weight = Buffer(
        initial_value=np.array(packed["residual"][-1], dtype=np.dtype("bfloat16"), copy=True),
        name="tail_residual_weights",
    )
    tail_head_weight = Buffer(
        initial_value=np.array(packed["head_weight"], dtype=np.dtype("bfloat16"), copy=True),
        name="tail_head_weights",
    )
    tail_head_bias = Buffer(
        initial_value=np.array(packed["head_bias"], dtype=np.dtype("bfloat16"), copy=True),
        name="tail_head_bias",
    )
    workers.append(
        Worker(
            tail_worker,
            fn_args=[
                act_inter[-1].cons(),
                logits_out.prod(),
                tail_res_weight,
                tail_head_weight,
                tail_head_bias,
                tail_kernel,
            ],
            placement=Tile(col=tail_col, row=tail_row),
            allocation_scheme="basic-sequential",
            stack_size=_tail_stack_size_bytes(B, H, n_cls_padded),
        )
    )

    in_tap = TensorAccessPattern(
        (stream_depth, B, H),
        0,
        [stream_depth, 1, B, H],
        [B * H, 0, H, 1],
    )
    out_tap = TensorAccessPattern(
        (stream_depth, B, n_cls_padded),
        0,
        [stream_depth, 1, B, n_cls_padded],
        [B * n_cls_padded, 0, n_cls_padded, 1],
    )

    rt = Runtime()
    with rt.sequence(host_in_ty, host_out_ty) as (inp, out):
        rt.start(*workers)

        tg = rt.task_group()
        rt.fill(act_in.prod(), inp, tap=in_tap, task_group=tg)
        rt.drain(logits_out.cons(), out, tap=out_tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate streaming residual+head MLIR")
    p.add_argument("--H", type=int, default=160)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--stream-depth", type=int, default=32)
    p.add_argument("--n-cls-padded", type=int, default=N_CLS_PADDED)
    p.add_argument("--weights-path", required=True)
    p.add_argument("-o", "--output", default="build/resmlp_streaming_logits.mlir")
    args = p.parse_args()

    module = snake_streaming_logits_pipeline(
        H=args.H,
        B=args.B,
        num_cols=args.cols,
        stream_depth=args.stream_depth,
        weights_path=args.weights_path,
        n_cls_padded=args.n_cls_padded,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(str(module))
    print(f"Written {args.output}")
