"""Streaming residual inference with the embed layer moved onto tile 0."""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern

from resmlp.design import ROWS_PER_COL, snake_tile_order

EMBED_INPUT_DIM = 784
EMBED_CHUNK_ROWS = 56


def _restore_bfloat16(array, shape):
    raw = np.asarray(array)
    if raw.itemsize != 2:
        raise ValueError(f"expected 2-byte packed weights, got dtype={raw.dtype}")
    return raw.view(bfloat16).reshape(shape)


def _load_embedded_weights(weights_path: str | Path, num_residual: int, H: int):
    data = np.load(Path(weights_path), allow_pickle=False)
    return {
        "embed": _restore_bfloat16(data["embed"], (EMBED_INPUT_DIM * H,)),
        "residual": _restore_bfloat16(data["residual"], (num_residual, H * H)),
    }


def snake_streaming_embed_pipeline(
    H=32,
    B=8,
    num_cols=8,
    stream_depth=32,
    archive_name="resmlp_streaming_embed_kernel.a",
    weights_path=None,
):
    assert 1 <= num_cols <= 8
    assert B % 8 == 0 and H % 8 == 0
    assert stream_depth >= 1
    if weights_path is None:
        raise ValueError("weights_path is required for embedded-weight streaming inference")

    num_tiles = num_cols * ROWS_PER_COL
    num_residual = num_tiles - 1
    packed_weights = _load_embedded_weights(weights_path, num_residual=num_residual, H=H)
    tile_order = snake_tile_order(num_cols)
    fifo_depth = 1 if stream_depth == 1 else 2

    x_elems = B * EMBED_INPUT_DIM
    x_ty = np.ndarray[(x_elems,), np.dtype[bfloat16]]
    act_ty = np.ndarray[(B * H,), np.dtype[bfloat16]]
    embed_wt_ty = np.ndarray[(EMBED_INPUT_DIM * H,), np.dtype[bfloat16]]
    res_wt_ty = np.ndarray[(H * H,), np.dtype[bfloat16]]
    host_in_ty = np.ndarray[(stream_depth, EMBED_INPUT_DIM // 8, 8, 8), np.dtype[bfloat16]]
    host_out_ty = np.ndarray[(stream_depth, B, H), np.dtype[bfloat16]]

    embed_kernel = Kernel(
        "embed_forward_resident_bf16",
        archive_name,
        [x_ty, embed_wt_ty, act_ty],
    )
    res_kernel = Kernel(
        "matmul_relu_skip_infer_bf16",
        archive_name,
        [act_ty, res_wt_ty, act_ty],
    )

    embed_in = ObjectFifo(x_ty, name="embed_in", depth=1)
    act_out = ObjectFifo(act_ty, name="act_out", depth=fifo_depth)
    act_inter = [
        ObjectFifo(act_ty, name=f"act_{i}", depth=1 if i == 0 else fifo_depth)
        for i in range(num_residual)
    ]

    def embed_worker(of_x, of_y, weight_buf, kern):
        stream_loop = range(1)
        if stream_depth > 1:
            stream_loop = range_(stream_depth)

        for _ in stream_loop:
            x = of_x.acquire(1)
            y = of_y.acquire(1)
            kern(x, weight_buf, y)
            of_x.release(1)
            of_y.release(1)

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

    workers = []
    embed_col, embed_row = tile_order[0]
    embed_weight_buf = Buffer(
        initial_value=np.array(packed_weights["embed"], dtype=np.dtype("bfloat16"), copy=True),
        name="embed_weights",
    )
    workers.append(
        Worker(
            embed_worker,
            fn_args=[embed_in.cons(), act_inter[0].prod(), embed_weight_buf, embed_kernel],
            placement=Tile(col=embed_col, row=embed_row),
            allocation_scheme="basic-sequential",
            stack_size=0x800,
        )
    )

    for idx, (col, row) in enumerate(tile_order[1:], start=0):
        in_ep = act_inter[idx].cons()
        out_ep = act_out.prod() if idx == num_residual - 1 else act_inter[idx + 1].prod()
        weight_buf = Buffer(
            initial_value=np.array(
                packed_weights["residual"][idx],
                dtype=np.dtype("bfloat16"),
                copy=True,
            ),
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

    in_tap = TensorAccessPattern(
        (stream_depth, EMBED_INPUT_DIM // 8, 8, 8),
        0,
        [stream_depth, EMBED_INPUT_DIM // 8, 8, 8],
        [x_elems, 64, 8, 1],
    )
    out_tap = TensorAccessPattern(
        (stream_depth, B, H),
        0,
        [stream_depth, 1, B, H],
        [B * H, 0, H, 1],
    )

    rt = Runtime()
    with rt.sequence(host_in_ty, host_out_ty) as (inp, out):
        rt.start(*workers)

        tg = rt.task_group()
        rt.fill(embed_in.prod(), inp, tap=in_tap, task_group=tg)
        rt.drain(act_out.cons(), out, tap=out_tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate streaming residual+embed MLIR")
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--stream-depth", type=int, default=32)
    p.add_argument("--weights-path", required=True)
    p.add_argument("-o", "--output", default="build/resmlp_streaming_embed.mlir")
    args = p.parse_args()

    module = snake_streaming_embed_pipeline(
        H=args.H,
        B=args.B,
        num_cols=args.cols,
        stream_depth=args.stream_depth,
        weights_path=args.weights_path,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(str(module))
    print(f"Written {args.output}")
