"""
IRON design for forward-only residual MLP streaming inference with embedded weights.

Each tile gets a compile-time initialized local weight buffer, so repeated host
calls only stream activations in and activations out.

For deeper resident chunks, the host boundary uses one repeated shim DMA fill
and one repeated shim DMA drain task instead of issuing one host DMA task per
microbatch. Tile-local activation FIFOs stay at depth 2, so the design keeps
the original SRAM footprint while reducing runtime task pressure.
"""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern

from resmlp.design import ROWS_PER_COL, snake_tile_order


def snake_streaming_pipeline(
    H=160,
    B=8,
    num_cols=8,
    stream_depth=32,
    archive_name="resmlp_streaming_kernel.a",
    weights_path=None,
):
    """Generate a forward-only residual snake with tile-local embedded weights."""
    assert 1 <= num_cols <= 8
    assert B % 8 == 0 and H % 8 == 0
    assert stream_depth >= 1
    if weights_path is None:
        raise ValueError("weights_path is required for embedded-weight streaming inference")

    packed_weights = np.load(Path(weights_path), allow_pickle=False).view(np.dtype("bfloat16"))
    num_tiles = num_cols * ROWS_PER_COL
    if packed_weights.shape != (num_tiles, H * H):
        raise ValueError(
            f"Expected packed weights shape {(num_tiles, H * H)}, got {packed_weights.shape}"
        )

    tile_order = snake_tile_order(num_cols)
    act_ty = np.ndarray[(B * H,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(H * H,), np.dtype[bfloat16]]
    fifo_depth = 1 if stream_depth == 1 else 2
    host_chunk_ty = np.ndarray[(stream_depth, B, H), np.dtype[bfloat16]]

    kernel = Kernel(
        "matmul_relu_skip_infer_bf16",
        archive_name,
        [act_ty, wt_ty, act_ty],
    )

    act_in = ObjectFifo(act_ty, name="act_in", depth=fifo_depth)
    act_out = ObjectFifo(act_ty, name="act_out", depth=fifo_depth)
    act_inter = [
        ObjectFifo(act_ty, name=f"act_{i}", depth=fifo_depth)
        for i in range(num_tiles - 1)
    ]

    def worker_fn(of_in, of_out, weight_buf, kern):
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
    for idx, (col, row) in enumerate(tile_order):
        if idx == 0:
            in_ep = act_in.cons()
        else:
            in_ep = act_inter[idx - 1].cons()

        if idx == num_tiles - 1:
            out_ep = act_out.prod()
        else:
            out_ep = act_inter[idx].prod()

        weight_buf = Buffer(
            initial_value=np.array(packed_weights[idx], dtype=np.dtype("bfloat16"), copy=True),
            name=f"weights_{idx}",
        )

        workers.append(
            Worker(
                worker_fn,
                fn_args=[in_ep, out_ep, weight_buf, kernel],
                placement=Tile(col=col, row=row),
                allocation_scheme="basic-sequential",
            )
        )

    stream_tap = TensorAccessPattern(
        (stream_depth, B, H),
        0,
        [stream_depth, 1, B, H],
        [B * H, 0, H, 1],
    )

    rt = Runtime()
    with rt.sequence(host_chunk_ty, host_chunk_ty) as (inp, out):
        rt.start(*workers)

        tg = rt.task_group()
        rt.fill(act_in.prod(), inp, tap=stream_tap, task_group=tg)
        rt.drain(act_out.cons(), out, tap=stream_tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate streaming residual snake MLIR")
    p.add_argument("--H", type=int, default=160)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--stream-depth", type=int, default=32)
    p.add_argument("--weights-path", required=True)
    p.add_argument("-o", "--output", default="build/resmlp_streaming.mlir")
    args = p.parse_args()

    module = snake_streaming_pipeline(
        H=args.H,
        B=args.B,
        num_cols=args.cols,
        stream_depth=args.stream_depth,
        weights_path=args.weights_path,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(str(module))
    print(f"Written {args.output}")
