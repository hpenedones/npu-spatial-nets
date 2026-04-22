"""IRON design for the full-NPU residual MLP streaming inference pipeline.

Each tile gets a compile-time initialized local weight buffer, so repeated host
calls only stream activations in and activations out. The array runs the embed
stage on the first tile, the residual body on the middle tiles, and the padded
head stage on the final tile.
"""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern

from resmlp import TILE_BLOCK
from resmlp.design import ROWS_PER_COL, snake_tile_order


def _stream_loop(stream_depth):
    return range_(stream_depth) if stream_depth > 1 else range(1)


def _residual_worker(of_in, of_out, weight_buf, kern, stream_depth):
    for _ in _stream_loop(stream_depth):
        x = of_in.acquire(1)
        y = of_out.acquire(1)
        kern(x, weight_buf, y)
        of_in.release(1)
        of_out.release(1)


def _linear_worker(of_in, of_out, weight_buf, bias_buf, kern, stream_depth):
    for _ in _stream_loop(stream_depth):
        x = of_in.acquire(1)
        y = of_out.acquire(1)
        kern(x, weight_buf, bias_buf, y)
        of_in.release(1)
        of_out.release(1)


def _load_bf16_array(path, expected_shape, label):
    array = np.load(Path(path), allow_pickle=False).view(np.dtype("bfloat16"))
    if array.shape != expected_shape:
        raise ValueError(f"Expected {label} shape {expected_shape}, got {array.shape}")
    return array


def _stream_tap(stream_depth, B, width):
    return TensorAccessPattern(
        (stream_depth, B, width),
        0,
        [stream_depth, 1, B, width],
        [B * width, 0, width, 1],
    )


def snake_streaming_pipeline(
    H=160,
    B=8,
    num_cols=8,
    stream_depth=32,
    residual_archive_name="resmlp_streaming_kernel.a",
    residual_weights_path=None,
    embed_archive_name=None,
    embed_weights_path=None,
    embed_bias_path=None,
    head_archive_name=None,
    head_weights_path=None,
    head_bias_path=None,
    input_dim_device=None,
    output_dim_device=None,
):
    """Generate the forward-only full-NPU residual snake with compile-time embedded weights."""
    assert 1 <= num_cols <= 8
    assert B % TILE_BLOCK == 0 and H % TILE_BLOCK == 0
    assert stream_depth >= 1
    if residual_weights_path is None:
        raise ValueError("residual_weights_path is required for streaming inference")
    if embed_archive_name is None or head_archive_name is None:
        raise ValueError("embed and head archive names are required")
    if embed_weights_path is None or embed_bias_path is None:
        raise ValueError("embed weights and bias are required")
    if head_weights_path is None or head_bias_path is None:
        raise ValueError("head weights and bias are required")
    if input_dim_device is None or output_dim_device is None:
        raise ValueError("input_dim_device and output_dim_device are required")
    if input_dim_device % TILE_BLOCK != 0 or output_dim_device % TILE_BLOCK != 0:
        raise ValueError("device-side input/output widths must be divisible by 8")

    num_tiles = num_cols * ROWS_PER_COL
    if num_tiles < 2:
        raise ValueError("streaming pipeline requires at least two tiles")
    tile_order = snake_tile_order(num_cols)
    fifo_depth = 1 if stream_depth == 1 else 2

    hidden_ty = np.ndarray[(B * H,), np.dtype[bfloat16]]
    hidden_wt_ty = np.ndarray[(H * H,), np.dtype[bfloat16]]
    residual_kernel = Kernel(
        "matmul_relu_skip_infer_bf16",
        residual_archive_name,
        [hidden_ty, hidden_wt_ty, hidden_ty],
    )

    residual_tiles = num_tiles - 2
    packed_residual_weights = _load_bf16_array(
        residual_weights_path,
        (residual_tiles, H * H),
        "packed residual weights",
    )
    packed_embed_weights = _load_bf16_array(
        embed_weights_path,
        (input_dim_device * H,),
        "packed embed weights",
    )
    packed_embed_bias = _load_bf16_array(
        embed_bias_path,
        (H * TILE_BLOCK,),
        "packed embed bias",
    )
    packed_head_weights = _load_bf16_array(
        head_weights_path,
        (H * output_dim_device,),
        "packed head weights",
    )
    packed_head_bias = _load_bf16_array(
        head_bias_path,
        (output_dim_device * TILE_BLOCK,),
        "packed head bias",
    )

    input_ty = np.ndarray[(B * input_dim_device,), np.dtype[bfloat16]]
    output_ty = np.ndarray[(B * output_dim_device,), np.dtype[bfloat16]]
    embed_wt_ty = np.ndarray[(input_dim_device * H,), np.dtype[bfloat16]]
    embed_bias_ty = np.ndarray[(H * TILE_BLOCK,), np.dtype[bfloat16]]
    head_wt_ty = np.ndarray[(H * output_dim_device,), np.dtype[bfloat16]]
    head_bias_ty = np.ndarray[(output_dim_device * TILE_BLOCK,), np.dtype[bfloat16]]

    embed_kernel = Kernel(
        "matmul_bias_embed_bf16",
        embed_archive_name,
        [input_ty, embed_wt_ty, embed_bias_ty, hidden_ty],
    )
    head_kernel = Kernel(
        "matmul_bias_head_bf16",
        head_archive_name,
        [hidden_ty, head_wt_ty, head_bias_ty, output_ty],
    )

    act_in = ObjectFifo(input_ty, name="act_in", depth=fifo_depth)
    act_out = ObjectFifo(output_ty, name="act_out", depth=fifo_depth)
    hidden_fifos = [
        ObjectFifo(hidden_ty, name=f"act_{i}", depth=fifo_depth)
        for i in range(residual_tiles + 1)
    ]

    workers = []

    embed_weight_buf = Buffer(
        initial_value=np.array(packed_embed_weights, dtype=np.dtype("bfloat16"), copy=True),
        name="embed_weights",
    )
    embed_bias_buf = Buffer(
        initial_value=np.array(packed_embed_bias, dtype=np.dtype("bfloat16"), copy=True),
        name="embed_bias",
    )
    workers.append(
        Worker(
            _linear_worker,
            fn_args=[
                act_in.cons(),
                hidden_fifos[0].prod(),
                embed_weight_buf,
                embed_bias_buf,
                embed_kernel,
                stream_depth,
            ],
            placement=Tile(col=tile_order[0][0], row=tile_order[0][1]),
            allocation_scheme="basic-sequential",
        )
    )

    for idx, (col, row) in enumerate(tile_order[1:-1]):
        weight_buf = Buffer(
            initial_value=np.array(packed_residual_weights[idx], dtype=np.dtype("bfloat16"), copy=True),
            name=f"weights_{idx}",
        )
        workers.append(
            Worker(
                _residual_worker,
                fn_args=[
                    hidden_fifos[idx].cons(),
                    hidden_fifos[idx + 1].prod(),
                    weight_buf,
                    residual_kernel,
                    stream_depth,
                ],
                placement=Tile(col=col, row=row),
                allocation_scheme="basic-sequential",
            )
        )

    head_weight_buf = Buffer(
        initial_value=np.array(packed_head_weights, dtype=np.dtype("bfloat16"), copy=True),
        name="head_weights",
    )
    head_bias_buf = Buffer(
        initial_value=np.array(packed_head_bias, dtype=np.dtype("bfloat16"), copy=True),
        name="head_bias",
    )
    workers.append(
        Worker(
            _linear_worker,
            fn_args=[
                hidden_fifos[-1].cons(),
                act_out.prod(),
                head_weight_buf,
                head_bias_buf,
                head_kernel,
                stream_depth,
            ],
            placement=Tile(col=tile_order[-1][0], row=tile_order[-1][1]),
            allocation_scheme="basic-sequential",
        )
    )

    host_in_chunk_ty = np.ndarray[(stream_depth, B, input_dim_device), np.dtype[bfloat16]]
    host_out_chunk_ty = np.ndarray[(stream_depth, B, output_dim_device), np.dtype[bfloat16]]

    rt = Runtime()
    with rt.sequence(host_in_chunk_ty, host_out_chunk_ty) as (inp, out):
        rt.start(*workers)

        tg = rt.task_group()
        rt.fill(act_in.prod(), inp, tap=_stream_tap(stream_depth, B, input_dim_device), task_group=tg)
        rt.drain(act_out.cons(), out, tap=_stream_tap(stream_depth, B, output_dim_device), wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate streaming residual snake MLIR")
    p.add_argument("--H", type=int, default=160)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--stream-depth", type=int, default=32)
    p.add_argument("--residual-archive-name", default="resmlp_streaming_kernel.a")
    p.add_argument("--residual-weights-path", required=True)
    p.add_argument("--embed-archive-name", required=True)
    p.add_argument("--embed-weights-path", required=True)
    p.add_argument("--embed-bias-path", required=True)
    p.add_argument("--head-archive-name", required=True)
    p.add_argument("--head-weights-path", required=True)
    p.add_argument("--head-bias-path", required=True)
    p.add_argument("--input-dim-device", type=int, required=True)
    p.add_argument("--output-dim-device", type=int, required=True)
    p.add_argument("-o", "--output", default="build/resmlp_streaming.mlir")
    args = p.parse_args()

    module = snake_streaming_pipeline(
        H=args.H,
        B=args.B,
        num_cols=args.cols,
        stream_depth=args.stream_depth,
        residual_archive_name=args.residual_archive_name,
        residual_weights_path=args.residual_weights_path,
        embed_archive_name=args.embed_archive_name,
        embed_weights_path=args.embed_weights_path,
        embed_bias_path=args.embed_bias_path,
        head_archive_name=args.head_archive_name,
        head_weights_path=args.head_weights_path,
        head_bias_path=args.head_bias_path,
        input_dim_device=args.input_dim_device,
        output_dim_device=args.output_dim_device,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(str(module))
    print(f"Written {args.output}")
