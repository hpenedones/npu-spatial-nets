"""Streaming 32-tile convolutional snake with embedded weights."""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern

from convsnake.config import BATCH_SIZE, DEFAULT_CONFIG, ConvSnakeConfig, ROWS_PER_COL, num_blocks_for_cols
from resmlp.design import snake_tile_order


def _restore_bfloat16(array, shape) -> np.ndarray:
    raw = np.asarray(array)
    if raw.itemsize != 2:
        raise ValueError(f"expected 2-byte packed weights, got dtype={raw.dtype}")
    return raw.view(bfloat16).reshape(shape)


def _load_embedded_weights(
    weights_path: str | Path,
    *,
    num_blocks: int,
    config: ConvSnakeConfig,
) -> dict[str, np.ndarray]:
    data = np.load(Path(weights_path), allow_pickle=False)
    return {
        "conv1": _restore_bfloat16(data["conv1"], config.conv1_npu_w_elems),
        "conv2": _restore_bfloat16(data["conv2"], config.conv2_npu_w_elems),
        "conv3": _restore_bfloat16(data["conv3"], config.conv3_npu_w_elems),
        "blocks": _restore_bfloat16(data["blocks"], (num_blocks, config.block_npu_w_elems)),
        "head": _restore_bfloat16(data["head"], config.head_w_elems),
    }


def convsnake_streaming_pipeline(
    B=BATCH_SIZE,
    num_cols=8,
    stream_depth=32,
    archive_name="convsnake_streaming_kernel.a",
    weights_path=None,
    config_kwargs=None,
):
    cfg = DEFAULT_CONFIG if config_kwargs is None else ConvSnakeConfig.from_dict(config_kwargs)

    assert B == cfg.batch_size
    assert 1 <= num_cols <= 8
    assert stream_depth >= 1
    if weights_path is None:
        raise ValueError("weights_path is required for embedded-weight streaming inference")

    num_tiles = num_cols * ROWS_PER_COL
    num_blocks = num_blocks_for_cols(num_cols)
    packed_weights = _load_embedded_weights(weights_path, num_blocks=num_blocks, config=cfg)
    tile_order = snake_tile_order(num_cols)
    fifo_depth = 1 if stream_depth == 1 else 2

    img_ty = np.ndarray[(cfg.img_elems,), np.dtype[bfloat16]]
    act1_ty = np.ndarray[(cfg.act1_elems,), np.dtype[bfloat16]]
    act2_ty = np.ndarray[(cfg.act2_elems,), np.dtype[bfloat16]]
    act3_ty = np.ndarray[(cfg.act3_elems,), np.dtype[bfloat16]]
    logits_ty = np.ndarray[(cfg.logits_elems,), np.dtype[bfloat16]]

    host_in_ty = np.ndarray[
        (stream_depth, cfg.batch_size, cfg.img_c * cfg.img_h, cfg.img_w),
        np.dtype[bfloat16],
    ]
    host_out_ty = np.ndarray[(stream_depth, cfg.batch_size, cfg.num_classes), np.dtype[bfloat16]]

    conv1_kernel = Kernel(
        "conv1_infer_relu_bf16",
        archive_name,
        [img_ty, np.ndarray[(cfg.conv1_npu_w_elems,), np.dtype[bfloat16]], act1_ty],
    )
    conv2_kernel = Kernel(
        "conv2_infer_relu_bf16",
        archive_name,
        [act1_ty, np.ndarray[(cfg.conv2_npu_w_elems,), np.dtype[bfloat16]], act2_ty],
    )
    conv3_kernel = Kernel(
        "conv3_infer_relu_bf16",
        archive_name,
        [act2_ty, np.ndarray[(cfg.conv3_npu_w_elems,), np.dtype[bfloat16]], act3_ty],
    )
    conv4_kernel = Kernel(
        "conv4_infer_relu_bf16",
        archive_name,
        [act3_ty, np.ndarray[(cfg.block_npu_w_elems,), np.dtype[bfloat16]], act3_ty],
    )
    head_kernel = Kernel(
        "flatten_head_infer_bf16",
        archive_name,
        [act3_ty, np.ndarray[(cfg.head_w_elems,), np.dtype[bfloat16]], logits_ty],
    )

    act_in = ObjectFifo(img_ty, name="images", depth=fifo_depth)
    act1_fifo = ObjectFifo(act1_ty, name="act1", depth=fifo_depth)
    act2_fifo = ObjectFifo(act2_ty, name="act2", depth=fifo_depth)
    act3_fifos = [ObjectFifo(act3_ty, name=f"act3_{i}", depth=fifo_depth) for i in range(num_tiles - 3)]
    act_out = ObjectFifo(logits_ty, name="logits", depth=fifo_depth)

    def conv_worker(of_in, of_out, weight_buf, kern):
        stream_loop = range(1)
        if stream_depth > 1:
            stream_loop = range_(stream_depth)

        for _ in stream_loop:
            x = of_in.acquire(1)
            y = of_out.acquire(1)
            kern(x, weight_buf, y)
            of_in.release(1)
            of_out.release(1)

    def head_worker(of_in, of_out, weight_buf, kern):
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
            out_ep = act1_fifo.prod()
            kernel = conv1_kernel
            weight_array = packed_weights["conv1"]
            worker_fn = conv_worker
        elif idx == 1:
            in_ep = act1_fifo.cons()
            out_ep = act2_fifo.prod()
            kernel = conv2_kernel
            weight_array = packed_weights["conv2"]
            worker_fn = conv_worker
        elif idx == 2:
            in_ep = act2_fifo.cons()
            out_ep = act3_fifos[0].prod()
            kernel = conv3_kernel
            weight_array = packed_weights["conv3"]
            worker_fn = conv_worker
        elif idx == num_tiles - 1:
            in_ep = act3_fifos[-1].cons()
            out_ep = act_out.prod()
            kernel = head_kernel
            weight_array = packed_weights["head"]
            worker_fn = head_worker
        else:
            block_idx = idx - 3
            in_ep = act3_fifos[block_idx].cons()
            out_ep = act3_fifos[block_idx + 1].prod()
            kernel = conv4_kernel
            weight_array = packed_weights["blocks"][block_idx]
            worker_fn = conv_worker

        weight_buf = Buffer(
            initial_value=np.array(weight_array, dtype=np.dtype("bfloat16"), copy=True),
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

    stream_in_tap = TensorAccessPattern(
        (stream_depth, cfg.batch_size, cfg.img_c * cfg.img_h, cfg.img_w),
        0,
        [stream_depth, cfg.batch_size, cfg.img_c * cfg.img_h, cfg.img_w],
        [
            cfg.img_elems,
            cfg.img_elems // cfg.batch_size,
            cfg.img_w,
            1,
        ],
    )
    stream_out_tap = TensorAccessPattern(
        (stream_depth, cfg.batch_size, cfg.num_classes),
        0,
        [stream_depth, cfg.batch_size, cfg.num_classes],
        [cfg.logits_elems, cfg.num_classes, 1],
    )

    rt = Runtime()
    with rt.sequence(host_in_ty, host_out_ty) as (inp, out):
        rt.start(*workers)

        tg = rt.task_group()
        rt.fill(act_in.prod(), inp, tap=stream_in_tap, task_group=tg)
        rt.drain(act_out.cons(), out, tap=stream_out_tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate streaming convolutional snake MLIR")
    p.add_argument("--B", type=int, default=BATCH_SIZE)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--stream-depth", type=int, default=32)
    p.add_argument("--weights-path", required=True)
    p.add_argument("--dataset", choices=("mnist", "cifar10"), default="mnist")
    p.add_argument("-o", "--output", default="build/convsnake_streaming.mlir")
    args = p.parse_args()

    module = convsnake_streaming_pipeline(
        B=args.B,
        num_cols=args.cols,
        stream_depth=args.stream_depth,
        weights_path=args.weights_path,
        config_kwargs={"dataset": args.dataset, "batch_size": args.B},
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(str(module))
    print(f"Written {args.output}")
