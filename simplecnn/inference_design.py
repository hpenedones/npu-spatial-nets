import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.placers import SequentialPlacer

from simplecnn.config import (
    ACT1_ELEMS,
    ACT2_ELEMS,
    ACT3_ELEMS,
    BATCH_SIZE,
    CONV1_CKPT_ELEMS,
    CONV1_W_ELEMS,
    CONV2_CKPT_ELEMS,
    CONV2_W_ELEMS,
    CONV3_CKPT_ELEMS,
    CONV3_W_ELEMS,
    HEAD_W_ELEMS,
    IMG_ELEMS,
    N_CLASSES,
    POOLED_ELEMS,
    TOTAL_WEIGHT_ELEMS,
    WEIGHT_OFFSETS,
)


def simplecnn_inference_pipeline(archive_name):
    img_ty = np.ndarray[(IMG_ELEMS,), np.dtype[bfloat16]]
    act1_ty = np.ndarray[(ACT1_ELEMS,), np.dtype[bfloat16]]
    act2_ty = np.ndarray[(ACT2_ELEMS,), np.dtype[bfloat16]]
    act3_ty = np.ndarray[(ACT3_ELEMS,), np.dtype[bfloat16]]
    pooled_ty = np.ndarray[(POOLED_ELEMS,), np.dtype[bfloat16]]
    logits_ty = np.ndarray[(BATCH_SIZE * N_CLASSES,), np.dtype[bfloat16]]

    conv1_w_ty = np.ndarray[(CONV1_W_ELEMS,), np.dtype[bfloat16]]
    conv2_w_ty = np.ndarray[(CONV2_W_ELEMS,), np.dtype[bfloat16]]
    conv3_w_ty = np.ndarray[(CONV3_W_ELEMS,), np.dtype[bfloat16]]
    head_w_ty = np.ndarray[(HEAD_W_ELEMS,), np.dtype[bfloat16]]
    host_weights_ty = np.ndarray[(TOTAL_WEIGHT_ELEMS,), np.dtype[bfloat16]]

    ckpt1_ty = np.ndarray[(CONV1_CKPT_ELEMS,), np.dtype[bfloat16]]
    ckpt2_ty = np.ndarray[(CONV2_CKPT_ELEMS,), np.dtype[bfloat16]]
    ckpt3_ty = np.ndarray[(CONV3_CKPT_ELEMS,), np.dtype[bfloat16]]

    conv1_fwd = Kernel(
        "conv1_forward_relu_bf16",
        archive_name,
        [img_ty, conv1_w_ty, act1_ty, ckpt1_ty],
    )
    conv2_fwd = Kernel(
        "conv2_forward_relu_bf16",
        archive_name,
        [act1_ty, conv2_w_ty, act2_ty, ckpt2_ty],
    )
    conv3_fwd = Kernel(
        "conv3_forward_relu_bf16",
        archive_name,
        [act2_ty, conv3_w_ty, act3_ty, ckpt3_ty],
    )
    gap_fwd = Kernel(
        "gap_forward_bf16",
        archive_name,
        [act3_ty, pooled_ty],
    )
    head_infer = Kernel(
        "simple_head_infer_bf16",
        archive_name,
        [pooled_ty, head_w_ty, logits_ty],
    )

    img_fifo = ObjectFifo(img_ty, name="images", depth=1)
    act1_fifo = ObjectFifo(act1_ty, name="act1", depth=1)
    act2_fifo = ObjectFifo(act2_ty, name="act2", depth=1)
    pooled_fifo = ObjectFifo(pooled_ty, name="pooled", depth=1)
    logits_fifo = ObjectFifo(logits_ty, name="logits", depth=1)

    conv1_w_fifo = ObjectFifo(conv1_w_ty, name="conv1_w", depth=1)
    conv2_w_fifo = ObjectFifo(conv2_w_ty, name="conv2_w", depth=1)
    conv3_w_fifo = ObjectFifo(conv3_w_ty, name="conv3_w", depth=1)
    head_w_fifo = ObjectFifo(head_w_ty, name="head_w", depth=1)

    ckpt1_fifo = ObjectFifo(ckpt1_ty, name="ckpt1", depth=1)
    ckpt2_fifo = ObjectFifo(ckpt2_ty, name="ckpt2", depth=1)
    ckpt3_fifo = ObjectFifo(ckpt3_ty, name="ckpt3", depth=1)
    act3_local_fifo = ObjectFifo(act3_ty, name="act3_local", depth=1)

    def conv_worker(of_in, of_out, of_w, of_ckpt_prod, of_ckpt_cons, fwd_k):
        x = of_in.acquire(1)
        y = of_out.acquire(1)
        w = of_w.acquire(1)
        ckpt = of_ckpt_prod.acquire(1)
        fwd_k(x, w, y, ckpt)
        of_ckpt_prod.release(1)
        ckpt_drop = of_ckpt_cons.acquire(1)
        del ckpt_drop
        of_ckpt_cons.release(1)
        of_w.release(1)
        of_out.release(1)
        of_in.release(1)

    def conv3_worker(
        of_in,
        of_pooled_out,
        of_w,
        of_ckpt_prod,
        of_ckpt_cons,
        of_act3_prod,
        of_act3_cons,
        fwd_k,
        gap_k,
    ):
        x = of_in.acquire(1)
        pooled = of_pooled_out.acquire(1)
        w = of_w.acquire(1)
        ckpt = of_ckpt_prod.acquire(1)
        act3 = of_act3_prod.acquire(1)
        fwd_k(x, w, act3, ckpt)
        of_ckpt_prod.release(1)
        ckpt_drop = of_ckpt_cons.acquire(1)
        del ckpt_drop
        of_ckpt_cons.release(1)
        of_act3_prod.release(1)
        act3_read = of_act3_cons.acquire(1)
        gap_k(act3_read, pooled)
        of_act3_cons.release(1)
        of_w.release(1)
        of_pooled_out.release(1)
        of_in.release(1)

    def head_worker(of_pooled_in, of_head_w, of_logits_out, head_k):
        pooled = of_pooled_in.acquire(1)
        w = of_head_w.acquire(1)
        logits = of_logits_out.acquire(1)
        head_k(pooled, w, logits)
        of_logits_out.release(1)
        of_head_w.release(1)
        of_pooled_in.release(1)

    workers = [
        Worker(
            conv_worker,
            [
                img_fifo.cons(),
                act1_fifo.prod(),
                conv1_w_fifo.cons(),
                ckpt1_fifo.prod(),
                ckpt1_fifo.cons(),
                conv1_fwd,
            ],
            placement=Tile(col=0, row=2),
            stack_size=0x1000,
        ),
        Worker(
            conv_worker,
            [
                act1_fifo.cons(),
                act2_fifo.prod(),
                conv2_w_fifo.cons(),
                ckpt2_fifo.prod(),
                ckpt2_fifo.cons(),
                conv2_fwd,
            ],
            placement=Tile(col=0, row=3),
            stack_size=0x1000,
        ),
        Worker(
            conv3_worker,
            [
                act2_fifo.cons(),
                pooled_fifo.prod(),
                conv3_w_fifo.cons(),
                ckpt3_fifo.prod(),
                ckpt3_fifo.cons(),
                act3_local_fifo.prod(),
                act3_local_fifo.cons(),
                conv3_fwd,
                gap_fwd,
            ],
            placement=Tile(col=0, row=4),
            stack_size=0x1000,
        ),
        Worker(
            head_worker,
            [pooled_fifo.cons(), head_w_fifo.cons(), logits_fifo.prod(), head_infer],
            placement=Tile(col=0, row=5),
            stack_size=0x3000,
        ),
    ]

    def weight_tap(offset: int, elems: int) -> TensorAccessPattern:
        return TensorAccessPattern(
            (1, TOTAL_WEIGHT_ELEMS),
            offset,
            [1, elems],
            [0, 1],
        )

    host_img_ty = np.ndarray[(IMG_ELEMS,), np.dtype[bfloat16]]
    host_logits_ty = np.ndarray[(BATCH_SIZE * N_CLASSES,), np.dtype[bfloat16]]

    rt = Runtime()
    with rt.sequence(host_img_ty, host_weights_ty, host_logits_ty) as (
        images,
        weights,
        logits,
    ):
        rt.start(*workers)

        tg = rt.task_group()
        rt.fill(img_fifo.prod(), images, task_group=tg)
        rt.fill(
            conv1_w_fifo.prod(),
            weights,
            tap=weight_tap(WEIGHT_OFFSETS.conv1, CONV1_W_ELEMS),
            task_group=tg,
        )
        rt.fill(
            conv2_w_fifo.prod(),
            weights,
            tap=weight_tap(WEIGHT_OFFSETS.conv2, CONV2_W_ELEMS),
            task_group=tg,
        )
        rt.fill(
            conv3_w_fifo.prod(),
            weights,
            tap=weight_tap(WEIGHT_OFFSETS.conv3, CONV3_W_ELEMS),
            task_group=tg,
        )
        rt.fill(
            head_w_fifo.prod(),
            weights,
            tap=weight_tap(WEIGHT_OFFSETS.head, HEAD_W_ELEMS),
            task_group=tg,
        )
        rt.drain(logits_fifo.cons(), logits, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())
