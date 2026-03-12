"""
One-column forward probe that spills each tile's input activation to host.

This is the first concrete checkpoint-strategy probe for NPU training:

- 4 residual layers on one column,
- normal forward activation flow,
- plus a joined checkpoint buffer that captures x_i for each tile.
"""

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.placers import SequentialPlacer


ROWS_PER_COL = 4


def forward_checkpoint_column(H=160, B=8):
    assert B % 8 == 0 and H % 8 == 0

    act_ty = np.ndarray[(B * H,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(H * H,), np.dtype[bfloat16]]
    col_wt_ty = np.ndarray[(ROWS_PER_COL * H * H,), np.dtype[bfloat16]]
    ckpt_ty = np.ndarray[(ROWS_PER_COL * B * H,), np.dtype[bfloat16]]

    fwd_kernel = Kernel(
        "matmul_relu_skip_bf16",
        "resmlp_forward_checkpoint_kernels.a",
        [act_ty, wt_ty, act_ty],
    )
    copy_kernel = Kernel(
        "copy_activation_bf16",
        "resmlp_forward_checkpoint_kernels.a",
        [act_ty, act_ty],
    )

    wt_ddr = ObjectFifo(col_wt_ty, name="wt_col0", depth=1)
    wt_parts = wt_ddr.cons().split(
        offsets=[H * H * r for r in range(ROWS_PER_COL)],
        obj_types=[wt_ty] * ROWS_PER_COL,
        names=[f"wt_{r}" for r in range(ROWS_PER_COL)],
        depths=[1] * ROWS_PER_COL,
        placement=Tile(col=0, row=1),
    )

    act_in = ObjectFifo(act_ty, name="act_in", depth=1)
    act_out = ObjectFifo(act_ty, name="act_out", depth=1)
    act_inter = [
        ObjectFifo(act_ty, name=f"act_{i}", depth=1)
        for i in range(ROWS_PER_COL - 1)
    ]

    ckpt_ddr = ObjectFifo(ckpt_ty, name="ckpt_out", depth=1)
    ckpt_parts = ckpt_ddr.prod().join(
        offsets=[B * H * i for i in range(ROWS_PER_COL)],
        obj_types=[act_ty] * ROWS_PER_COL,
        names=[f"ckpt_{i}" for i in range(ROWS_PER_COL)],
        depths=[1] * ROWS_PER_COL,
        placement=Tile(col=0, row=1),
    )

    def worker_fn(of_in, of_out, of_w, of_ckpt, copy_kern, fwd_kern):
        x = of_in.acquire(1)
        y = of_out.acquire(1)
        w = of_w.acquire(1)
        ckpt = of_ckpt.acquire(1)
        copy_kern(x, ckpt)
        fwd_kern(x, w, y)
        of_in.release(1)
        of_out.release(1)
        of_w.release(1)
        of_ckpt.release(1)

    workers = []
    for idx, row in enumerate(range(2, 6)):
        in_ep = act_in.cons() if idx == 0 else act_inter[idx - 1].cons()
        out_ep = act_out.prod() if idx == ROWS_PER_COL - 1 else act_inter[idx].prod()
        wt_ep = wt_parts[idx].cons() if hasattr(wt_parts[idx], "cons") else wt_parts[idx]
        workers.append(
            Worker(
                worker_fn,
                [in_ep, out_ep, wt_ep, ckpt_parts[idx].prod(), copy_kernel, fwd_kernel],
                placement=Tile(col=0, row=row),
            )
        )

    rt = Runtime()
    with rt.sequence(act_ty, col_wt_ty, ckpt_ty, act_ty) as (inp, wts, ckpt, out):
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(act_in.prod(), inp, task_group=tg)
        rt.fill(wt_ddr.prod(), wts, task_group=tg)
        rt.drain(ckpt_ddr.cons(), ckpt, wait=True, task_group=tg)
        rt.drain(act_out.cons(), out, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())
