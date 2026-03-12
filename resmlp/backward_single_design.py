"""
Single-tile IRON designs for phase-0 residual backward validation.

Two tiny operators are exposed:

    grad_input : gx = gy + (gy * mask) @ W^T
    weight_grad: dW = x^T @ (gy * mask)

They intentionally keep the outputs separate so each run fits within the tile
SRAM budget.
"""

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.placers import SequentialPlacer


def backward_single(H=160, B=8, mode="grad_input"):
    assert B % 8 == 0 and H % 8 == 0

    act_ty = np.ndarray[(B * H,), np.dtype[bfloat16]]
    gx_state_ty = np.ndarray[(2 * B * H,), np.dtype[bfloat16]]
    dw_state_ty = np.ndarray[(3 * B * H,), np.dtype[bfloat16]]
    wt_ty = np.ndarray[(H * H,), np.dtype[bfloat16]]

    if mode == "grad_input":
        kernel = Kernel(
            "residual_grad_input_bf16",
            "resmlp_backward_grad_input_kernel.a",
            [gx_state_ty, wt_ty, act_ty],
        )

        state_in = ObjectFifo(gx_state_ty, name="state_in", depth=1)
        wt_in = ObjectFifo(wt_ty, name="wt_in", depth=1)
        gx_out = ObjectFifo(act_ty, name="gx_out", depth=1)

        def worker_fn(of_state, of_wt, of_out, kern):
            state = of_state.acquire(1)
            wt = of_wt.acquire(1)
            out = of_out.acquire(1)
            kern(state, wt, out)
            of_state.release(1)
            of_wt.release(1)
            of_out.release(1)

        worker = Worker(
            worker_fn,
            [state_in.cons(), wt_in.cons(), gx_out.prod(), kernel],
            placement=Tile(col=0, row=2),
            stack_size=0xC00,
        )

        rt = Runtime()
        with rt.sequence(gx_state_ty, wt_ty, act_ty) as (state, wt, gx):
            rt.start(worker)
            tg = rt.task_group()
            rt.fill(state_in.prod(), state, task_group=tg)
            rt.fill(wt_in.prod(), wt, task_group=tg)
            rt.drain(gx_out.cons(), gx, wait=True, task_group=tg)
            rt.finish_task_group(tg)

        return Program(NPU2(), rt).resolve_program(SequentialPlacer())

    if mode == "weight_grad":
        kernel = Kernel(
            "residual_weight_grad_bf16",
            "resmlp_backward_weight_grad_kernel.a",
            [dw_state_ty, wt_ty],
        )

        state_in = ObjectFifo(dw_state_ty, name="state_in", depth=1)
        dw_out = ObjectFifo(wt_ty, name="dw_out", depth=1)

        def worker_fn(of_state, of_out, kern):
            state = of_state.acquire(1)
            out = of_out.acquire(1)
            kern(state, out)
            of_state.release(1)
            of_out.release(1)

        worker = Worker(
            worker_fn,
            [state_in.cons(), dw_out.prod(), kernel],
            placement=Tile(col=0, row=2),
            stack_size=0xC00,
        )

        rt = Runtime()
        with rt.sequence(dw_state_ty, wt_ty) as (state, dw):
            rt.start(worker)
            tg = rt.task_group()
            rt.fill(state_in.prod(), state, task_group=tg)
            rt.drain(dw_out.cons(), dw, wait=True, task_group=tg)
            rt.finish_task_group(tg)

        return Program(NPU2(), rt).resolve_program(SequentialPlacer())

    raise ValueError(f"Unsupported backward mode: {mode}")
