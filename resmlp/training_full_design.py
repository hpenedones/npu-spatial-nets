"""
IRON design for 100% on-NPU MNIST training pipeline (H=32).

Tile layout (32 tiles = 1 embed + 30 residual + 1 head):
    Tile 0  (col 0, row 2):  Embed — 784 → 32 matmul
    Tiles 1–30 (snake):       Residual — relu(x @ W) + x
    Tile 31 (col 7, row 5):  Head — 32 → 16 matmul + softmax + CE loss

Forward:
    Host → x_raw[8×784] → [Embed] → act → [Res 0..29] → [Head] → loss → Host
    Host → labels[8] → [Head]

Backward:
    [Head] → gy[8×32] → [Res 29..0] → gy[8×32] → [Embed]
    Host → x_raw[8×784] → [Embed]  (re-stream for embed grad)
"""

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib.tap import TensorAccessPattern

ROWS_PER_COL = 4
NUM_RESIDUAL = 30
N_CLS_PADDED = 16


def snake_tile_order(num_cols):
    tiles = []
    for col in range(num_cols):
        rows = range(2, 6) if col % 2 == 0 else range(5, 1, -1)
        for row in rows:
            tiles.append((col, row))
    return tiles


def full_training_pipeline(H=32, B=8, K_EMBED=784, num_cols=8):
    assert num_cols == 8 and B % 8 == 0 and H % 8 == 0 and K_EMBED % 8 == 0
    num_tiles = num_cols * ROWS_PER_COL
    assert num_tiles == NUM_RESIDUAL + 2
    tile_order = snake_tile_order(num_cols)
    # ── Types ──────────────────────────────────────────────────────
    act_ty     = np.ndarray[(B * H,),             np.dtype[bfloat16]]
    embed_in_ty = np.ndarray[(B * K_EMBED,),      np.dtype[bfloat16]]
    embed_wt_ty = np.ndarray[(K_EMBED * H,),      np.dtype[bfloat16]]
    res_wt_ty  = np.ndarray[(H * H,),             np.dtype[bfloat16]]
    head_wt_ty = np.ndarray[(H * N_CLS_PADDED,),  np.dtype[bfloat16]]
    ckpt_ty    = np.ndarray[(2 * B * H,),         np.dtype[bfloat16]]
    labels_ty  = np.ndarray[(B,),                 np.dtype[np.int32]]
    d_logits_ty = np.ndarray[(B * N_CLS_PADDED,), np.dtype[bfloat16]]

    # ── Kernels ────────────────────────────────────────────────────
    embed_fwd_k  = Kernel("embed_forward_bf16",  "full_training_kernels.a",
                          [embed_in_ty, embed_wt_ty, act_ty])
    embed_bwd_k  = Kernel("embed_backward_bf16", "full_training_kernels.a",
                          [embed_in_ty, embed_wt_ty, act_ty])
    res_fwd_k    = Kernel("matmul_relu_skip_bf16", "full_training_kernels.a",
                          [act_ty, res_wt_ty, act_ty, ckpt_ty, np.int32])
    res_copy_k   = Kernel("copy_activation_bf16", "full_training_kernels.a",
                          [act_ty, act_ty])
    res_bwd_k    = Kernel("residual_backward_and_update_bf16", "full_training_kernels.a",
                          [ckpt_ty, res_wt_ty, act_ty, act_ty])
    head_fwd_k   = Kernel("head_forward_loss_bf16", "full_training_kernels.a",
                          [act_ty, head_wt_ty, labels_ty, d_logits_ty, labels_ty])
    head_bwd_k   = Kernel("head_backward_bf16", "full_training_kernels.a",
                          [act_ty, head_wt_ty, d_logits_ty, act_ty])
    head_wt_copy_k = Kernel("copy_head_weight_bf16", "full_training_kernels.a",
                            [head_wt_ty, head_wt_ty])

    # ── Weight ObjectFifos ─────────────────────────────────────────
    embed_wt_fifo = ObjectFifo(embed_wt_ty, name="embed_wt", depth=1)
    head_wt_fifo  = ObjectFifo(head_wt_ty,  name="head_wt",  depth=1)
    head_wt_out = ObjectFifo(head_wt_ty, name="head_wt_out", depth=1)

    # Residual weights split per column:
    #   col 0: 3 res tiles, cols 1-6: 4 each, col 7: 3 res tiles
    res_per_col = [3, 4, 4, 4, 4, 4, 4, 3]
    res_wt_ddrs = []
    res_wt_endpoints = []
    for col_idx, n_res in enumerate(res_per_col):
        col_wt_ty = np.ndarray[(n_res * H * H,), np.dtype[bfloat16]]
        wt_ddr = ObjectFifo(col_wt_ty, name=f"res_wt_col{col_idx}", depth=1)
        splits = wt_ddr.cons().split(
            offsets=[H * H * r for r in range(n_res)],
            obj_types=[res_wt_ty] * n_res,
            names=[f"res_wt_{col_idx}_{r}" for r in range(n_res)],
            depths=[1] * n_res,
            placement=Tile(col=col_idx, row=1),
        )
        res_wt_ddrs.append(wt_ddr)
        for r in range(n_res):
            res_wt_endpoints.append(splits[r])

    # ── Activation chain (forward) ─────────────────────────────────
    embed_in = ObjectFifo(embed_in_ty, name="embed_in", depth=1)
    act_fifos = [ObjectFifo(act_ty, name=f"act_{i}", depth=1)
                 for i in range(NUM_RESIDUAL + 1)]

    # ── Labels / Loss ──────────────────────────────────────────────
    labels_fifo = ObjectFifo(labels_ty, name="labels", depth=1)
    preds_fifo  = ObjectFifo(labels_ty, name="preds",  depth=1)
    done_fifo   = ObjectFifo(np.ndarray[(1,), np.dtype[np.int32]], name="done", depth=1)

    # ── Gradient chain (backward) ──────────────────────────────────
    grad_fifos = [ObjectFifo(act_ty, name=f"grad_{i}", depth=1)
                  for i in range(NUM_RESIDUAL + 1)]

    # ═══════════════════════════════════════════════════════════════
    # WORKERS
    # ═══════════════════════════════════════════════════════════════
    workers = []

    # ── Embed tile (tile 0) ────────────────────────────────────────
    embed_col, embed_row = tile_order[0]
    embed_wt_ep = embed_wt_fifo.cons()
    embed_wt_cons = embed_wt_ep.cons() if hasattr(embed_wt_ep, 'cons') else embed_wt_ep

    def make_embed_worker():
        def w(of_x_in, of_y_out, of_wt, of_grad_in, of_done,
              fwd_k, bwd_k):
            x = of_x_in.acquire(1)
            y = of_y_out.acquire(1)
            wt = of_wt.acquire(1)
            fwd_k(x, wt, y)
            of_x_in.release(1)
            of_y_out.release(1)

            dy = of_grad_in.acquire(1)
            x2 = of_x_in.acquire(1)
            bwd_k(x2, wt, dy)
            done = of_done.acquire(1)
            done[0] = 1
            of_grad_in.release(1)
            of_x_in.release(1)
            of_done.release(1)
            of_wt.release(1)
        return w

    workers.append(Worker(
        make_embed_worker(),
        [embed_in.cons(), act_fifos[0].prod(), embed_wt_cons,
         grad_fifos[NUM_RESIDUAL].cons(), done_fifo.prod(),
         embed_fwd_k, embed_bwd_k],
        placement=Tile(col=embed_col, row=embed_row),
        stack_size=0x400,
    ))

    # ── Residual tiles (tiles 1..30) ───────────────────────────────
    for res_idx in range(NUM_RESIDUAL):
        col, row = tile_order[res_idx + 1]
        local_ckpt = ObjectFifo(ckpt_ty, name=f"ckpt_{res_idx}", depth=1)

        in_ep  = act_fifos[res_idx].cons()
        out_ep = act_fifos[res_idx + 1].prod()
        wt_ep  = res_wt_endpoints[res_idx]
        wt_c   = wt_ep.cons() if hasattr(wt_ep, 'cons') else wt_ep
        g_in   = grad_fifos[NUM_RESIDUAL - 1 - res_idx].cons()
        g_out  = grad_fifos[NUM_RESIDUAL - res_idx].prod()

        def make_res(ie, oe, we, gi, go):
            def w(of_in, of_out, of_w, of_gin, of_gout,
                  of_cp, of_cr, cp_k, fwd_k, bwd_k):
                x = of_in.acquire(1)
                y = of_out.acquire(1)
                ww = of_w.acquire(1)
                ck = of_cp.acquire(1)
                cp_k(x, ck[0:B*H])
                fwd_k(x, ww, y, ck, B * H)
                of_in.release(1)
                of_out.release(1)
                of_cp.release(1)

                gy = of_gin.acquire(1)
                gx = of_gout.acquire(1)
                cr = of_cr.acquire(1)
                bwd_k(cr, ww, gy, gx)
                of_gin.release(1)
                of_gout.release(1)
                of_cr.release(1)
                of_w.release(1)
            return w

        workers.append(Worker(
            make_res(in_ep, out_ep, wt_c, g_in, g_out),
            [in_ep, out_ep, wt_c, g_in, g_out,
             local_ckpt.prod(), local_ckpt.cons(),
             res_copy_k, res_fwd_k, res_bwd_k],
            placement=Tile(col=col, row=row),
            stack_size=0x1000,
        ))

    # ── Head tile (tile 31) ────────────────────────────────────────
    head_col, head_row = tile_order[31]
    head_ckpt_fifo = ObjectFifo(act_ty, name="head_ckpt", depth=1)
    d_logits_fifo  = ObjectFifo(d_logits_ty, name="d_logits", depth=1)
    head_wt_ep = head_wt_fifo.cons()
    head_wt_cons = head_wt_ep.cons() if hasattr(head_wt_ep, 'cons') else head_wt_ep

    def make_head():
        def w(of_yin, of_wt, of_lab,
              of_wt_out, of_cp, of_cr, of_dlp, of_dlc, of_gout, of_preds,
              cp_k, fwd_k, bwd_k, copy_wt_k):
            y = of_yin.acquire(1)
            wt = of_wt.acquire(1)
            lb = of_lab.acquire(1)
            ck = of_cp.acquire(1)
            dl = of_dlp.acquire(1)
            pred = of_preds.acquire(1)

            cp_k(y, ck)
            fwd_k(y, wt, lb, dl, pred)
            of_yin.release(1)
            of_lab.release(1)
            of_cp.release(1)
            of_dlp.release(1)
            of_preds.release(1)

            cr = of_cr.acquire(1)
            dr = of_dlc.acquire(1)
            gy = of_gout.acquire(1)
            bwd_k(cr, wt, dr, gy)
            wt_out = of_wt_out.acquire(1)
            copy_wt_k(wt, wt_out)
            of_cr.release(1)
            of_dlc.release(1)
            of_gout.release(1)
            of_wt_out.release(1)
            of_wt.release(1)
        return w

    workers.append(Worker(
        make_head(),
        [act_fifos[NUM_RESIDUAL].cons(), head_wt_cons,
         labels_fifo.cons(),
         head_wt_out.prod(),
         head_ckpt_fifo.prod(), head_ckpt_fifo.cons(),
         d_logits_fifo.prod(), d_logits_fifo.cons(),
         grad_fifos[0].prod(), preds_fifo.prod(),
         res_copy_k, head_fwd_k, head_bwd_k, head_wt_copy_k],
        placement=Tile(col=head_col, row=head_row),
        stack_size=0x4000,
    ))

    # ═══════════════════════════════════════════════════════════════
    # RUNTIME SEQUENCE
    # ═══════════════════════════════════════════════════════════════
    host_embed_in_ty = np.ndarray[(B * K_EMBED,),          np.dtype[bfloat16]]
    host_embed_wt_ty = np.ndarray[(K_EMBED * H,),          np.dtype[bfloat16]]
    host_res_wt_ty   = np.ndarray[(NUM_RESIDUAL * H * H,), np.dtype[bfloat16]]
    host_head_wt_ty  = np.ndarray[(H * N_CLS_PADDED,),     np.dtype[bfloat16]]
    host_labels_ty   = np.ndarray[(2 * B,),                np.dtype[np.int32]]
    host_done_ty     = np.ndarray[(1,),                    np.dtype[np.int32]]

    rt = Runtime()
    with rt.sequence(
        host_embed_in_ty, host_embed_wt_ty, host_res_wt_ty,
        host_head_wt_ty, host_labels_ty, host_done_ty,
    ) as (x_raw, wt_embed, wt_res, wt_head, labels, done_out):
        rt.start(*workers)

        tg_fwd = rt.task_group()
        rt.fill(embed_wt_fifo.prod(), wt_embed, task_group=tg_fwd)
        rt.fill(head_wt_fifo.prod(),  wt_head,  task_group=tg_fwd)

        res_wt_offset = 0
        for n_res, wt_ddr in zip(res_per_col, res_wt_ddrs):
            col_elems = n_res * H * H
            tap = TensorAccessPattern(
                (1, NUM_RESIDUAL * H * H),
                res_wt_offset,
                [1, n_res, H, H],
                [0, H * H, H, 1],
            )
            rt.fill(wt_ddr.prod(), wt_res, tap, task_group=tg_fwd)
            res_wt_offset += col_elems

        labels_in_tap = TensorAccessPattern((1, 2 * B), 0, [1, B], [0, 1])
        preds_out_tap = TensorAccessPattern((1, 2 * B), B, [1, B], [0, 1])
        rt.fill(embed_in.prod(),    x_raw,  task_group=tg_fwd)
        rt.fill(labels_fifo.prod(), labels, tap=labels_in_tap, task_group=tg_fwd)
        rt.drain(preds_fifo.cons(), labels, tap=preds_out_tap, wait=True, task_group=tg_fwd)
        rt.finish_task_group(tg_fwd)

        tg_bwd = rt.task_group()
        rt.fill(embed_in.prod(), x_raw, task_group=tg_bwd)
        rt.drain(done_fifo.cons(), done_out, wait=True, task_group=tg_bwd)
        rt.drain(head_wt_out.cons(), wt_head, wait=True, task_group=tg_bwd)
        rt.finish_task_group(tg_bwd)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())
