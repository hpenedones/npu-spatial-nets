"""Resident streaming operator for residual inference that emits logits."""

from hashlib import sha1
from pathlib import Path
import time

import numpy as np

from iron.common import (
    AIEOperatorBase,
    InstsBinArtifact,
    KernelArchiveArtifact,
    KernelObjectArtifact,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
    XclbinArtifact,
)
from iron.common.aie_device_manager import pyxrt

from resmlp.artifact_utils import source_fingerprint
from resmlp.design import ROWS_PER_COL
from resmlp.streaming_logits_design import N_CLS_PADDED


class StreamingResMLPLogits(AIEOperatorBase):
    """Forward-only residual snake with a fused head on the last tile."""

    def __init__(
        self,
        packed_weights_by_tile,
        packed_head_weight,
        packed_head_bias,
        *,
        H=160,
        B=8,
        num_cols=8,
        stream_depth=32,
        n_cls_padded=N_CLS_PADDED,
        context=None,
    ):
        self.H = H
        self.B = B
        self.num_cols = num_cols
        self.num_tiles = num_cols * ROWS_PER_COL
        self.stream_depth = stream_depth
        self.n_cls_padded = n_cls_padded
        self._insts_synced = False

        packed_weights_by_tile = np.asarray(packed_weights_by_tile)
        expected_residual = (self.num_tiles, H * H)
        if packed_weights_by_tile.shape != expected_residual:
            raise ValueError(
                f"Expected packed_weights_by_tile shape {expected_residual}, got {packed_weights_by_tile.shape}"
            )

        packed_head_weight = np.asarray(packed_head_weight)
        if packed_head_weight.shape != (H * self.n_cls_padded,):
            raise ValueError(
                f"Expected packed_head_weight shape {(H * self.n_cls_padded,)}, got {packed_head_weight.shape}"
            )

        packed_head_bias = np.asarray(packed_head_bias)
        if packed_head_bias.shape != (self.n_cls_padded,):
            raise ValueError(
                f"Expected packed_head_bias shape {(self.n_cls_padded,)}, got {packed_head_bias.shape}"
            )

        self.packed_weights_by_tile = packed_weights_by_tile
        self.packed_head_weight = packed_head_weight
        self.packed_head_bias = packed_head_bias
        self.weight_tag = sha1(
            self.packed_weights_by_tile.tobytes()
            + self.packed_head_weight.tobytes()
            + self.packed_head_bias.tobytes()
        ).hexdigest()[:10]
        self.weights_path = self._store_embedded_weights()

        super().__init__(context=context)

    def _store_embedded_weights(self):
        build_dir = Path(__file__).resolve().parent.parent / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        path = build_dir / (
            f"resmlp_streaming_logits_weights_{self.num_tiles}t_h{self.H}_c{self.n_cls_padded}_{self.weight_tag}.npz"
        )
        if not path.exists():
            np.savez(
                path,
                residual=np.asarray(self.packed_weights_by_tile).view(np.uint16),
                head_weight=np.asarray(self.packed_head_weight).view(np.uint16),
                head_bias=np.asarray(self.packed_head_bias).view(np.uint16),
            )
        return path

    def get_artifacts(self, prefix="resmlp_streaming_logits_"):
        operator_dir = Path(__file__).parent
        project_dir = operator_dir.parent
        kernels_dir = project_dir / "aie_kernels"
        H, B = self.H, self.B

        kernel_fp = source_fingerprint(
            kernels_dir / "matmul_relu_skip.cc",
            kernels_dir / "residual_head_infer.cc",
        )
        build_fp = source_fingerprint(
            operator_dir / "streaming_logits_design.py",
            operator_dir / "streaming_logits_op.py",
            kernels_dir / "matmul_relu_skip.cc",
            kernels_dir / "residual_head_infer.cc",
            self.weights_path,
        )
        kernel_tag = f"b{B}_h{H}_c{self.n_cls_padded}_{kernel_fp}"
        archive_name = f"{prefix}kernel_{kernel_tag}.a"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{prefix}{B}x{H}x{self.n_cls_padded}_{self.num_cols}col_s{self.stream_depth}_{self.weight_tag}_{build_fp}.mlir",
            import_path=operator_dir / "streaming_logits_design.py",
            callback_fn="snake_streaming_logits_pipeline",
            callback_kwargs={
                "H": H,
                "B": B,
                "num_cols": self.num_cols,
                "stream_depth": self.stream_depth,
                "archive_name": archive_name,
                "weights_path": str(self.weights_path),
                "n_cls_padded": self.n_cls_padded,
            },
            requires_context=False,
        )

        residual_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_K={H}",
            f"-DDIM_N={H}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]
        tail_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_H={H}",
            f"-DDIM_N_CLS={self.n_cls_padded}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        xclbin_artifact = XclbinArtifact.new(
            f"{prefix}{B}x{H}x{self.n_cls_padded}_{self.num_cols}col_s{self.stream_depth}_{self.weight_tag}_{build_fp}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    archive_name,
                    depends=[
                        KernelObjectArtifact.new(
                            f"residual_forward_{kernel_tag}.o",
                            extra_flags=residual_flags,
                            depends=[
                                SourceArtifact.new(project_dir / "aie_kernels" / "matmul_relu_skip.cc")
                            ],
                        ),
                        KernelObjectArtifact.new(
                            f"residual_head_{kernel_tag}.o",
                            extra_flags=tail_flags,
                            depends=[
                                SourceArtifact.new(project_dir / "aie_kernels" / "residual_head_infer.cc")
                            ],
                        ),
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{prefix}{B}x{H}x{self.n_cls_padded}_{self.num_cols}col_s{self.stream_depth}_{self.weight_tag}_{build_fp}.bin",
            depends=[mlir_artifact],
        )
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        xclbin, insts = self.get_artifacts()
        self.xclbin_artifact = xclbin
        self.insts_artifact = insts
        self.add_artifacts([xclbin, insts])

    def set_up_runtime(self):
        self.add_kernel(
            "resmlp_streaming_logits",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("input", self.stream_depth * self.B * self.H)
        self.add_buffer("output", self.stream_depth * self.B * self.n_cls_padded)

    def _run_args(self):
        return (
            self.buffer_bos["input"],
            self.buffer_bos["output"],
        )

    def _sync_buffers(self, names, direction):
        for name in names:
            self.buffer_bos[name].sync(direction)

    def run_stream(self):
        _, xrt_kernel, insts_bo, insts_len = self.xrt_kernels["resmlp_streaming_logits"]
        if not self._insts_synced:
            insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._insts_synced = True

        self._sync_buffers(["input"], pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        start = time.perf_counter()
        run = xrt_kernel(3, insts_bo, insts_len, *self._run_args())
        result = run.wait()
        elapsed = time.perf_counter() - start
        if result != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(
                f"Kernel resmlp_streaming_logits did not complete correctly: {result}"
            )

        self._sync_buffers(["output"], pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return elapsed
