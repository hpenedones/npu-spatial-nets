"""Resident streaming operator for forward-only residual MLP inference."""

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


class StreamingResMLP(AIEOperatorBase):
    """Forward-only residual snake with compile-time embedded inference weights."""

    def __init__(self, packed_weights_by_tile, H=160, B=8, num_cols=8, stream_depth=6, context=None):
        self.H = H
        self.B = B
        self.num_cols = num_cols
        self.num_tiles = num_cols * ROWS_PER_COL
        self.stream_depth = stream_depth
        self._insts_synced = False

        packed_weights_by_tile = np.asarray(packed_weights_by_tile)
        expected_shape = (self.num_tiles, H * H)
        if packed_weights_by_tile.shape != expected_shape:
            raise ValueError(
                f"Expected packed_weights_by_tile shape {expected_shape}, got {packed_weights_by_tile.shape}"
            )
        self.packed_weights_by_tile = packed_weights_by_tile
        self.weight_tag = sha1(self.packed_weights_by_tile.tobytes()).hexdigest()[:10]
        self.weights_path = self._store_embedded_weights()

        super().__init__(context=context)

    def _store_embedded_weights(self):
        build_dir = Path(__file__).resolve().parent.parent / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        path = build_dir / (
            f"resmlp_streaming_weights_{self.num_tiles}t_h{self.H}_{self.weight_tag}.npy"
        )
        if not path.exists():
            np.save(path, self.packed_weights_by_tile, allow_pickle=False)
        return path

    def get_artifacts(self, prefix="resmlp_streaming_"):
        operator_dir = Path(__file__).parent
        project_dir = operator_dir.parent
        kernels_dir = project_dir / "aie_kernels"
        H, B = self.H, self.B

        kernel_fp = source_fingerprint(kernels_dir / "matmul_relu_skip.cc")
        build_fp = source_fingerprint(
            operator_dir / "streaming_design.py",
            operator_dir / "streaming_op.py",
            kernels_dir / "matmul_relu_skip.cc",
            self.weights_path,
        )
        kernel_tag = f"b{B}_h{H}_{kernel_fp}"
        archive_name = f"{prefix}kernel_{kernel_tag}.a"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{prefix}{B}x{H}_{self.num_cols}col_s{self.stream_depth}_{self.weight_tag}_{build_fp}.mlir",
            import_path=operator_dir / "streaming_design.py",
            callback_fn="snake_streaming_pipeline",
            callback_kwargs={
                "H": H,
                "B": B,
                "num_cols": self.num_cols,
                "stream_depth": self.stream_depth,
                "archive_name": archive_name,
                "weights_path": str(self.weights_path),
            },
            requires_context=False,
        )

        kernel_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_K={H}",
            f"-DDIM_N={H}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        xclbin_artifact = XclbinArtifact.new(
            f"{prefix}{B}x{H}_{self.num_cols}col_s{self.stream_depth}_{self.weight_tag}_{build_fp}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    archive_name,
                    depends=[
                        KernelObjectArtifact.new(
                            f"{prefix}kernel_{kernel_tag}.o",
                            extra_flags=kernel_flags,
                            depends=[
                                SourceArtifact.new(
                                    project_dir / "aie_kernels" / "matmul_relu_skip.cc"
                                )
                            ],
                        )
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{prefix}{B}x{H}_{self.num_cols}col_s{self.stream_depth}_{self.weight_tag}_{build_fp}.bin",
            depends=[mlir_artifact],
        )
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        xclbin, insts = self.get_artifacts()
        self.xclbin_artifact = xclbin
        self.insts_artifact = insts
        self.add_artifacts([xclbin, insts])

    def set_up_runtime(self):
        H, B = self.H, self.B
        self.add_kernel(
            "resmlp_streaming",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("input", self.stream_depth * B * H)
        self.add_buffer("output", self.stream_depth * B * H)

    def _run_args(self):
        return (
            self.buffer_bos["input"],
            self.buffer_bos["output"],
        )

    def _sync_buffers(self, names, direction):
        for name in names:
            self.buffer_bos[name].sync(direction)

    def run_stream(self):
        _, xrt_kernel, insts_bo, insts_len = self.xrt_kernels["resmlp_streaming"]
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
                f"Kernel resmlp_streaming did not complete correctly: {result}"
            )

        self._sync_buffers(["output"], pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return elapsed
