"""Resident streaming operator for embed-on-NPU residual inference."""

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
from resmlp.streaming_embed_design import EMBED_INPUT_DIM


class StreamingEmbedResMLP(AIEOperatorBase):
    """Forward-only residual snake with embed on tile 0 and residual weights embedded."""

    def __init__(
        self,
        embed_packed,
        residual_packed_by_tile,
        *,
        H=32,
        B=8,
        num_cols=8,
        stream_depth=32,
        context=None,
    ):
        self.H = H
        self.B = B
        self.num_cols = num_cols
        self.num_tiles = num_cols * ROWS_PER_COL
        self.num_residual = self.num_tiles - 1
        self.stream_depth = stream_depth
        self._insts_synced = False

        embed_packed = np.asarray(embed_packed)
        expected_embed = (EMBED_INPUT_DIM * H,)
        if embed_packed.shape != expected_embed:
            raise ValueError(f"Expected embed_packed shape {expected_embed}, got {embed_packed.shape}")

        residual_packed_by_tile = np.asarray(residual_packed_by_tile)
        expected_residual = (self.num_residual, H * H)
        if residual_packed_by_tile.shape != expected_residual:
            raise ValueError(
                f"Expected residual_packed_by_tile shape {expected_residual}, got {residual_packed_by_tile.shape}"
            )

        self.embed_packed = embed_packed
        self.residual_packed_by_tile = residual_packed_by_tile
        self.weight_tag = sha1(
            self.embed_packed.tobytes() + self.residual_packed_by_tile.tobytes()
        ).hexdigest()[:10]
        self.weights_path = self._store_embedded_weights()

        super().__init__(context=context)

    def _store_embedded_weights(self):
        build_dir = Path(__file__).resolve().parent.parent / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        path = build_dir / (
            f"resmlp_streaming_embed_weights_{self.num_tiles}t_h{self.H}_{self.weight_tag}.npz"
        )
        if not path.exists():
            np.savez(
                path,
                embed=np.asarray(self.embed_packed).view(np.uint16),
                residual=np.asarray(self.residual_packed_by_tile).view(np.uint16),
            )
        return path

    def get_artifacts(self, prefix="resmlp_streaming_embed_"):
        operator_dir = Path(__file__).parent
        project_dir = operator_dir.parent
        kernels_dir = project_dir / "aie_kernels"
        H, B = self.H, self.B

        kernel_fp = source_fingerprint(
            kernels_dir / "embed_forward.cc",
            kernels_dir / "matmul_relu_skip.cc",
        )
        build_fp = source_fingerprint(
            operator_dir / "streaming_embed_design.py",
            operator_dir / "streaming_embed_op.py",
            kernels_dir / "embed_forward.cc",
            kernels_dir / "matmul_relu_skip.cc",
            self.weights_path,
        )
        kernel_tag = f"b{B}_h{H}_{kernel_fp}"
        archive_name = f"{prefix}kernel_{kernel_tag}.a"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{prefix}{B}x{H}_{self.num_cols}col_s{self.stream_depth}_{self.weight_tag}_{build_fp}.mlir",
            import_path=operator_dir / "streaming_embed_design.py",
            callback_fn="snake_streaming_embed_pipeline",
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

        residual_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_K={H}",
            f"-DDIM_N={H}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]
        embed_flags = [
            f"-DDIM_M={B}",
            "-DDIM_K_EMBED=56",
            f"-DDIM_H={H}",
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
                            f"embed_forward_{kernel_tag}.o",
                            extra_flags=embed_flags,
                            depends=[SourceArtifact.new(project_dir / "aie_kernels" / "embed_forward.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"residual_forward_{kernel_tag}.o",
                            extra_flags=residual_flags,
                            depends=[SourceArtifact.new(project_dir / "aie_kernels" / "matmul_relu_skip.cc")],
                        ),
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
        self.add_kernel(
            "resmlp_streaming_embed",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("input", self.stream_depth * self.B * EMBED_INPUT_DIM)
        self.add_buffer("output", self.stream_depth * self.B * self.H)

    def _run_args(self):
        return (
            self.buffer_bos["input"],
            self.buffer_bos["output"],
        )

    def _sync_buffers(self, names, direction):
        for name in names:
            self.buffer_bos[name].sync(direction)

    def run_stream(self):
        _, xrt_kernel, insts_bo, insts_len = self.xrt_kernels["resmlp_streaming_embed"]
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
                f"Kernel resmlp_streaming_embed did not complete correctly: {result}"
            )

        self._sync_buffers(["output"], pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return elapsed
