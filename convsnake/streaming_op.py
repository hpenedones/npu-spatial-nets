"""Resident streaming operator for the convolutional snake inference path."""

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

from convsnake.config import BATCH_SIZE, DEFAULT_CONFIG, ConvSnakeConfig, ROWS_PER_COL, num_blocks_for_cols
from resmlp.artifact_utils import source_fingerprint


class StreamingConvSnake(AIEOperatorBase):
    def __init__(
        self,
        embedded_weights,
        *,
        config: ConvSnakeConfig | None = None,
        B=BATCH_SIZE,
        num_cols=8,
        stream_depth=32,
        context=None,
    ):
        self.cfg = DEFAULT_CONFIG if config is None else config
        self.B = B
        self.num_cols = num_cols
        self.num_tiles = num_cols * ROWS_PER_COL
        self.num_blocks = num_blocks_for_cols(num_cols)
        self.stream_depth = stream_depth
        self._insts_synced = False

        if self.B != self.cfg.batch_size:
            raise ValueError(f"Batch size must be {self.cfg.batch_size}")

        self.embedded_weights = self._normalize_weights(embedded_weights)
        self.weight_tag = sha1(self._weight_bytes()).hexdigest()[:10]
        self.weights_path = self._store_embedded_weights()

        super().__init__(context=context)

    def _normalize_weights(self, embedded_weights):
        weights = {
            "conv1": np.asarray(embedded_weights["conv1"]).reshape(self.cfg.conv1_npu_w_elems),
            "conv2": np.asarray(embedded_weights["conv2"]).reshape(self.cfg.conv2_npu_w_elems),
            "conv3": np.asarray(embedded_weights["conv3"]).reshape(self.cfg.conv3_npu_w_elems),
            "blocks": np.asarray(embedded_weights["blocks"]).reshape(
                self.num_blocks, self.cfg.block_npu_w_elems
            ),
            "head": np.asarray(embedded_weights["head"]).reshape(self.cfg.head_w_elems),
        }
        return weights

    def _weight_bytes(self) -> bytes:
        return b"".join(
            np.asarray(self.embedded_weights[key]).tobytes()
            for key in ("conv1", "conv2", "conv3", "blocks", "head")
        )

    def _store_embedded_weights(self):
        build_dir = Path(__file__).resolve().parent.parent / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        path = build_dir / (
            f"convsnake_{self.cfg.dataset}_streaming_weights_{self.num_tiles}t_s{self.stream_depth}_{self.weight_tag}.npz"
        )
        if not path.exists():
            np.savez(
                path,
                conv1=np.asarray(self.embedded_weights["conv1"]).view(np.uint16),
                conv2=np.asarray(self.embedded_weights["conv2"]).view(np.uint16),
                conv3=np.asarray(self.embedded_weights["conv3"]).view(np.uint16),
                blocks=np.asarray(self.embedded_weights["blocks"]).view(np.uint16),
                head=np.asarray(self.embedded_weights["head"]).view(np.uint16),
            )
        return path

    def _conv1_flags(self):
        return [
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            f"-DBATCH_SIZE={self.cfg.batch_size}",
            f"-DIN_H={self.cfg.img_h}",
            f"-DIN_W={self.cfg.img_w}",
            f"-DIN_C={self.cfg.img_c}",
            f"-DOUT_H={self.cfg.conv1_out_h}",
            f"-DOUT_W={self.cfg.conv1_out_w}",
            f"-DOUT_C={self.cfg.c1}",
            f"-DOUT_C_PAD={self.cfg.conv1_npu_out_c}",
            f"-DK_PADDED={self.cfg.conv1_npu_k}",
            f"-DPOSITIONS_CHUNK={self.cfg.conv1_positions_chunk}",
        ]

    def _conv2_flags(self):
        return [
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            f"-DBATCH_SIZE={self.cfg.batch_size}",
            f"-DIN_H={self.cfg.conv1_out_h}",
            f"-DIN_W={self.cfg.conv1_out_w}",
            f"-DIN_C={self.cfg.c1}",
            f"-DOUT_H={self.cfg.conv2_out_h}",
            f"-DOUT_W={self.cfg.conv2_out_w}",
            f"-DOUT_C={self.cfg.c2}",
            f"-DK_PADDED={self.cfg.conv2_npu_k}",
            f"-DPOSITIONS_CHUNK={self.cfg.conv2_positions_chunk}",
        ]

    def _conv3_flags(self):
        return [
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            f"-DBATCH_SIZE={self.cfg.batch_size}",
            f"-DIN_H={self.cfg.conv2_out_h}",
            f"-DIN_W={self.cfg.conv2_out_w}",
            f"-DIN_C={self.cfg.c2}",
            f"-DOUT_H={self.cfg.conv3_out_h}",
            f"-DOUT_W={self.cfg.conv3_out_w}",
            f"-DOUT_C={self.cfg.c3}",
            f"-DPOSITIONS_CHUNK={self.cfg.conv3_positions_chunk}",
        ]

    def get_artifacts(self, prefix="convsnake_streaming_"):
        operator_dir = Path(__file__).parent
        project_dir = operator_dir.parent
        kernels_dir = project_dir / "aie_kernels"

        kernel_fp = source_fingerprint(
            kernels_dir / "conv1_infer.cc",
            kernels_dir / "conv2_infer.cc",
            kernels_dir / "conv3_infer.cc",
            kernels_dir / "conv4_infer.cc",
            kernels_dir / "flatten_head_infer.cc",
        )
        build_fp = source_fingerprint(
            operator_dir / "config.py",
            operator_dir / "streaming_design.py",
            operator_dir / "streaming_op.py",
            kernels_dir / "conv1_infer.cc",
            kernels_dir / "conv2_infer.cc",
            kernels_dir / "conv3_infer.cc",
            kernels_dir / "conv4_infer.cc",
            kernels_dir / "flatten_head_infer.cc",
            self.weights_path,
        )
        archive_name = f"{prefix}{self.cfg.dataset}_kernel_{self.weight_tag}_{kernel_fp}.a"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{prefix}{self.cfg.dataset}_{self.num_tiles}t_s{self.stream_depth}_{self.weight_tag}_{build_fp}.mlir",
            import_path=operator_dir / "streaming_design.py",
            callback_fn="convsnake_streaming_pipeline",
            callback_kwargs={
                "B": self.B,
                "num_cols": self.num_cols,
                "stream_depth": self.stream_depth,
                "archive_name": archive_name,
                "weights_path": str(self.weights_path),
                "config_kwargs": self.cfg.to_dict(),
            },
            requires_context=False,
        )

        head_flags = [
            f"-DBATCH_SIZE={self.B}",
            f"-DIN_DIM={self.cfg.flat_dim}",
            f"-DNUM_CLASSES={self.cfg.num_classes}",
        ]

        xclbin_artifact = XclbinArtifact.new(
            f"{prefix}{self.cfg.dataset}_{self.num_tiles}t_s{self.stream_depth}_{self.weight_tag}_{build_fp}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    archive_name,
                    depends=[
                        KernelObjectArtifact.new(
                            f"conv1_infer_{self.cfg.dataset}_{kernel_fp}.o",
                            extra_flags=self._conv1_flags(),
                            depends=[SourceArtifact.new(kernels_dir / "conv1_infer.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"conv2_infer_{self.cfg.dataset}_{kernel_fp}.o",
                            extra_flags=self._conv2_flags(),
                            depends=[SourceArtifact.new(kernels_dir / "conv2_infer.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"conv3_infer_{self.cfg.dataset}_{kernel_fp}.o",
                            extra_flags=self._conv3_flags(),
                            depends=[SourceArtifact.new(kernels_dir / "conv3_infer.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"conv4_infer_{self.cfg.dataset}_b{self.B}_{kernel_fp}.o",
                            extra_flags=[
                                "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
                                f"-DBATCH_SIZE={self.B}",
                            ],
                            depends=[SourceArtifact.new(kernels_dir / "conv4_infer.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"flatten_head_infer_{self.cfg.dataset}_{kernel_fp}.o",
                            extra_flags=head_flags,
                            depends=[SourceArtifact.new(kernels_dir / "flatten_head_infer.cc")],
                        ),
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{prefix}{self.cfg.dataset}_{self.num_tiles}t_s{self.stream_depth}_{self.weight_tag}_{build_fp}.bin",
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
            "convsnake_streaming",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("input", self.stream_depth * self.cfg.img_elems)
        self.add_buffer("output", self.stream_depth * self.cfg.logits_elems)

    def _run_args(self):
        return (
            self.buffer_bos["input"],
            self.buffer_bos["output"],
        )

    def _sync_buffers(self, names, direction):
        for name in names:
            self.buffer_bos[name].sync(direction)

    def run_stream(self):
        _, xrt_kernel, insts_bo, insts_len = self.xrt_kernels["convsnake_streaming"]
        if not self._insts_synced:
            insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._insts_synced = True

        self._sync_buffers(["input"], pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        start = time.perf_counter()
        run = xrt_kernel(3, insts_bo, insts_len, *self._run_args())
        result = run.wait()
        elapsed = time.perf_counter() - start
        if result != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"Kernel convsnake_streaming did not complete correctly: {result}")

        self._sync_buffers(["output"], pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return elapsed
