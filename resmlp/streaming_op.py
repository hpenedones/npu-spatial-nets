"""Resident streaming operator for the full-NPU residual MLP inference pipeline."""

from hashlib import sha1
from pathlib import Path
import time

import numpy as np

from resmlp import round_up_to_tile_multiple
from resmlp.xrt_env import ensure_xrt_python_path

ensure_xrt_python_path()

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
    """Forward-only residual MLP inference with compile-time embedded weights."""

    def __init__(
        self,
        packed_weights_by_tile,
        H=160,
        B=8,
        num_cols=8,
        stream_depth=32,
        context=None,
        input_dim=None,
        output_dim=None,
        packed_embed_weight=None,
        packed_embed_bias=None,
        packed_head_weight=None,
        packed_head_bias=None,
    ):
        self.H = H
        self.B = B
        self.num_cols = num_cols
        self.num_tiles = num_cols * ROWS_PER_COL
        self.stream_depth = stream_depth
        self._insts_synced = False

        if input_dim is None or output_dim is None:
            raise ValueError("input_dim and output_dim are required")
        self.actual_input_dim = input_dim
        self.actual_output_dim = output_dim
        self.input_dim_device = round_up_to_tile_multiple(self.actual_input_dim)
        self.output_dim_device = round_up_to_tile_multiple(self.actual_output_dim)

        packed_weights_by_tile = np.asarray(packed_weights_by_tile)
        expected_residual_tiles = self.num_tiles - 2
        expected_residual_shape = (expected_residual_tiles, H * H)
        if packed_weights_by_tile.shape != expected_residual_shape:
            raise ValueError(
                f"Expected packed_weights_by_tile shape {expected_residual_shape}, "
                f"got {packed_weights_by_tile.shape}"
            )
        self.packed_weights_by_tile = packed_weights_by_tile

        self.packed_embed_weight = self._validated_array(
            packed_embed_weight,
            (self.input_dim_device * H,),
            "packed_embed_weight",
        )
        self.packed_embed_bias = self._validated_array(
            packed_embed_bias,
            (H * 8,),
            "packed_embed_bias",
        )
        self.packed_head_weight = self._validated_array(
            packed_head_weight,
            (H * self.output_dim_device,),
            "packed_head_weight",
        )
        self.packed_head_bias = self._validated_array(
            packed_head_bias,
            (self.output_dim_device * 8,),
            "packed_head_bias",
        )

        payloads = [
            self.packed_weights_by_tile.tobytes(),
            self.packed_embed_weight.tobytes(),
            self.packed_embed_bias.tobytes(),
            self.packed_head_weight.tobytes(),
            self.packed_head_bias.tobytes(),
        ]
        self.weight_tag = sha1(b"".join(payloads)).hexdigest()[:10]

        self.residual_weights_path = self._store_embedded_array("residual_weights", self.packed_weights_by_tile)
        self.embed_weights_path = self._store_embedded_array("embed_weights", self.packed_embed_weight)
        self.embed_bias_path = self._store_embedded_array("embed_bias", self.packed_embed_bias)
        self.head_weights_path = self._store_embedded_array("head_weights", self.packed_head_weight)
        self.head_bias_path = self._store_embedded_array("head_bias", self.packed_head_bias)

        super().__init__(context=context)

    @staticmethod
    def _validated_array(array, expected_shape, label):
        if array is None:
            raise ValueError(f"{label} is required")
        validated = np.asarray(array)
        if validated.shape != expected_shape:
            raise ValueError(f"Expected {label} shape {expected_shape}, got {validated.shape}")
        return validated

    def _store_embedded_array(self, stem, array):
        build_dir = Path(__file__).resolve().parent.parent / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        path = build_dir / (
            f"resmlp_streaming_{stem}_{self.num_tiles}t_b{self.B}_h{self.H}_{self.weight_tag}.npy"
        )
        if not path.exists():
            np.save(path, np.asarray(array), allow_pickle=False)
        return path

    def get_artifacts(self, prefix="resmlp_streaming_"):
        operator_dir = Path(__file__).parent
        project_dir = operator_dir.parent
        kernels_dir = project_dir / "aie_kernels"
        H, B = self.H, self.B

        residual_kernel_fp = source_fingerprint(kernels_dir / "matmul_relu_skip.cc")
        linear_kernel_fp = source_fingerprint(kernels_dir / "matmul_bias.cc")
        build_inputs = [
            operator_dir / "streaming_design.py",
            operator_dir / "streaming_op.py",
            kernels_dir / "matmul_relu_skip.cc",
            kernels_dir / "matmul_bias.cc",
            self.residual_weights_path,
            self.embed_weights_path,
            self.embed_bias_path,
            self.head_weights_path,
            self.head_bias_path,
        ]

        residual_kernel_tag = f"body_b{B}_h{H}_{residual_kernel_fp}"
        residual_archive_name = f"{prefix}body_{residual_kernel_tag}.a"
        residual_kernel_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_K={H}",
            f"-DDIM_N={H}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]
        linear_source = SourceArtifact.new(project_dir / "aie_kernels" / "matmul_bias.cc")
        embed_kernel_tag = f"embed_b{B}_k{self.input_dim_device}_n{H}_{linear_kernel_fp}"
        head_kernel_tag = f"head_b{B}_k{H}_n{self.output_dim_device}_{linear_kernel_fp}"
        embed_archive_name = f"{prefix}embed_{embed_kernel_tag}.a"
        head_archive_name = f"{prefix}head_{head_kernel_tag}.a"

        archive_artifacts = [
            KernelArchiveArtifact.new(
                residual_archive_name,
                depends=[
                    KernelObjectArtifact.new(
                        f"{prefix}body_{residual_kernel_tag}.o",
                        extra_flags=residual_kernel_flags,
                        depends=[SourceArtifact.new(project_dir / "aie_kernels" / "matmul_relu_skip.cc")],
                    )
                ],
            ),
            KernelArchiveArtifact.new(
                embed_archive_name,
                depends=[
                    KernelObjectArtifact.new(
                        f"{prefix}embed_{embed_kernel_tag}.o",
                        extra_flags=[
                            f"-DDIM_M={B}",
                            f"-DDIM_K={self.input_dim_device}",
                            f"-DDIM_N={H}",
                            "-DMATMUL_BIAS_ENTRY_NAME=matmul_bias_embed_bf16",
                            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
                        ],
                        depends=[linear_source],
                    )
                ],
            ),
            KernelArchiveArtifact.new(
                head_archive_name,
                depends=[
                    KernelObjectArtifact.new(
                        f"{prefix}head_{head_kernel_tag}.o",
                        extra_flags=[
                            f"-DDIM_M={B}",
                            f"-DDIM_K={H}",
                            f"-DDIM_N={self.output_dim_device}",
                            "-DMATMUL_BIAS_ENTRY_NAME=matmul_bias_head_bf16",
                            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
                        ],
                        depends=[linear_source],
                    )
                ],
            ),
        ]

        callback_kwargs = {
            "H": H,
            "B": B,
            "num_cols": self.num_cols,
            "stream_depth": self.stream_depth,
            "residual_archive_name": residual_archive_name,
            "residual_weights_path": str(self.residual_weights_path),
            "input_dim_device": self.input_dim_device,
            "output_dim_device": self.output_dim_device,
            "embed_archive_name": embed_archive_name,
            "embed_weights_path": str(self.embed_weights_path),
            "embed_bias_path": str(self.embed_bias_path),
            "head_archive_name": head_archive_name,
            "head_weights_path": str(self.head_weights_path),
            "head_bias_path": str(self.head_bias_path),
        }

        build_fp = source_fingerprint(*build_inputs)
        input_tag = f"in{self.input_dim_device}_out{self.output_dim_device}"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{prefix}{B}x{H}_{self.num_cols}col_s{self.stream_depth}_{input_tag}_{self.weight_tag}_{build_fp}.mlir",
            import_path=operator_dir / "streaming_design.py",
            callback_fn="snake_streaming_pipeline",
            callback_kwargs=callback_kwargs,
            requires_context=False,
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{prefix}{B}x{H}_{self.num_cols}col_s{self.stream_depth}_{input_tag}_{self.weight_tag}_{build_fp}.xclbin",
            depends=[mlir_artifact, *archive_artifacts],
        )
        insts_artifact = InstsBinArtifact.new(
            f"{prefix}{B}x{H}_{self.num_cols}col_s{self.stream_depth}_{input_tag}_{self.weight_tag}_{build_fp}.bin",
            depends=[mlir_artifact],
        )
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        xclbin, insts = self.get_artifacts()
        self.xclbin_artifact = xclbin
        self.insts_artifact = insts
        self.add_artifacts([xclbin, insts])

    NUM_SLOTS = 2

    def set_up_runtime(self):
        self.add_kernel(
            "resmlp_streaming",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        input_elems = self.stream_depth * self.B * self.input_dim_device
        output_elems = self.stream_depth * self.B * self.output_dim_device
        for slot in range(self.NUM_SLOTS):
            self.add_buffer(f"input_{slot}", input_elems)
            self.add_buffer(f"output_{slot}", output_elems)

    def _run_args(self, slot):
        return (
            self.buffer_bos[f"input_{slot}"],
            self.buffer_bos[f"output_{slot}"],
        )

    def _sync_buffers(self, names, direction):
        for name in names:
            self.buffer_bos[name].sync(direction)

    def write_input_slot(self, slot, array):
        self.write_buffer(f"input_{slot}", array)

    def read_output_slot(self, slot, shape, copy=True):
        return self.read_buffer(f"output_{slot}", shape, copy=copy)

    def sync_input_h2d(self, slot):
        self.buffer_bos[f"input_{slot}"].sync(
            pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        )

    def sync_output_d2h(self, slot):
        self.buffer_bos[f"output_{slot}"].sync(
            pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        )

    def _ensure_insts_synced(self):
        if not self._insts_synced:
            _, _, insts_bo, _ = self.xrt_kernels["resmlp_streaming"]
            insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._insts_synced = True

    def dispatch(self, slot):
        """Non-blocking kernel dispatch. Returns an XRT Run handle to wait on."""
        _, xrt_kernel, insts_bo, insts_len = self.xrt_kernels["resmlp_streaming"]
        self._ensure_insts_synced()
        return xrt_kernel(3, insts_bo, insts_len, *self._run_args(slot))

    def run_stream(self, timings=None, slot=0):
        """Legacy synchronous single-slot path (kept for smoke tests and eval fallback)."""
        self._ensure_insts_synced()

        h2d_t0 = time.perf_counter()
        self.sync_input_h2d(slot)
        h2d_s = time.perf_counter() - h2d_t0

        start = time.perf_counter()
        run = self.dispatch(slot)
        result = run.wait()
        elapsed = time.perf_counter() - start
        if result != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(
                f"Kernel resmlp_streaming did not complete correctly: {result}"
            )

        d2h_t0 = time.perf_counter()
        self.sync_output_d2h(slot)
        d2h_s = time.perf_counter() - d2h_t0

        if timings is not None:
            timings["h2d_sync_s"] = timings.get("h2d_sync_s", 0.0) + h2d_s
            timings["d2h_sync_s"] = timings.get("d2h_sync_s", 0.0) + d2h_s
        return elapsed
