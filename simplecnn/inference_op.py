from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from iron.common import (
    AIEOperatorBase,
    InstsBinArtifact,
    KernelArchiveArtifact,
    KernelObjectArtifact,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
    XclbinArtifact,
)

from resmlp.artifact_utils import source_fingerprint
from simplecnn.config import (
    BATCH_SIZE,
    C3,
    CONV3_OUT_H,
    CONV3_OUT_W,
    IMG_ELEMS,
    N_CLASSES,
    TOTAL_WEIGHT_ELEMS,
)


class SimpleCNNInferencePipeline(AIEOperatorBase):
    def __init__(self, context=None):
        super().__init__(context=context)

    def get_artifacts(self, prefix="simplecnn_inference_"):
        operator_dir = Path(__file__).parent
        project_dir = operator_dir.parent
        kernels_dir = project_dir / "aie_kernels"

        kernel_fp = source_fingerprint(
            kernels_dir / "conv1_train.cc",
            kernels_dir / "conv2_train.cc",
            kernels_dir / "conv3_train.cc",
            kernels_dir / "gap_pool.cc",
            kernels_dir / "simple_head.cc",
        )
        build_fp = source_fingerprint(
            operator_dir / "config.py",
            operator_dir / "inference_design.py",
            operator_dir / "inference_op.py",
            kernels_dir / "conv1_train.cc",
            kernels_dir / "conv2_train.cc",
            kernels_dir / "conv3_train.cc",
            kernels_dir / "gap_pool.cc",
            kernels_dir / "simple_head.cc",
        )

        archive_name = f"{prefix}kernels_{kernel_fp}.a"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{prefix}b{BATCH_SIZE}_{build_fp}.mlir",
            import_path=operator_dir / "inference_design.py",
            callback_fn="simplecnn_inference_pipeline",
            callback_kwargs={"archive_name": archive_name},
            requires_context=False,
        )

        head_flags = [
            f"-DBATCH_SIZE={BATCH_SIZE}",
            f"-DDIM_H={C3}",
            f"-DNUM_CLASSES={N_CLASSES}",
        ]

        xclbin_artifact = XclbinArtifact.new(
            f"{prefix}b{BATCH_SIZE}_{build_fp}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    archive_name,
                    depends=[
                        KernelObjectArtifact.new(
                            f"conv1_infer_{kernel_fp}.o",
                            depends=[SourceArtifact.new(kernels_dir / "conv1_train.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"conv2_infer_{kernel_fp}.o",
                            depends=[SourceArtifact.new(kernels_dir / "conv2_train.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"conv3_infer_{kernel_fp}.o",
                            depends=[SourceArtifact.new(kernels_dir / "conv3_train.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"gap_infer_{kernel_fp}.o",
                            extra_flags=[
                                f"-DBATCH_SIZE={BATCH_SIZE}",
                                f"-DIN_C={C3}",
                                f"-DIN_H={CONV3_OUT_H}",
                                f"-DIN_W={CONV3_OUT_W}",
                            ],
                            depends=[SourceArtifact.new(kernels_dir / "gap_pool.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"simple_head_infer_{kernel_fp}.o",
                            extra_flags=head_flags,
                            depends=[SourceArtifact.new(kernels_dir / "simple_head.cc")],
                        ),
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{prefix}b{BATCH_SIZE}_{build_fp}.bin",
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
            "simplecnn_inference",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("images", IMG_ELEMS)
        self.add_buffer("weights", TOTAL_WEIGHT_ELEMS)
        self.add_buffer("logits", BATCH_SIZE * N_CLASSES)
        self.add_to_runlist("simplecnn_inference", "images", "weights", "logits")

    def write_weights(self, packed_weights: np.ndarray) -> None:
        self.write_buffer("weights", packed_weights)

    def run_batch(self, packed_images: np.ndarray) -> np.ndarray:
        self.write_buffer("images", packed_images)
        self.write_buffer("logits", np.zeros(BATCH_SIZE * N_CLASSES, dtype=bfloat16))
        self.run_runlist()
        return self.read_buffer(
            "logits",
            (BATCH_SIZE * N_CLASSES,),
            copy=True,
            dtype=bfloat16,
        )
