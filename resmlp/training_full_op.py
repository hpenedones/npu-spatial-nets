"""Operator wrapper for the full 100% NPU training pipeline."""

from pathlib import Path

from iron.common import (
    AIEOperatorBase,
    InstsBinArtifact,
    KernelArchiveArtifact,
    KernelObjectArtifact,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
    XclbinArtifact,
)

from resmlp.training_full_design import (
    ROWS_PER_COL, NUM_RESIDUAL, N_CLS_PADDED,
)


class FullTrainingPipeline(AIEOperatorBase):
    """32-tile pipeline: embed(784→H) + 30×residual(H→H) + head(H→10) + loss + SGD."""

    def __init__(self, H=32, B=8, K_EMBED=784, num_cols=8, context=None):
        self.H = H
        self.B = B
        self.K_EMBED = K_EMBED
        self.num_cols = num_cols
        super().__init__(context=context)

    def get_artifacts(self, prefix="full_training_"):
        operator_dir = Path(__file__).parent
        project_dir = operator_dir.parent
        H, B, K = self.H, self.B, self.K_EMBED

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{prefix}{B}x{H}.mlir",
            import_path=operator_dir / "training_full_design.py",
            callback_fn="full_training_pipeline",
            callback_kwargs={"H": H, "B": B, "K_EMBED": K, "num_cols": self.num_cols},
            requires_context=False,
        )

        # Residual kernels (same as before, but with H=32)
        res_kernel_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_K={H}",
            f"-DDIM_N={H}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        # Embed kernels
        embed_kernel_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_K_EMBED={K}",
            f"-DDIM_H={H}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        # Head kernels
        head_kernel_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_H={H}",
            f"-DDIM_N_CLS={N_CLS_PADDED}",
            f"-DNUM_CLASSES=10",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        copy_head_flags = [
            f"-DDIM_M={H}",
            f"-DDIM_K={N_CLS_PADDED}",
            "-DCOPY_KERNEL_NAME=copy_head_weight_bf16",
        ]
        kernels_dir = project_dir / "aie_kernels"

        xclbin_artifact = XclbinArtifact.new(
            f"{prefix}{B}x{H}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    "full_training_kernels.a",
                    depends=[
                        KernelObjectArtifact.new(
                            "full_matmul_relu_skip.o",
                            extra_flags=res_kernel_flags,
                            depends=[SourceArtifact.new(kernels_dir / "matmul_relu_skip.cc")],
                        ),
                        KernelObjectArtifact.new(
                            "full_residual_backward.o",
                            extra_flags=res_kernel_flags,
                            depends=[SourceArtifact.new(kernels_dir / "residual_backward.cc")],
                        ),
                        KernelObjectArtifact.new(
                            "full_copy_activation.o",
                            extra_flags=[f"-DDIM_M={B}", f"-DDIM_K={H}"],
                            depends=[SourceArtifact.new(kernels_dir / "copy_activation.cc")],
                        ),
                        KernelObjectArtifact.new(
                            "copy_head_weight.o",
                            extra_flags=copy_head_flags,
                            depends=[SourceArtifact.new(kernels_dir / "copy_activation.cc")],
                        ),
                        KernelObjectArtifact.new(
                            "embed_forward.o",
                            extra_flags=embed_kernel_flags,
                            depends=[SourceArtifact.new(kernels_dir / "embed_forward.cc")],
                        ),
                        KernelObjectArtifact.new(
                            "embed_backward.o",
                            extra_flags=embed_kernel_flags,
                            depends=[SourceArtifact.new(kernels_dir / "embed_backward.cc")],
                        ),
                        KernelObjectArtifact.new(
                            "head_forward_loss.o",
                            extra_flags=head_kernel_flags,
                            depends=[SourceArtifact.new(kernels_dir / "head_forward_loss.cc")],
                        ),
                        KernelObjectArtifact.new(
                            "head_backward.o",
                            extra_flags=head_kernel_flags,
                            depends=[SourceArtifact.new(kernels_dir / "head_backward.cc")],
                        ),
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{prefix}{B}x{H}.bin",
            depends=[mlir_artifact],
        )

        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        xclbin, insts = self.get_artifacts()
        self.xclbin_artifact = xclbin
        self.insts_artifact = insts
        self.add_artifacts([xclbin, insts])

    def set_up_runtime(self):
        H, B, K = self.H, self.B, self.K_EMBED

        self.add_kernel(
            "full_training",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("x_raw", B * K)          # embed input
        self.add_buffer("embed_wt", K * H)        # embed weights
        self.add_buffer("res_wt", NUM_RESIDUAL * H * H)  # residual weights
        self.add_buffer("head_wt", H * N_CLS_PADDED)     # head weights
        self.add_buffer("labels", 2 * B, dtype="int32")   # labels + preds

        self.add_to_runlist(
            "full_training",
            "x_raw",
            "embed_wt",
            "res_wt",
            "head_wt",
            "labels",
        )
