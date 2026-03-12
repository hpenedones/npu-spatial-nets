"""
IRON operator wrapper for the single-layer residual backward kernel.
"""

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


class ResidualBackwardSingle(AIEOperatorBase):
    """Single-tile backward operator for one residual layer."""

    def __init__(self, H=160, B=8, mode="grad_input", context=None):
        self.H = H
        self.B = B
        self.mode = mode
        super().__init__(context=context)

    def get_artifacts(self, prefix="resmlp_backward_"):
        operator_dir = Path(__file__).parent
        project_dir = operator_dir.parent
        H, B = self.H, self.B
        mode = self.mode

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{prefix}{mode}_{B}x{H}.mlir",
            import_path=operator_dir / "backward_single_design.py",
            callback_fn="backward_single",
            callback_kwargs={"H": H, "B": B, "mode": mode},
            requires_context=False,
        )

        archive_name = f"{prefix}{mode}_kernel.a"
        kernel_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_K={H}",
            f"-DDIM_N={H}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        xclbin_artifact = XclbinArtifact.new(
            f"{prefix}{mode}_{B}x{H}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    archive_name,
                    depends=[
                        KernelObjectArtifact.new(
                            f"{prefix}{mode}_kernel.o",
                            extra_flags=kernel_flags,
                            depends=[
                                SourceArtifact.new(
                                    project_dir
                                    / "aie_kernels"
                                    / "residual_backward.cc"
                                )
                            ],
                        )
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{prefix}{mode}_{B}x{H}.bin",
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
            "residual_backward",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        if self.mode == "grad_input":
            self.add_buffer("state", 2 * B * H)
            self.add_buffer("weights_t", H * H)
            self.add_buffer("grad_in", B * H)
            self.add_to_runlist(
                "residual_backward",
                "state",
                "weights_t",
                "grad_in",
            )
            return

        if self.mode == "weight_grad":
            self.add_buffer("state", 3 * B * H)
            self.add_buffer("dweights", H * H)
            self.add_to_runlist(
                "residual_backward",
                "state",
                "dweights",
            )
            return

        raise ValueError(f"Unsupported backward mode: {self.mode}")
