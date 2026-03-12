"""
Operator wrapper for the 4-tile forward checkpoint probe.
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

from resmlp.forward_checkpoint_design import ROWS_PER_COL


class ForwardCheckpointColumn(AIEOperatorBase):
    """4-tile column probe that emits per-tile input checkpoints."""

    def __init__(self, H=160, B=8, context=None):
        self.H = H
        self.B = B
        super().__init__(context=context)

    def get_artifacts(self, prefix="resmlp_forward_checkpoint_"):
        operator_dir = Path(__file__).parent
        project_dir = operator_dir.parent
        H, B = self.H, self.B

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{prefix}{B}x{H}.mlir",
            import_path=operator_dir / "forward_checkpoint_design.py",
            callback_fn="forward_checkpoint_column",
            callback_kwargs={"H": H, "B": B},
            requires_context=False,
        )

        kernel_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_K={H}",
            f"-DDIM_N={H}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        xclbin_artifact = XclbinArtifact.new(
            f"{prefix}{B}x{H}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    "resmlp_forward_checkpoint_kernels.a",
                    depends=[
                        KernelObjectArtifact.new(
                            "forward_ckpt_kernel.o",
                            extra_flags=kernel_flags,
                            depends=[
                                SourceArtifact.new(
                                    project_dir / "aie_kernels" / "matmul_relu_skip.cc"
                                )
                            ],
                        ),
                        KernelObjectArtifact.new(
                            "copy_activation.o",
                            extra_flags=[f"-DDIM_M={B}", f"-DDIM_K={H}"],
                            depends=[
                                SourceArtifact.new(
                                    project_dir / "aie_kernels" / "copy_activation.cc"
                                )
                            ],
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
        H, B = self.H, self.B
        self.add_kernel(
            "forward_checkpoint",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("input", B * H)
        self.add_buffer("weights", ROWS_PER_COL * H * H)
        self.add_buffer("checkpoints", ROWS_PER_COL * B * H)
        self.add_buffer("output", B * H)
        self.add_to_runlist(
            "forward_checkpoint",
            "input",
            "weights",
            "checkpoints",
            "output",
        )
