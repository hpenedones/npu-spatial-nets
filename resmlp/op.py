"""
IRON operator wrapper for the residual MLP snake pipeline.

Handles compilation of the AIE kernel and MLIR design, buffer setup,
and NPU execution.
"""

from pathlib import Path
from iron.common import (
    AIEOperatorBase,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)

ROWS_PER_COL = 4


class ResMLP(AIEOperatorBase):
    """AIE-accelerated 32-layer residual MLP: y = relu(x @ W) + x per tile."""

    def __init__(self, H=160, B=8, num_cols=8, context=None):
        self.H = H
        self.B = B
        self.num_cols = num_cols
        self.num_tiles = num_cols * ROWS_PER_COL
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="resmlp_"):
        operator_dir = Path(__file__).parent
        project_dir = operator_dir.parent  # npu-spatial-nets root
        H, B = self.H, self.B

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{prefix}{B}x{H}_{self.num_cols}col.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="snake_pipeline",
            callback_kwargs={"H": H, "B": B, "num_cols": self.num_cols},
            requires_context=False,
        )

        archive_name = f"{prefix}kernel.a"
        kernel_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_K={H}",
            f"-DDIM_N={H}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        xclbin_artifact = XclbinArtifact.new(
            f"{prefix}{B}x{H}_{self.num_cols}col.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    archive_name,
                    depends=[
                        KernelObjectArtifact.new(
                            f"{prefix}kernel.o",
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
            f"{prefix}{B}x{H}_{self.num_cols}col.bin",
            depends=[mlir_artifact],
        )

        return (xclbin_artifact, insts_artifact)

    def set_up_artifacts(self):
        xclbin, insts = self.get_artifacts()
        self.xclbin_artifact = xclbin
        self.insts_artifact = insts
        self.add_artifacts([xclbin, insts])

    def set_up_runtime(self):
        H, B = self.H, self.B
        self.add_kernel(
            "resmlp",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("input", B * H)
        self.add_buffer("weights", self.num_tiles * H * H)
        self.add_buffer("output", B * H)
        self.add_to_runlist("resmlp", "input", "weights", "output")
