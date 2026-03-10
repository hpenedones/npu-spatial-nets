# SPDX-License-Identifier: Apache-2.0
"""
AIE operator class for the Recurrent MLP.

Single compute tile per column × up to 8 columns. Weight loaded once,
activations loop on-chip in a hardware loop for num_iters steps.
Effective depth = 2 × num_iters.
"""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from iron.common import (
    AIEOperatorBase,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)


class AIEAutoregMLP(AIEOperatorBase):
    """
    Recurrent MLP: 1-8 compute tiles looping on-chip.

    Each tile holds one H×H weight matrix and applies ReLU(x @ W)
    in a tight hardware loop. Activations ping-pong between two
    SRAM buffers. DDR I/O only at start and end.
    """

    def __init__(
        self,
        H: int = 128,
        B: int = 16,
        num_pipelines: int = 8,
        num_iters: int = 1000,
        context=None,
    ):
        self.H = H
        self.B = B
        self.num_pipelines = num_pipelines
        self.num_iters = num_iters

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        project_dir = Path(__file__).parent.parent  # npu-spatial-nets/
        operator_dir = Path(__file__).parent  # spatial_mlp/
        iron_dir = Path(self.context.base_dir)  # IRON/

        name = (f"autoreg_mlp_{self.H}h_{self.B}b_"
                f"{self.num_pipelines}p_{self.num_iters}i")

        # Source files
        mm_source = str(iron_dir / "aie_kernels" / "aie2p" / "mm.cc")
        relu_source = str(project_dir / "aie_kernels" / "mlp_kernels.cc")

        mm_defines = [
            f"-DDIM_M={self.B}",
            f"-DDIM_K={self.H}",
            f"-DDIM_N={self.H}",
            "-Dbf16_bf16_ONLY",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        relu_defines = [
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{name}.mlir",
            import_path=operator_dir / "design_autoreg.py",
            callback_fn="autoreg_mlp",
            callback_kwargs={
                "H": self.H,
                "B": self.B,
                "num_pipelines": self.num_pipelines,
                "num_iters": self.num_iters,
            },
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{name}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    "mlp_kernels.a",
                    depends=[
                        KernelObjectArtifact.new(
                            "mlp_mm.o",
                            extra_flags=mm_defines,
                            depends=[SourceArtifact.new(mm_source)],
                        ),
                        KernelObjectArtifact.new(
                            "mlp_relu.o",
                            extra_flags=relu_defines,
                            depends=[SourceArtifact.new(relu_source)],
                        ),
                    ],
                ),
            ],
            extra_flags=["--dynamic-objFifos"],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{name}.bin",
            depends=[mlir_artifact],
            extra_flags=["--dynamic-objFifos"],
        )

        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        H, B = self.H, self.B
        np_ = self.num_pipelines

        self.add_kernel(
            "autoreg",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )

        input_count = np_ * B * H
        weights_count = H * H
        output_count = np_ * B * H

        self.add_buffer("input", input_count)
        self.add_buffer("weights", weights_count)
        self.add_buffer("output", output_count)

        self.add_to_runlist("autoreg", "input", "weights", "output")
