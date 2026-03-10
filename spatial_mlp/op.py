# SPDX-License-Identifier: Apache-2.0
"""
AIE operator class for the Recurrent MLP.

Up to 32 compute tiles (4 rows x 8 columns), each with weight held in SRAM.
Hardware loop applies ReLU(x @ W) for num_iters iterations on-chip.
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


class AIERecurrentMLP(AIEOperatorBase):

    def __init__(self, H=128, B=16, num_tiles=24, num_iters=1000, context=None):
        self.H = H
        self.B = B
        self.num_tiles = num_tiles
        self.num_iters = num_iters
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        project_dir = Path(__file__).parent.parent
        operator_dir = Path(__file__).parent
        iron_dir = Path(self.context.base_dir)

        name = f"recurrent_mlp_{self.H}h_{self.B}b_{self.num_tiles}t_{self.num_iters}i"

        mm_source = str(iron_dir / "aie_kernels" / "aie2p" / "mm.cc")
        relu_source = str(project_dir / "aie_kernels" / "mlp_kernels.cc")

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{name}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="recurrent_mlp",
            callback_kwargs={
                "H": self.H, "B": self.B,
                "num_tiles": self.num_tiles, "num_iters": self.num_iters,
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
                            extra_flags=[
                                f"-DDIM_M={self.B}", f"-DDIM_K={self.H}",
                                f"-DDIM_N={self.H}", "-Dbf16_bf16_ONLY",
                                "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
                            ],
                            depends=[SourceArtifact.new(mm_source)],
                        ),
                        KernelObjectArtifact.new(
                            "mlp_relu.o",
                            extra_flags=["-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"],
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
        H, B, nt = self.H, self.B, self.num_tiles
        self.add_kernel(
            "recurrent", self.xclbin_artifact,
            self.xclbin_artifact.kernel_name, self.insts_artifact,
        )
        self.add_buffer("input", nt * B * H)
        self.add_buffer("weights", H * H)
        self.add_buffer("output", nt * B * H)
        self.add_to_runlist("recurrent", "input", "weights", "output")
