# SPDX-License-Identifier: Apache-2.0
"""
IRON operator class for the Pipeline MLP.

Wraps the MLIR design from ``pipeline_design.py`` into an operator that
IRON's compilation and runtime system can manage.

Typical usage::

    from iron.common.aie_context import AIEContext
    from spatial_mlp.pipeline_op import AIEPipelineMLP

    ctx = AIEContext()
    op = AIEPipelineMLP(H=128, B=48, num_cols=8, context=ctx)
    ctx.compile_all()
    ctx.prepare_runtime()

    op.write_buffer("input", input_data)     # num_cols × B × H bf16
    op.write_buffer("weights", weight_data)  # num_cols × 4 × H × H bf16
    op.write_buffer("output", np.zeros(...))
    elapsed = op.run_runlist()
    result = op.read_buffer("output", shape, copy=True)
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

_BFP16_FLAG = "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"

STAGES_PER_COL = 4


class AIEPipelineMLP(AIEOperatorBase):
    """IRON operator for the 4-stage pipeline MLP on the XDNA 2 NPU.

    Each column is a 4-stage pipeline with different weights per stage.
    All columns run in parallel on different batch slices.

    Args:
        H: Hidden dimension (weight matrix is H×H).
        B: Batch size per column.
        num_cols: Number of parallel columns (1-8).
        context: IRON AIEContext for compilation.
    """

    def __init__(self, H=128, B=48, num_cols=8, context=None):
        self.H = H
        self.B = B
        self.num_cols = num_cols
        self.num_stages = num_cols * STAGES_PER_COL
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def _artifact_name(self):
        return f"pipeline_mlp_{self.H}h_{self.B}b_{self.num_cols}c"

    def set_up_artifacts(self):
        project_dir = Path(__file__).parent.parent
        name = self._artifact_name()

        matmul_relu_source = SourceArtifact.new(
            str(project_dir / "aie_kernels" / "matmul_relu.cc"))
        copy_source = SourceArtifact.new(
            str(project_dir / "aie_kernels" / "mlp_kernels.cc"))

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{name}.mlir",
            import_path=Path(__file__).parent / "pipeline_design.py",
            callback_fn="pipeline_mlp",
            callback_kwargs={
                "H": self.H, "B": self.B, "num_cols": self.num_cols,
            },
        )

        kernel_archive = KernelArchiveArtifact.new(
            "mlp_kernels.a",
            depends=[
                KernelObjectArtifact.new(
                    "mlp_mm_relu.o",
                    extra_flags=[
                        f"-DDIM_M={self.B}", f"-DDIM_K={self.H}",
                        f"-DDIM_N={self.H}",
                        _BFP16_FLAG,
                    ],
                    depends=[matmul_relu_source],
                ),
                KernelObjectArtifact.new(
                    "mlp_copy.o",
                    extra_flags=[_BFP16_FLAG],
                    depends=[copy_source],
                ),
            ],
        )

        self.xclbin_artifact = XclbinArtifact.new(
            f"{name}.xclbin",
            depends=[mlir_artifact, kernel_archive],
            extra_flags=["--dynamic-objFifos"],
        )
        self.insts_artifact = InstsBinArtifact.new(
            f"{name}.bin",
            depends=[mlir_artifact],
            extra_flags=["--dynamic-objFifos"],
        )
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        """Register kernel, buffers, and runlist.

        Buffers:
        - ``input``:   num_cols × B × H bf16 (tiled layout)
        - ``weights``: num_cols × 4 × H × H bf16 (tiled layout)
        - ``output``:  num_cols × B × H bf16 (tiled layout)
        """
        total_act = self.num_cols * self.B * self.H
        total_wt = self.num_cols * STAGES_PER_COL * self.H * self.H

        self.add_kernel(
            "pipeline", self.xclbin_artifact,
            self.xclbin_artifact.kernel_name, self.insts_artifact,
        )
        self.add_buffer("input", total_act)
        self.add_buffer("weights", total_wt)
        self.add_buffer("output", total_act)
        self.add_to_runlist("pipeline", "input", "weights", "output")
