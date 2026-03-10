# SPDX-License-Identifier: Apache-2.0
"""
IRON operator class for the Recurrent MLP.

This module wraps the MLIR design from ``design.py`` into an ``AIEOperatorBase``
subclass that IRON's compilation and runtime system can manage. It defines
the artifact dependency graph (source → object → archive → xclbin) and the
runtime buffers that the host program reads/writes.

Typical usage::

    from iron.common.aie_context import AIEContext
    from spatial_mlp.op import AIERecurrentMLP

    ctx = AIEContext()
    op = AIERecurrentMLP(H=128, B=16, num_tiles=24, num_iters=1000, context=ctx)
    ctx.compile_all()
    ctx.prepare_runtime()

    op.write_buffer("input", input_data)
    op.write_buffer("weights", weight_data)
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

# Path from IRON's aie_kernels directory to the accumulating matmul kernel.
_MM_KERNEL_SUBPATH = Path("aie_kernels") / "aie2p" / "mm.cc"

# Compiler flag to use BFP16 emulation for bfloat16 matmul on AIE2P.
_BFP16_FLAG = "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"


class AIERecurrentMLP(AIEOperatorBase):
    """IRON operator for the recurrent MLP on the XDNA 2 NPU.

    Manages compilation artifacts and runtime buffers for a recurrent MLP
    that runs on ``num_tiles`` compute tiles with a hardware loop of
    ``num_iters`` iterations.

    Args:
        H: Hidden dimension (weight matrix is H×H).
        B: Batch size per tile.
        num_tiles: Number of parallel compute tiles (1–24).
        num_iters: Hardware loop count (effective depth = 2 × num_iters).
        context: IRON AIEContext for compilation.
    """

    def __init__(self, H=128, B=16, num_tiles=24, num_iters=1000, context=None):
        self.H = H
        self.B = B
        self.num_tiles = num_tiles
        self.num_iters = num_iters
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def _artifact_name(self):
        """Unique name for this configuration's compiled artifacts."""
        return (f"recurrent_mlp_{self.H}h_{self.B}b_"
                f"{self.num_tiles}t_{self.num_iters}i")

    def set_up_artifacts(self):
        """Define the compilation dependency graph.

        The graph is::

            mm.cc ──► mlp_mm.o ──┐
                                 ├──► mlp_kernels.a ──┐
            mlp_kernels.cc ──► mlp_relu.o ──┘          │
                                                       ├──► {name}.xclbin
            design.py ──► {name}.mlir ────────────────┘
                              │
                              └──► {name}.bin  (instruction sequence)

        Both xclbin and bin use ``--dynamic-objFifos`` for runtime-flexible
        ObjectFIFO sizing.
        """
        project_dir = Path(__file__).parent.parent
        iron_dir = Path(self.context.base_dir)
        name = self._artifact_name()

        mm_source = SourceArtifact.new(str(iron_dir / _MM_KERNEL_SUBPATH))
        relu_source = SourceArtifact.new(
            str(project_dir / "aie_kernels" / "mlp_kernels.cc"))

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{name}.mlir",
            import_path=Path(__file__).parent / "design.py",
            callback_fn="recurrent_mlp",
            callback_kwargs={
                "H": self.H, "B": self.B,
                "num_tiles": self.num_tiles, "num_iters": self.num_iters,
            },
        )

        kernel_archive = KernelArchiveArtifact.new(
            "mlp_kernels.a",
            depends=[
                KernelObjectArtifact.new(
                    "mlp_mm.o",
                    extra_flags=[
                        f"-DDIM_M={self.B}", f"-DDIM_K={self.H}",
                        f"-DDIM_N={self.H}", "-Dbf16_bf16_ONLY",
                        _BFP16_FLAG,
                    ],
                    depends=[mm_source],
                ),
                KernelObjectArtifact.new(
                    "mlp_relu.o",
                    extra_flags=[_BFP16_FLAG],
                    depends=[relu_source],
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
        """Register kernel, buffers, and runlist for host-side execution.

        Buffers:
        - ``input``:   num_tiles × B × H bf16 values (tiled layout)
        - ``weights``: H × H bf16 values (tiled layout)
        - ``output``:  num_tiles × B × H bf16 values (tiled layout)
        """
        self.add_kernel(
            "recurrent", self.xclbin_artifact,
            self.xclbin_artifact.kernel_name, self.insts_artifact,
        )
        self.add_buffer("input", self.num_tiles * self.B * self.H)
        self.add_buffer("weights", self.H * self.H)
        self.add_buffer("output", self.num_tiles * self.B * self.H)
        self.add_to_runlist("recurrent", "input", "weights", "output")
