from dataclasses import dataclass


BATCH_SIZE = 8

KERNEL_SIZE = 3
STEM_STRIDE = 2
STEM_PADDING = 1
BLOCK_STRIDE = 1
BLOCK_PADDING = 1

C1 = 4
C2 = 8
C3 = 16

N_CLASSES = 10
ROWS_PER_COL = 4
DEFAULT_NUM_COLS = 8
SUPPORTED_DATASETS = ("mnist", "cifar10")


def normalize_dataset_name(dataset: str) -> str:
    key = dataset.lower()
    if key not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported convsnake dataset '{dataset}'. Expected one of: {supported}")
    return key


def conv_out_size(size: int, *, stride: int, padding: int) -> int:
    return ((size + 2 * padding - KERNEL_SIZE) // stride) + 1


def round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class ConvSnakeConfig:
    dataset: str
    batch_size: int
    img_c: int
    img_h: int
    img_w: int
    c1: int
    c2: int
    c3: int
    num_classes: int
    conv1_out_h: int
    conv1_out_w: int
    conv2_out_h: int
    conv2_out_w: int
    conv3_out_h: int
    conv3_out_w: int
    img_elems: int
    act1_elems: int
    act2_elems: int
    act3_elems: int
    logits_elems: int
    conv1_w_elems: int
    conv2_w_elems: int
    conv3_w_elems: int
    block_w_elems: int
    flat_dim: int
    head_w_elems: int
    conv1_npu_out_c: int
    conv1_npu_k: int
    conv1_npu_w_elems: int
    conv2_npu_out_c: int
    conv2_npu_k: int
    conv2_npu_w_elems: int
    conv3_npu_out_c: int
    conv3_npu_k: int
    conv3_npu_w_elems: int
    block_npu_out_c: int
    block_npu_k: int
    block_npu_w_elems: int
    conv1_positions_chunk: int
    conv2_positions_chunk: int
    conv3_positions_chunk: int

    def to_dict(self) -> dict[str, int | str]:
        return {"dataset": self.dataset, "batch_size": self.batch_size}

    @classmethod
    def from_dict(cls, values: dict[str, int | str]) -> "ConvSnakeConfig":
        return build_config(
            dataset=str(values.get("dataset", "mnist")),
            batch_size=(
                int(values["batch_size"])
                if "batch_size" in values and values["batch_size"] is not None
                else None
            ),
        )


def _dataset_image_spec(dataset: str) -> tuple[int, int, int, int, int, int, int]:
    key = normalize_dataset_name(dataset)
    if key == "mnist":
        return (BATCH_SIZE, 1, 28, 28, 56, 56, 64)
    if key == "cifar10":
        return (4, 3, 32, 32, 64, 64, 64)
    raise AssertionError(f"Unhandled dataset: {dataset}")


def build_config(dataset: str = "mnist", *, batch_size: int | None = None) -> ConvSnakeConfig:
    default_batch_size, img_c, img_h, img_w, conv1_chunk, conv2_chunk, conv3_chunk = _dataset_image_spec(
        dataset
    )
    dataset = normalize_dataset_name(dataset)
    if batch_size is None:
        batch_size = default_batch_size

    conv1_out_h = conv_out_size(img_h, stride=STEM_STRIDE, padding=STEM_PADDING)
    conv1_out_w = conv_out_size(img_w, stride=STEM_STRIDE, padding=STEM_PADDING)
    conv2_out_h = conv_out_size(conv1_out_h, stride=STEM_STRIDE, padding=STEM_PADDING)
    conv2_out_w = conv_out_size(conv1_out_w, stride=STEM_STRIDE, padding=STEM_PADDING)
    conv3_out_h = conv_out_size(conv2_out_h, stride=STEM_STRIDE, padding=STEM_PADDING)
    conv3_out_w = conv_out_size(conv2_out_w, stride=STEM_STRIDE, padding=STEM_PADDING)

    if (conv3_out_h, conv3_out_w) != (4, 4):
        raise ValueError(
            f"ConvSnake expects a 4x4 post-stem feature map, got {(conv3_out_h, conv3_out_w)} "
            f"for dataset '{dataset}'"
        )

    img_elems = batch_size * img_h * img_w * img_c
    act1_elems = batch_size * conv1_out_h * conv1_out_w * C1
    act2_elems = batch_size * conv2_out_h * conv2_out_w * C2
    act3_elems = batch_size * conv3_out_h * conv3_out_w * C3
    logits_elems = batch_size * N_CLASSES

    conv1_w_elems = C1 * img_c * KERNEL_SIZE * KERNEL_SIZE
    conv2_w_elems = C2 * C1 * KERNEL_SIZE * KERNEL_SIZE
    conv3_w_elems = C3 * C2 * KERNEL_SIZE * KERNEL_SIZE
    block_w_elems = C3 * C3 * KERNEL_SIZE * KERNEL_SIZE
    flat_dim = conv3_out_h * conv3_out_w * C3
    head_w_elems = flat_dim * N_CLASSES

    conv1_npu_out_c = round_up_to_multiple(C1, 8)
    conv1_npu_k = round_up_to_multiple(conv1_w_elems // C1, 8)
    conv2_npu_out_c = round_up_to_multiple(C2, 8)
    conv2_npu_k = round_up_to_multiple(conv2_w_elems // C2, 8)
    conv3_npu_out_c = round_up_to_multiple(C3, 8)
    conv3_npu_k = round_up_to_multiple(conv3_w_elems // C3, 8)
    block_npu_out_c = round_up_to_multiple(C3, 8)
    block_npu_k = round_up_to_multiple(block_w_elems // C3, 8)

    conv1_positions = batch_size * conv1_out_h * conv1_out_w
    conv2_positions = batch_size * conv2_out_h * conv2_out_w
    conv3_positions = batch_size * conv3_out_h * conv3_out_w
    if conv1_positions % conv1_chunk != 0:
        raise ValueError(f"conv1 chunk {conv1_chunk} does not divide positions {conv1_positions}")
    if conv2_positions % conv2_chunk != 0:
        raise ValueError(f"conv2 chunk {conv2_chunk} does not divide positions {conv2_positions}")
    if conv3_positions % conv3_chunk != 0:
        raise ValueError(f"conv3 chunk {conv3_chunk} does not divide positions {conv3_positions}")

    return ConvSnakeConfig(
        dataset=dataset,
        batch_size=batch_size,
        img_c=img_c,
        img_h=img_h,
        img_w=img_w,
        c1=C1,
        c2=C2,
        c3=C3,
        num_classes=N_CLASSES,
        conv1_out_h=conv1_out_h,
        conv1_out_w=conv1_out_w,
        conv2_out_h=conv2_out_h,
        conv2_out_w=conv2_out_w,
        conv3_out_h=conv3_out_h,
        conv3_out_w=conv3_out_w,
        img_elems=img_elems,
        act1_elems=act1_elems,
        act2_elems=act2_elems,
        act3_elems=act3_elems,
        logits_elems=logits_elems,
        conv1_w_elems=conv1_w_elems,
        conv2_w_elems=conv2_w_elems,
        conv3_w_elems=conv3_w_elems,
        block_w_elems=block_w_elems,
        flat_dim=flat_dim,
        head_w_elems=head_w_elems,
        conv1_npu_out_c=conv1_npu_out_c,
        conv1_npu_k=conv1_npu_k,
        conv1_npu_w_elems=conv1_w_elems,
        conv2_npu_out_c=conv2_npu_out_c,
        conv2_npu_k=conv2_npu_k,
        conv2_npu_w_elems=conv2_w_elems,
        conv3_npu_out_c=conv3_npu_out_c,
        conv3_npu_k=conv3_npu_k,
        conv3_npu_w_elems=conv3_w_elems,
        block_npu_out_c=block_npu_out_c,
        block_npu_k=block_npu_k,
        block_npu_w_elems=block_w_elems,
        conv1_positions_chunk=conv1_chunk,
        conv2_positions_chunk=conv2_chunk,
        conv3_positions_chunk=conv3_chunk,
    )


DEFAULT_CONFIG = build_config("mnist")

IMG_C = DEFAULT_CONFIG.img_c
IMG_H = DEFAULT_CONFIG.img_h
IMG_W = DEFAULT_CONFIG.img_w

CONV1_OUT_H = DEFAULT_CONFIG.conv1_out_h
CONV1_OUT_W = DEFAULT_CONFIG.conv1_out_w
CONV2_OUT_H = DEFAULT_CONFIG.conv2_out_h
CONV2_OUT_W = DEFAULT_CONFIG.conv2_out_w
CONV3_OUT_H = DEFAULT_CONFIG.conv3_out_h
CONV3_OUT_W = DEFAULT_CONFIG.conv3_out_w

IMG_ELEMS = DEFAULT_CONFIG.img_elems
ACT1_ELEMS = DEFAULT_CONFIG.act1_elems
ACT2_ELEMS = DEFAULT_CONFIG.act2_elems
ACT3_ELEMS = DEFAULT_CONFIG.act3_elems
LOGITS_ELEMS = DEFAULT_CONFIG.logits_elems

CONV1_W_ELEMS = DEFAULT_CONFIG.conv1_w_elems
CONV2_W_ELEMS = DEFAULT_CONFIG.conv2_w_elems
CONV3_W_ELEMS = DEFAULT_CONFIG.conv3_w_elems
BLOCK_W_ELEMS = DEFAULT_CONFIG.block_w_elems
FLAT_DIM = DEFAULT_CONFIG.flat_dim
HEAD_W_ELEMS = DEFAULT_CONFIG.head_w_elems

CONV1_NPU_OUT_C = DEFAULT_CONFIG.conv1_npu_out_c
CONV1_NPU_K = DEFAULT_CONFIG.conv1_npu_k
CONV1_NPU_W_ELEMS = DEFAULT_CONFIG.conv1_npu_w_elems

CONV2_NPU_OUT_C = DEFAULT_CONFIG.conv2_npu_out_c
CONV2_NPU_K = DEFAULT_CONFIG.conv2_npu_k
CONV2_NPU_W_ELEMS = DEFAULT_CONFIG.conv2_npu_w_elems

CONV3_NPU_OUT_C = DEFAULT_CONFIG.conv3_npu_out_c
CONV3_NPU_K = DEFAULT_CONFIG.conv3_npu_k
CONV3_NPU_W_ELEMS = DEFAULT_CONFIG.conv3_npu_w_elems

BLOCK_NPU_OUT_C = DEFAULT_CONFIG.block_npu_out_c
BLOCK_NPU_K = DEFAULT_CONFIG.block_npu_k
BLOCK_NPU_W_ELEMS = DEFAULT_CONFIG.block_npu_w_elems

DEFAULT_NUM_TILES = DEFAULT_NUM_COLS * ROWS_PER_COL
DEFAULT_NUM_BLOCKS = DEFAULT_NUM_TILES - 4


def num_blocks_for_cols(num_cols: int) -> int:
    num_tiles = num_cols * ROWS_PER_COL
    if num_tiles < 4:
        raise ValueError("conv snake needs at least 4 tiles (3 conv stages + head)")
    return num_tiles - 4


def total_flops_per_image(num_cols: int = DEFAULT_NUM_COLS, *, config: ConvSnakeConfig | None = None) -> int:
    cfg = DEFAULT_CONFIG if config is None else config
    num_blocks = num_blocks_for_cols(num_cols)
    conv1_macs = cfg.conv1_out_h * cfg.conv1_out_w * cfg.c1 * KERNEL_SIZE * KERNEL_SIZE * cfg.img_c
    conv2_macs = cfg.conv2_out_h * cfg.conv2_out_w * cfg.c2 * KERNEL_SIZE * KERNEL_SIZE * cfg.c1
    conv3_macs = cfg.conv3_out_h * cfg.conv3_out_w * cfg.c3 * KERNEL_SIZE * KERNEL_SIZE * cfg.c2
    block_macs = cfg.conv3_out_h * cfg.conv3_out_w * cfg.c3 * KERNEL_SIZE * KERNEL_SIZE * cfg.c3
    head_macs = cfg.flat_dim * cfg.num_classes
    return 2 * (conv1_macs + conv2_macs + conv3_macs + num_blocks * block_macs + head_macs)


@dataclass(frozen=True)
class EmbeddedWeightShapes:
    conv1: tuple[int, ...]
    conv2: tuple[int, ...]
    conv3: tuple[int, ...]
    head: tuple[int, ...]


WEIGHT_SHAPES = EmbeddedWeightShapes(
    conv1=(CONV1_W_ELEMS,),
    conv2=(CONV2_W_ELEMS,),
    conv3=(CONV3_W_ELEMS,),
    head=(HEAD_W_ELEMS,),
)
