"""Shared dataset and preview helpers for ResMLP workflows."""

import gzip
from math import ceil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

DEFAULT_VAL_SIZE = 10_000
DEFAULT_SPLIT_SEED = 1234
HIGGS_PADDED_INPUT_DIM = 56
HIGGS_RAW_INPUT_DIM = 28
HIGGS_TEST_FRACTION = 0.1

DATASET_CONFIGS = {
    "mnist": {
        "kind": "image",
        "input_dim": 28 * 28,
        "num_classes": 10,
        "mean": (0.1307,),
        "std": (0.3081,),
        "image_size": (28, 28),
        "channels": 1,
    },
    "cifar10": {
        "kind": "image",
        "input_dim": 3 * 32 * 32,
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "image_size": (32, 32),
        "channels": 3,
    },
    "higgs": {
        "kind": "tabular",
        "input_dim": HIGGS_PADDED_INPUT_DIM,
        "raw_input_dim": HIGGS_RAW_INPUT_DIM,
        "num_classes": 2,
        "test_fraction": HIGGS_TEST_FRACTION,
    },
}
SUPPORTED_DATASETS = tuple(DATASET_CONFIGS)
SUPPORTED_TRAIN_AUGS = ("none", "basic", "strong")

_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


class NormalizedTabularDataset(Dataset):
    def __init__(self, features, labels, *, mean, std):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)
        if self.features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {tuple(self.features.shape)}")
        if self.labels.ndim != 1 or self.labels.shape[0] != self.features.shape[0]:
            raise ValueError("Labels must be a 1D tensor aligned with features")

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (self.features[idx] - self.mean) / self.std, int(self.labels[idx])


def _find_higgs_path(data_dir, names):
    data_dir = Path(data_dir)
    for name in names:
        path = data_dir / name
        if path.exists():
            return path
    return None


def _prepare_higgs_tensors(features, labels):
    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if features.ndim != 2:
        raise ValueError(f"HIGGS features must be 2D, got shape {tuple(features.shape)}")
    if features.shape[1] not in {HIGGS_RAW_INPUT_DIM, HIGGS_PADDED_INPUT_DIM}:
        raise ValueError(
            f"HIGGS features must have {HIGGS_RAW_INPUT_DIM} or {HIGGS_PADDED_INPUT_DIM} columns, "
            f"got {features.shape[1]}"
        )
    if features.shape[1] == HIGGS_RAW_INPUT_DIM:
        padded = torch.zeros(features.shape[0], HIGGS_PADDED_INPUT_DIM, dtype=torch.float32)
        padded[:, :HIGGS_RAW_INPUT_DIM] = features
        features = padded
    labels = labels.view(-1)
    if labels.shape[0] != features.shape[0]:
        raise ValueError("HIGGS labels/features row count mismatch")
    return features, labels


def _load_higgs_tensors(data_dir="data"):
    data_dir = Path(data_dir)
    cache_path = _find_higgs_path(data_dir, ("HIGGS.pt", "higgs.pt"))
    if cache_path is not None:
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict) or "features" not in data or "labels" not in data:
            raise ValueError(f"Expected HIGGS tensor cache dict with features/labels at {cache_path}")
        return _prepare_higgs_tensors(data["features"], data["labels"])

    raw_path = _find_higgs_path(data_dir, ("HIGGS.csv.gz", "HIGGS.csv", "higgs.csv.gz", "higgs.csv"))
    if raw_path is None:
        raise FileNotFoundError(
            f"HIGGS dataset not found under {data_dir}. Expected one of: "
            "HIGGS.pt, higgs.pt, HIGGS.csv.gz, HIGGS.csv, higgs.csv.gz, higgs.csv"
        )

    opener = gzip.open if raw_path.suffix == ".gz" else open
    with opener(raw_path, "rt") as handle:
        table = np.loadtxt(handle, delimiter=",", dtype=np.float32)
    if table.ndim != 2 or table.shape[1] < 1 + HIGGS_RAW_INPUT_DIM:
        raise ValueError(
            f"HIGGS raw file must have at least {1 + HIGGS_RAW_INPUT_DIM} columns, got shape {table.shape}"
        )

    labels = table[:, 0].astype(np.int64, copy=False)
    features = table[:, 1 : 1 + HIGGS_RAW_INPUT_DIM]
    features_t, labels_t = _prepare_higgs_tensors(features, labels)
    torch.save({"features": features_t, "labels": labels_t}, data_dir / "HIGGS.pt")
    return features_t, labels_t


def load_higgs_datasets(data_dir="data", *, split_seed=DEFAULT_SPLIT_SEED):
    features, labels = _load_higgs_tensors(data_dir=data_dir)
    total = features.shape[0]
    if total < 2:
        raise ValueError("HIGGS dataset must contain at least 2 rows")

    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1e-6)
    full_ds = NormalizedTabularDataset(features, labels, mean=mean, std=std)

    test_size = max(1, int(round(total * HIGGS_TEST_FRACTION)))
    if test_size >= total:
        test_size = total // 10 or 1
    generator = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(total, generator=generator)
    test_idx = perm[:test_size].tolist()
    train_idx = perm[test_size:].tolist()
    return Subset(full_ds, train_idx), Subset(full_ds, test_idx)


def normalize_dataset_name(dataset_name):
    key = dataset_name.lower()
    if key not in DATASET_CONFIGS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of: {supported}")
    return key


def resolve_dataset_name(requested_dataset=None, checkpoint_dataset=None):
    requested = normalize_dataset_name(requested_dataset) if requested_dataset else None
    checkpoint = normalize_dataset_name(checkpoint_dataset) if checkpoint_dataset else None
    if requested and checkpoint and requested != checkpoint:
        raise ValueError(
            f"Checkpoint was trained for dataset '{checkpoint}', but '{requested}' was requested"
        )
    return requested or checkpoint or "mnist"


def get_dataset_config(dataset_name):
    return dict(DATASET_CONFIGS[normalize_dataset_name(dataset_name)])


def normalize_train_aug(train_aug):
    key = train_aug.lower()
    if key not in SUPPORTED_TRAIN_AUGS:
        supported = ", ".join(SUPPORTED_TRAIN_AUGS)
        raise ValueError(f"Unsupported train augmentation '{train_aug}'. Expected one of: {supported}")
    return key


def dataset_transform(dataset_name, *, train=False, train_aug="none"):
    dataset_name = normalize_dataset_name(dataset_name)
    train_aug = normalize_train_aug(train_aug)
    config = get_dataset_config(dataset_name)
    if config["kind"] != "image":
        if train_aug != "none":
            raise ValueError(
                f"Train augmentation '{train_aug}' is only implemented for image datasets, not '{dataset_name}'"
            )
        return None
    steps = []
    if train and train_aug != "none":
        if dataset_name != "cifar10":
            raise ValueError(
                f"Train augmentation '{train_aug}' is only implemented for CIFAR-10, not '{dataset_name}'"
            )
        steps.extend(
            [
                transforms.RandomCrop(config["image_size"], padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"]),
        ]
    )
    if train and dataset_name == "cifar10" and train_aug == "strong":
        steps.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value="random"))
    return transforms.Compose(steps)


def load_datasets(dataset_name, data_dir="data", *, train_aug="none", split_seed=DEFAULT_SPLIT_SEED):
    dataset_name = normalize_dataset_name(dataset_name)
    config = get_dataset_config(dataset_name)
    if config["kind"] == "tabular":
        return load_higgs_datasets(data_dir=data_dir, split_seed=split_seed)

    train_transform = dataset_transform(dataset_name, train=True, train_aug=train_aug)
    eval_transform = dataset_transform(dataset_name, train=False)
    if dataset_name == "mnist":
        dataset_cls = datasets.MNIST
    elif dataset_name == "cifar10":
        dataset_cls = datasets.CIFAR10
    else:
        raise AssertionError(f"Unhandled dataset: {dataset_name}")

    train_ds = dataset_cls(data_dir, train=True, download=True, transform=train_transform)
    test_ds = dataset_cls(data_dir, train=False, download=True, transform=eval_transform)
    return train_ds, test_ds


def split_train_val(train_ds, val_size=DEFAULT_VAL_SIZE, split_seed=DEFAULT_SPLIT_SEED):
    if not 0 <= val_size < len(train_ds):
        raise ValueError(f"val_size must be in [0, {len(train_ds) - 1}], got {val_size}")

    if val_size == 0:
        return train_ds, None

    train_size = len(train_ds) - val_size
    generator = torch.Generator().manual_seed(split_seed)
    return random_split(train_ds, [train_size, val_size], generator=generator)


def get_dataset_dataloaders(
    dataset_name,
    batch_size,
    *,
    data_dir="data",
    train_aug="none",
    val_size=DEFAULT_VAL_SIZE,
    split_seed=DEFAULT_SPLIT_SEED,
    train_num_workers=2,
    eval_num_workers=None,
    pin_memory=True,
    drop_last_train=False,
    eval_batch_size=None,
):
    train_full, test_ds = load_datasets(
        dataset_name,
        data_dir=data_dir,
        train_aug=train_aug,
        split_seed=split_seed,
    )
    train_ds, val_ds = split_train_val(
        train_full,
        val_size=val_size,
        split_seed=split_seed,
    )

    if eval_num_workers is None:
        eval_num_workers = train_num_workers
    if eval_batch_size is None:
        eval_batch_size = batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=eval_num_workers,
            pin_memory=pin_memory,
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def get_eval_dataset(
    dataset_name,
    *,
    split,
    data_dir="data",
    val_size=DEFAULT_VAL_SIZE,
    split_seed=DEFAULT_SPLIT_SEED,
):
    train_ds, test_ds = load_datasets(dataset_name, data_dir=data_dir, split_seed=split_seed)
    _, val_ds = split_train_val(
        train_ds,
        val_size=val_size,
        split_seed=split_seed,
    )

    if split == "val":
        if val_ds is None:
            raise ValueError("Validation split requested, but val_size=0")
        return val_ds
    if split == "test":
        return test_ds
    raise ValueError(f"Unknown split: {split}")


def denormalize_images(images, dataset_name):
    config = get_dataset_config(dataset_name)
    if config["kind"] != "image":
        raise ValueError(f"denormalize_images only supports image datasets, not '{dataset_name}'")
    images = torch.as_tensor(images).detach().cpu().float()
    mean = torch.tensor(config["mean"], dtype=images.dtype).view(1, -1, 1, 1)
    std = torch.tensor(config["std"], dtype=images.dtype).view(1, -1, 1, 1)
    return images * std + mean


def save_prediction_preview(
    images,
    labels,
    preds,
    out_path,
    *,
    dataset_name,
    max_items=16,
    cols=4,
    scale=4,
):
    config = get_dataset_config(dataset_name)
    if config["kind"] != "image":
        raise ValueError(f"save_prediction_preview only supports image datasets, not '{dataset_name}'")
    images = denormalize_images(images, dataset_name)
    labels = [int(x) for x in torch.as_tensor(labels).view(-1).tolist()]
    preds = [int(x) for x in torch.as_tensor(preds).view(-1).tolist()]
    count = min(len(labels), len(preds), len(images), max_items)
    if count <= 0:
        raise ValueError("No prediction samples provided for preview")

    cols = max(1, min(cols, count))
    rows = ceil(count / cols)
    image_h, image_w = config["image_size"]
    thumb_w = image_w * scale
    thumb_h = image_h * scale
    label_h = 18

    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for idx in range(count):
        row = idx // cols
        col = idx % cols
        x0 = col * thumb_w
        y0 = row * (thumb_h + label_h)

        img = images[idx].clamp(0.0, 1.0).mul(255).byte().permute(1, 2, 0).numpy()
        if config["channels"] == 1:
            tile = Image.fromarray(img[:, :, 0], mode="L")
            tile = tile.resize((thumb_w, thumb_h), resample=_NEAREST).convert("RGB")
        elif config["channels"] == 3:
            tile = Image.fromarray(img, mode="RGB").resize(
                (thumb_w, thumb_h), resample=_NEAREST
            )
        else:
            raise ValueError(f"Unsupported channel count: {config['channels']}")
        canvas.paste(tile, (x0, y0))

        ok = preds[idx] == labels[idx]
        color = (0, 128, 0) if ok else (180, 0, 0)
        draw.rectangle((x0, y0, x0 + thumb_w - 1, y0 + thumb_h - 1), outline=color, width=2)
        draw.text((x0 + 2, y0 + thumb_h + 2), f"t={labels[idx]} p={preds[idx]}", fill=color)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path
