"""Shared MNIST dataset and preview helpers for ResMLP workflows."""

from math import ceil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image, ImageDraw

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
DEFAULT_VAL_SIZE = 10_000
DEFAULT_SPLIT_SEED = 1234

_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


def mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])


def load_mnist_datasets(data_dir="data"):
    transform = mnist_transform()
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, transform=transform)
    return train_ds, test_ds


def split_mnist_train_val(train_ds, val_size=DEFAULT_VAL_SIZE, split_seed=DEFAULT_SPLIT_SEED):
    if not 0 <= val_size < len(train_ds):
        raise ValueError(f"val_size must be in [0, {len(train_ds) - 1}], got {val_size}")

    if val_size == 0:
        return train_ds, None

    train_size = len(train_ds) - val_size
    generator = torch.Generator().manual_seed(split_seed)
    return random_split(train_ds, [train_size, val_size], generator=generator)


def get_mnist_dataloaders(
    batch_size,
    *,
    data_dir="data",
    val_size=DEFAULT_VAL_SIZE,
    split_seed=DEFAULT_SPLIT_SEED,
    train_num_workers=2,
    eval_num_workers=None,
    pin_memory=True,
    drop_last_train=False,
    eval_batch_size=None,
):
    train_full, test_ds = load_mnist_datasets(data_dir)
    train_ds, val_ds = split_mnist_train_val(
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


def get_mnist_eval_dataset(
    *,
    split,
    data_dir="data",
    val_size=DEFAULT_VAL_SIZE,
    split_seed=DEFAULT_SPLIT_SEED,
):
    train_ds, test_ds = load_mnist_datasets(data_dir)
    _, val_ds = split_mnist_train_val(
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
    raise ValueError(f"Unknown MNIST split: {split}")


def denormalize_mnist(images):
    images = torch.as_tensor(images).detach().cpu().float()
    mean = torch.tensor(MNIST_MEAN, dtype=images.dtype).view(1, -1, 1, 1)
    std = torch.tensor(MNIST_STD, dtype=images.dtype).view(1, -1, 1, 1)
    return images * std + mean


def save_prediction_preview(
    images,
    labels,
    preds,
    out_path,
    *,
    max_items=16,
    cols=4,
    scale=4,
):
    images = denormalize_mnist(images)
    labels = [int(x) for x in torch.as_tensor(labels).view(-1).tolist()]
    preds = [int(x) for x in torch.as_tensor(preds).view(-1).tolist()]
    count = min(len(labels), len(preds), len(images), max_items)
    if count <= 0:
        raise ValueError("No prediction samples provided for preview")

    cols = max(1, min(cols, count))
    rows = ceil(count / cols)
    thumb_w = 28 * scale
    thumb_h = 28 * scale
    label_h = 18

    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for idx in range(count):
        row = idx // cols
        col = idx % cols
        x0 = col * thumb_w
        y0 = row * (thumb_h + label_h)

        img = images[idx].clamp(0.0, 1.0).mul(255).byte().squeeze(0).numpy()
        tile = Image.fromarray(img, mode="L").resize((thumb_w, thumb_h), resample=_NEAREST).convert("RGB")
        canvas.paste(tile, (x0, y0))

        ok = preds[idx] == labels[idx]
        color = (0, 128, 0) if ok else (180, 0, 0)
        draw.rectangle((x0, y0, x0 + thumb_w - 1, y0 + thumb_h - 1), outline=color, width=2)
        draw.text((x0 + 2, y0 + thumb_h + 2), f"t={labels[idx]} p={preds[idx]}", fill=color)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path
