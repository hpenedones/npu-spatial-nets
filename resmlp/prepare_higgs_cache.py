"""Download public HIGGS parquet splits and materialize a native-width HIGGS.pt cache."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, urlretrieve

import numpy as np
import pandas as pd
import torch

from resmlp.data_utils import _prepare_higgs_tensors


PARQUET_API_TEMPLATE = "https://datasets-server.huggingface.co/parquet?dataset={dataset}"


def fetch_parquet_urls(dataset):
    api_url = PARQUET_API_TEMPLATE.format(dataset=quote(dataset, safe=""))
    with urlopen(api_url, timeout=60) as response:
        payload = json.load(response)
    urls_by_split = defaultdict(list)
    for item in payload["parquet_files"]:
        if item.get("config") == "default":
            urls_by_split[item["split"]].append(item["url"])
    return dict(urls_by_split)


def ensure_download(url, path, *, force=False):
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urlretrieve(url, path)


def load_split(path):
    frame = pd.read_parquet(path)
    if "inputs" not in frame or "label" not in frame:
        raise ValueError(f"Expected parquet columns 'inputs' and 'label' in {path}")
    features = np.stack(frame["inputs"].to_list()).astype(np.float32, copy=False)
    labels = frame["label"].to_numpy(dtype=np.int64, copy=False)
    return _prepare_higgs_tensors(features, labels)


def combine_splits(split_names, urls, download_dir, *, prefix, force_download=False):
    split_features = []
    split_labels = []
    for split in split_names:
        if split not in urls:
            available = ", ".join(sorted(urls))
            raise ValueError(f"Unknown split '{split}'. Available splits: {available}")
        shard_urls = urls[split]
        for shard_idx, shard_url in enumerate(shard_urls):
            parquet_path = download_dir / f"{prefix}_{split}_{shard_idx:04d}.parquet"
            ensure_download(shard_url, parquet_path, force=force_download)
            features, labels = load_split(parquet_path)
            split_features.append(features)
            split_labels.append(labels)
    return torch.cat(split_features, dim=0), torch.cat(split_labels, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Prepare a split-aware local HIGGS.pt cache")
    parser.add_argument("--dataset", default="jxie/higgs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["train_63k", "val_16k"],
        help="Parquet splits to concatenate into the local training pool",
    )
    parser.add_argument(
        "--test-splits",
        nargs="+",
        default=["test_20k"],
        help="Parquet splits to concatenate into the local test split",
    )
    parser.add_argument("--cache-name", default="HIGGS.pt")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--keep-parquet", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    urls = fetch_parquet_urls(args.dataset)
    slug = args.dataset.replace("/", "_")

    train_features, train_labels = combine_splits(
        args.train_splits,
        urls,
        data_dir,
        prefix=f"{slug}_train",
        force_download=args.force_download,
    )
    test_features, test_labels = combine_splits(
        args.test_splits,
        urls,
        data_dir,
        prefix=f"{slug}_test",
        force_download=args.force_download,
    )

    cache_path = data_dir / args.cache_name
    torch.save(
        {
            "train_features": train_features,
            "train_labels": train_labels,
            "test_features": test_features,
            "test_labels": test_labels,
            "source_dataset": args.dataset,
            "source_train_splits": list(args.train_splits),
            "source_test_splits": list(args.test_splits),
        },
        cache_path,
    )
    print(
        f"Saved {cache_path} with train={train_features.shape[0]:,} "
        f"test={test_features.shape[0]:,} feature_dim={train_features.shape[1]}"
    )

    if not args.keep_parquet:
        for split in args.train_splits:
            for shard_idx, _ in enumerate(urls[split]):
                parquet_path = data_dir / f"{slug}_train_{split}_{shard_idx:04d}.parquet"
                if parquet_path.exists():
                    parquet_path.unlink()
                    print(f"Removed {parquet_path}")
        for split in args.test_splits:
            for shard_idx, _ in enumerate(urls[split]):
                parquet_path = data_dir / f"{slug}_test_{split}_{shard_idx:04d}.parquet"
                if parquet_path.exists():
                    parquet_path.unlink()
                    print(f"Removed {parquet_path}")


if __name__ == "__main__":
    main()
