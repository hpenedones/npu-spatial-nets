"""
Smoke tests for the native-width HIGGS data path.
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resmlp.data_utils import HIGGS_INPUT_DIM, _prepare_higgs_tensors, load_higgs_datasets


def test_prepare_higgs_tensors_keeps_native_width():
    features = torch.arange(2 * HIGGS_INPUT_DIM, dtype=torch.float32).reshape(2, HIGGS_INPUT_DIM)
    labels = torch.tensor([0, 1], dtype=torch.long)

    got_features, got_labels = _prepare_higgs_tensors(features, labels)

    assert got_features.shape == (2, HIGGS_INPUT_DIM)
    assert torch.equal(got_features, features)
    assert torch.equal(got_labels, labels)


def test_prepare_higgs_tensors_rejects_old_padded_width():
    try:
        _prepare_higgs_tensors(torch.zeros(1, 56), torch.tensor([0]))
    except ValueError as exc:
        assert f"{HIGGS_INPUT_DIM} columns" in str(exc)
        return
    raise AssertionError("Expected 56-wide HIGGS features to be rejected")


def test_load_higgs_datasets_uses_native_width_from_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "HIGGS.pt"
        train_features = torch.arange(4 * HIGGS_INPUT_DIM, dtype=torch.float32).reshape(4, HIGGS_INPUT_DIM)
        train_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        test_features = torch.arange(2 * HIGGS_INPUT_DIM, dtype=torch.float32).reshape(2, HIGGS_INPUT_DIM)
        test_labels = torch.tensor([1, 0], dtype=torch.long)
        torch.save(
            {
                "train_features": train_features,
                "train_labels": train_labels,
                "test_features": test_features,
                "test_labels": test_labels,
            },
            cache_path,
        )

        train_ds, test_ds = load_higgs_datasets(tmpdir)

        assert train_ds.features.shape == (4, HIGGS_INPUT_DIM)
        assert test_ds.features.shape == (2, HIGGS_INPUT_DIM)
        sample_x, sample_y = train_ds[0]
        assert sample_x.shape == (HIGGS_INPUT_DIM,)
        assert isinstance(sample_y, int)


def main():
    test_prepare_higgs_tensors_keeps_native_width()
    test_prepare_higgs_tensors_rejects_old_padded_width()
    test_load_higgs_datasets_uses_native_width_from_cache()
    print("HIGGS data tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
