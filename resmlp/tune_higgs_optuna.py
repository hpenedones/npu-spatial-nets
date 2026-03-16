"""Run full-data HIGGS hyperparameter sweeps with Optuna and MLflow.

This driver keeps the heavyweight HIGGS dataset resident within one Python
process, runs one trial at a time on the requested device, logs per-epoch
metrics to a local MLflow experiment store, and persists the Optuna study in a
SQLite database so the search can be resumed later.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import mlflow
import optuna
from optuna.exceptions import TrialPruned
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from resmlp.data_utils import DEFAULT_SPLIT_SEED, DEFAULT_VAL_SIZE, load_datasets, split_train_val
from resmlp.model import ResMLP
from resmlp.train import (
    build_checkpoint,
    build_optimizer,
    build_scheduler,
    evaluate,
    resolve_device,
    set_seed,
    train_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna/MLflow sweep for full-data HIGGS")
    parser.add_argument("--dataset", default="higgs")
    parser.add_argument("--data-dir", default="data/higgs_full")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--study-name", default="higgs-full-overnight")
    parser.add_argument("--experiment-name", default="higgs-full-optuna")
    parser.add_argument("--tracking-dir", default="mlruns")
    parser.add_argument("--storage", default="sqlite:///build/higgs_optuna.db")
    parser.add_argument("--save-root", default="build/higgs_optuna")
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--val-size", type=int, default=100_000)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-num-workers", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=32_768)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--num-layers-options", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--lr-min", type=float, default=1e-3)
    parser.add_argument("--lr-max", type=float, default=4e-3)
    parser.add_argument("--weight-decay-min", type=float, default=1e-5)
    parser.add_argument("--weight-decay-max", type=float, default=5e-3)
    parser.add_argument("--min-lr-ratio-min", type=float, default=0.02)
    parser.add_argument("--min-lr-ratio-max", type=float, default=0.2)
    parser.add_argument("--label-smoothing-max", type=float, default=0.05)
    parser.add_argument("--allow-residual-bias", action="store_true")
    parser.add_argument("--prune-after", type=int, default=12)
    parser.add_argument("--no-prune", action="store_true")
    parser.add_argument("--sampler-seed", type=int, default=1234)
    return parser.parse_args()


def ensure_parent_dirs(args):
    tracking_dir = Path(args.tracking_dir)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    if args.storage.startswith("sqlite:///"):
        sqlite_path = Path(args.storage.removeprefix("sqlite:///"))
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return tracking_dir, save_root


def build_trial_args(args, params):
    return SimpleNamespace(
        dataset=args.dataset,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        residual_bias=params["residual_bias"],
        val_size=args.val_size,
        split_seed=args.split_seed,
        seed=params["seed"],
        train_aug="none",
        optimizer="adamw",
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        label_smoothing=params["label_smoothing"],
        scheduler="cosine",
        momentum=0.9,
        min_lr=params["min_lr"],
        batch_size=params["batch_size"],
        epochs=args.max_epochs,
    )


@torch.no_grad()
def score_classifier(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs = []
    all_labels = []
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "log_loss": float(log_loss(labels, probs, labels=[0, 1])),
    }


def make_loader(dataset, *, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def sample_params(trial, args):
    lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
    min_lr_ratio = trial.suggest_float(
        "min_lr_ratio",
        args.min_lr_ratio_min,
        args.min_lr_ratio_max,
        log=True,
    )
    residual_bias_options = [False, True] if args.allow_residual_bias else [False]
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", args.hidden_dims),
        "num_layers": trial.suggest_categorical("num_layers", args.num_layers_options),
        "batch_size": trial.suggest_categorical("batch_size", args.batch_sizes),
        "lr": lr,
        "weight_decay": trial.suggest_float(
            "weight_decay",
            args.weight_decay_min,
            args.weight_decay_max,
            log=True,
        ),
        "min_lr_ratio": min_lr_ratio,
        "min_lr": max(1e-6, lr * min_lr_ratio),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, args.label_smoothing_max),
        "residual_bias": trial.suggest_categorical("residual_bias", residual_bias_options),
        "seed": args.seed + trial.number,
    }


def objective_factory(args, train_ds, val_ds, test_ds, device, save_root):
    pin_memory = device == "cuda"
    input_dim = train_ds.dataset.features.shape[1] if hasattr(train_ds, "dataset") else 28

    def objective(trial):
        params = sample_params(trial, args)
        set_seed(params["seed"])
        trial_dir = save_root / args.study_name / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"{args.study_name}-trial-{trial.number:04d}"

        train_loader = make_loader(
            train_ds,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=args.train_num_workers,
            pin_memory=pin_memory,
        )
        eval_batch_size = max(args.eval_batch_size, params["batch_size"])
        val_loader = make_loader(
            val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.eval_num_workers,
            pin_memory=pin_memory,
        )
        test_loader = make_loader(
            test_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.eval_num_workers,
            pin_memory=pin_memory,
        )

        trial_args = build_trial_args(args, params)
        model = ResMLP(
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            input_dim=input_dim,
            num_classes=2,
            residual_bias=params["residual_bias"],
        ).to(device)
        optimizer = build_optimizer(trial_args, model.parameters())
        scheduler = build_scheduler(trial_args, optimizer)
        criterion = nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"])

        best_val_acc = float("-inf")
        best_epoch = -1
        best_path = trial_dir / "resmlp_best.pt"
        summary_path = trial_dir / "summary.json"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "dataset": args.dataset,
                    "data_dir": args.data_dir,
                    "max_epochs": args.max_epochs,
                    "val_size": args.val_size,
                    "split_seed": args.split_seed,
                    "device": device,
                    **params,
                }
            )
            mlflow.set_tags(
                {
                    "study_name": args.study_name,
                    "trial_number": str(trial.number),
                    "model_family": "resmlp",
                }
            )

            try:
                for epoch in range(args.max_epochs):
                    t0 = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
                    t1 = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
                    start = None
                    if t0 is not None:
                        t0.record()
                    else:
                        start = time.time()

                    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
                    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

                    if scheduler is not None:
                        scheduler.step()

                    if t1 is not None:
                        t1.record()
                        torch.cuda.synchronize()
                        elapsed = t0.elapsed_time(t1) / 1000.0
                    else:
                        elapsed = time.time() - start

                    samples_per_sec = len(train_loader.dataset) / max(elapsed, 1e-6)
                    lr = optimizer.param_groups[0]["lr"]

                    mlflow.log_metrics(
                        {
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "lr": lr,
                            "train_samples_per_sec": samples_per_sec,
                        },
                        step=epoch,
                    )

                    checkpoint = build_checkpoint(
                        trial_args, epoch, model, optimizer, scheduler, "val", val_loss, val_acc
                    )
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = epoch
                        torch.save(checkpoint, best_path)
                        mlflow.log_metric("best_val_acc", best_val_acc, step=epoch)

                    if (epoch + 1) % 10 == 0 or epoch == args.max_epochs - 1:
                        torch.save(checkpoint, trial_dir / f"resmlp_epoch{epoch:03d}.pt")

                    trial.report(best_val_acc, step=epoch)
                    if not args.no_prune and epoch + 1 >= args.prune_after and trial.should_prune():
                        payload = {
                            "status": "pruned",
                            "best_val_acc": best_val_acc,
                            "best_epoch": best_epoch,
                            "params": params,
                        }
                        summary_path.write_text(json.dumps(payload, indent=2) + "\n")
                        mlflow.log_artifact(summary_path)
                        mlflow.set_tag("status", "pruned")
                        raise TrialPruned()

                best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
                best_model = ResMLP(
                    hidden_dim=best_checkpoint["hidden_dim"],
                    num_layers=best_checkpoint["num_layers"],
                    input_dim=best_checkpoint["input_dim"],
                    num_classes=best_checkpoint["num_classes"],
                    residual_bias=bool(best_checkpoint.get("residual_bias", False)),
                ).to(device)
                best_model.load_state_dict(best_checkpoint["model"])
                test_metrics = score_classifier(best_model, test_loader, device, criterion)
                payload = {
                    "status": "completed",
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                    "test_metrics": test_metrics,
                    "params": params,
                    "best_checkpoint": str(best_path),
                }
                summary_path.write_text(json.dumps(payload, indent=2) + "\n")
                mlflow.log_metrics(
                    {
                        "test_accuracy": test_metrics["accuracy"],
                        "test_roc_auc": test_metrics["roc_auc"],
                        "test_pr_auc": test_metrics["pr_auc"],
                        "test_log_loss": test_metrics["log_loss"],
                    }
                )
                mlflow.log_artifact(summary_path)
                mlflow.log_artifact(best_path)
                mlflow.set_tag("status", "completed")
                trial.set_user_attr("summary_path", str(summary_path))
                trial.set_user_attr("best_checkpoint", str(best_path))
                trial.set_user_attr("best_epoch", best_epoch)
                trial.set_user_attr("test_accuracy", test_metrics["accuracy"])
                trial.set_user_attr("test_roc_auc", test_metrics["roc_auc"])
                return best_val_acc
            except TrialPruned:
                raise
            except Exception:
                trace_path = trial_dir / "traceback.txt"
                trace_path.write_text(traceback.format_exc())
                mlflow.log_artifact(trace_path)
                mlflow.set_tag("status", "failed")
                raise
            finally:
                del train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion
                if device == "cuda":
                    torch.cuda.empty_cache()

    return objective


def enqueue_baselines(study, args):
    baseline_trials = [
        {
            "hidden_dim": 32,
            "num_layers": 32,
            "batch_size": 8192,
            "lr": 3e-3,
            "weight_decay": 1e-4,
            "min_lr_ratio": 1 / 30,
            "label_smoothing": 0.0,
            "residual_bias": False,
        },
        {
            "hidden_dim": 32,
            "num_layers": 8,
            "batch_size": 8192,
            "lr": 3e-3,
            "weight_decay": 1e-4,
            "min_lr_ratio": 1 / 30,
            "label_smoothing": 0.0,
            "residual_bias": False,
        },
        {
            "hidden_dim": 32,
            "num_layers": 32,
            "batch_size": 8192,
            "lr": 2e-3,
            "weight_decay": 1e-3,
            "min_lr_ratio": 0.05,
            "label_smoothing": 0.0,
            "residual_bias": False,
        },
    ]
    residual_bias_options = [False, True] if args.allow_residual_bias else [False]
    for params in baseline_trials:
        if params["hidden_dim"] not in args.hidden_dims:
            continue
        if params["num_layers"] not in args.num_layers_options:
            continue
        if params["batch_size"] not in args.batch_sizes:
            continue
        if params["residual_bias"] not in residual_bias_options:
            continue
        if not (args.lr_min <= params["lr"] <= args.lr_max):
            continue
        if not (args.weight_decay_min <= params["weight_decay"] <= args.weight_decay_max):
            continue
        if not (args.min_lr_ratio_min <= params["min_lr_ratio"] <= args.min_lr_ratio_max):
            continue
        if not (0.0 <= params["label_smoothing"] <= args.label_smoothing_max):
            continue
        study.enqueue_trial(params, skip_if_exists=True)


def main():
    args = parse_args()
    tracking_dir, save_root = ensure_parent_dirs(args)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(args.experiment_name)

    device = resolve_device(args.device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    set_seed(args.seed)
    train_full, test_ds = load_datasets(args.dataset, data_dir=args.data_dir, split_seed=args.split_seed)
    train_ds, val_ds = split_train_val(train_full, val_size=args.val_size, split_seed=args.split_seed)
    if val_ds is None:
        raise ValueError("This sweep expects a validation split; choose val_size > 0")

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True)
    if args.no_prune:
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=args.prune_after)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    enqueue_baselines(study, args)

    objective = objective_factory(args, train_ds, val_ds, test_ds, device, save_root)
    timeout_seconds = None if args.timeout_hours <= 0 else int(math.ceil(args.timeout_hours * 3600))
    study.optimize(objective, n_trials=args.n_trials, timeout=timeout_seconds, gc_after_trial=True)

    best_payload = {
        "study_name": args.study_name,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_trial.params,
        "best_user_attrs": study.best_trial.user_attrs,
    }
    best_path = save_root / args.study_name / "best_trial.json"
    best_path.write_text(json.dumps(best_payload, indent=2) + "\n")
    print(json.dumps(best_payload, indent=2))


if __name__ == "__main__":
    main()
