# within_loio.py
#
# Purpose:
# - Within-dataset evaluation with leave-one-individual-out (LOIO) instead of the
#   single 25 % individual holdout used in the Within_* modes of the model scripts.
# - Each individual is the test unit once; the model is trained on the remaining
#   individuals of the same dataset. Predictions of all folds are pooled, so every
#   window of a dataset is predicted exactly once, as in the Cross_* mode.
# - Model definitions, settings, training cap, minimum counts, split unit, and
#   metric functions are imported unchanged from 01_CNN.py ... 06_LightGBM.py.
#
# Usage (from this folder, with the project venv):
#   python within_loio.py                         # all six models, seed 42, CPU
#   python within_loio.py --models RF LGBM        # selected models only
#   python within_loio.py --seeds 42 43 44 45 46  # repeated runs, averaged per dataset
#   python within_loio.py --device mps            # Apple GPU (not bit-reproducible)
#
# With several models, each model runs in its own process (avoids OpenMP clashes
# between PyTorch and LightGBM on macOS).
#
# Outputs: <base>/Output_checks/Within_LOIO/<model>/statistics/
#   metrics_all.csv           one row per dataset with complete LOIO coverage (used by H1.R)
#   metrics_all_incl_partial.csv  same, incl. datasets in which a fold had to be skipped
#   behavior_metrics_all.csv  class-wise metrics for the same datasets
#   metrics_by_seed.csv       pooled metrics per dataset and seed, incl. incomplete datasets
#   fold_metrics.csv          one row per fold (test individual), incl. skipped folds
#   predictions.csv.gz        window-level truth and prediction for all folds
#   run_log.txt               settings and skip notes
# and <base>/Output_checks/Within_LOIO/comparison_holdout_vs_loio.csv

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

MODEL_SPECS = {
    "CNN": {"file": "01_CNN.py", "fit": "fit_cnn_model", "input": "raw", "torch": True},
    "ResNet": {"file": "02_ResNet.py", "fit": "fit_resnet_model", "input": "raw", "torch": True},
    "HYDRA": {"file": "03_HYDRA.py", "fit": "fit_hydra_model", "input": "raw", "torch": False},
    "MultiRocket": {"file": "04_MultiRocket.py", "fit": "fit_multirocket_model", "input": "raw", "torch": False},
    "RF": {"file": "05_Random_Forest.py", "fit": "fit_rf_model", "input": "features", "torch": False},
    "LGBM": {"file": "06_LightGBM.py", "fit": "fit_lgbm_model", "input": "features", "torch": False},
}

# Raw windows are loaded once with the HYDRA loader and features once with the RF
# loader; check_splits_and_training.py confirmed both yield the same 146,667 windows.
RAW_LOADER_FILE = "03_HYDRA.py"
FEATURE_LOADER_FILE = "05_Random_Forest.py"


# =========================================================
# Helpers
# =========================================================
def log(msg: str) -> None:
    print(msg, flush=True)


def load_module(file_name: str) -> ModuleType:
    alias = "loio_" + Path(file_name).stem.lower()
    spec = importlib.util.spec_from_file_location(alias, HERE / file_name)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {file_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_base(module: ModuleType, base: Path) -> None:
    module.BASE = base
    if hasattr(module, "RAW_ROOT"):
        module.RAW_ROOT = base / "Data_Processed" / "data_processed_final"
    if hasattr(module, "FEATURE_ROOT"):
        module.FEATURE_ROOT = base / "Data_Processed" / "data_processed_features"


def load_input(kind: str, base: Path) -> tuple[pd.DataFrame, tuple[list[str], list[str], list[str]] | None]:
    if kind == "raw":
        loader = load_module(RAW_LOADER_FILE)
        set_base(loader, base)
        all_dt, _ = loader.load_raw_data()
        all_dt, x_cols, y_cols, z_cols = loader.clean_raw_data(all_dt)
        axis_cols = (x_cols, y_cols, z_cols)
    else:
        loader = load_module(FEATURE_LOADER_FILE)
        set_base(loader, base)
        all_dt, _ = loader.load_feature_data()
        all_dt = loader.clean_feature_data(all_dt)
        axis_cols = None

    all_dt["analysis_dataset"] = all_dt["analysis_dataset"].astype(str)
    all_dt["behavior"] = all_dt["behavior"].astype(str)
    return all_dt, axis_cols


def call_fit(name: str, module: ModuleType, train_dt: pd.DataFrame, test_dt: pd.DataFrame, axis_cols, device) -> dict:
    spec = MODEL_SPECS[name]
    fit_fn = getattr(module, spec["fit"])
    if spec["input"] == "features":
        return fit_fn(train_dt, test_dt)
    x_cols, y_cols, z_cols = axis_cols
    if spec["torch"]:
        return fit_fn(train_dt, test_dt, x_cols, y_cols, z_cols, device)
    return fit_fn(train_dt, test_dt, x_cols, y_cols, z_cols)


# =========================================================
# LOIO
# =========================================================
def run_loio(name: str, module: ModuleType, all_dt: pd.DataFrame, axis_cols, seed: int, device) -> tuple[pd.DataFrame, pd.DataFrame]:
    behaviors = list(module.KEEP_BEHAVIORS)
    min_n = module.MIN_N_WITHIN_DATASET_BEHAVIOR
    cap = module.CAP_N_WITHIN_TRAIN_BEHAVIOR

    # The fit functions read the module-level SEED for random_state.
    module.SEED = seed

    # Same dataset eligibility as the Within_* modes.
    counts = all_dt.groupby(["analysis_dataset", "behavior"]).size().reset_index(name="N")
    ready = counts[counts["N"] >= min_n].groupby("analysis_dataset")["behavior"].nunique()
    ready = sorted(ready[ready == len(behaviors)].index)

    fold_rows: list[dict] = []
    pred_parts: list[pd.DataFrame] = []

    for ds in ready:
        ds_dt = all_dt[all_dt["analysis_dataset"] == ds]
        split_var = module.choose_split_var(ds_dt)
        if split_var is None:
            fold_rows.append({"model": name, "seed": seed, "dataset": ds, "status": "skipped: no valid split unit"})
            continue

        keys = ds_dt[split_var].fillna("").astype(str).str.strip()
        units = sorted(keys[keys != ""].unique())

        for i, unit in enumerate(units, start=1):
            test_dt = ds_dt[keys == unit]
            train_dt = ds_dt[keys != unit]
            train_counts = train_dt.groupby("behavior").size()
            test_counts = test_dt.groupby("behavior").size()

            row: dict[str, object] = {
                "model": name,
                "seed": seed,
                "dataset": ds,
                "split_var": split_var,
                "test_unit": unit,
                "n_test": len(test_dt),
            }
            for b in behaviors:
                row[f"train_{b}"] = int(train_counts.get(b, 0))
                row[f"test_{b}"] = int(test_counts.get(b, 0))

            # Same fold rules as the Within_* modes.
            if not all(train_counts.get(b, 0) >= min_n for b in behaviors):
                row["status"] = f"skipped: train set below {min_n} windows in a behavior"
                fold_rows.append(row)
                continue
            if sum(test_counts.get(b, 0) >= 1 for b in behaviors) < 2:
                row["status"] = "skipped: test set has fewer than 2 behaviors"
                fold_rows.append(row)
                continue

            train_bal = module.cap_by_groups(train_dt, ["behavior"], cap, seed)
            if hasattr(module, "set_seed"):
                module.set_seed(seed)

            log(f"[INFO] {name} | seed {seed} | {ds} | fold {i}/{len(units)} | test unit {unit}")
            start = time.time()
            fit = call_fit(name, module, train_bal, test_dt, axis_cols, device)

            metrics = pd.DataFrame(fit["overall_metrics"]).iloc[0]
            row.update(
                {
                    "status": "used",
                    "n_train": len(train_bal),
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "seconds": round(time.time() - start, 1),
                }
            )
            history = fit.get("history_df")
            if isinstance(history, pd.DataFrame) and not history.empty:
                row["final_train_loss"] = float(history["train_loss"].iloc[-1])
            fold_rows.append(row)

            part = pd.DataFrame(
                {
                    "model": name,
                    "seed": seed,
                    "analysis_dataset": ds,
                    "test_unit": unit,
                    "truth": pd.Series(fit["truth"]).loc[test_dt.index].astype(str).to_numpy(),
                    "pred": pd.Series(fit["pred"]).loc[test_dt.index].astype(str).to_numpy(),
                },
                index=test_dt.index,
            )
            for col in ["window_id", "window_start", "source_file"]:
                if col in test_dt.columns:
                    part[col] = test_dt[col].to_numpy()
            pred_parts.append(part)

    folds = pd.DataFrame(fold_rows)
    preds = pd.concat(pred_parts) if pred_parts else pd.DataFrame()
    return folds, preds


def pooled_metrics(module: ModuleType, all_dt: pd.DataFrame, folds: pd.DataFrame, preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_windows = all_dt.groupby("analysis_dataset").size()
    metric_rows: list[pd.DataFrame] = []
    behavior_rows: list[pd.DataFrame] = []

    if preds.empty:
        return pd.DataFrame(), pd.DataFrame()

    for (seed, ds), sub in preds.groupby(["seed", "analysis_dataset"]):
        ds_folds = folds[(folds["seed"] == seed) & (folds["dataset"] == ds)]
        n_used = int((ds_folds["status"] == "used").sum())
        coverage = len(sub) / int(n_windows[ds])

        m = module.get_metrics(sub["truth"], sub["pred"])
        m["analysis_dataset"] = ds
        m["seed"] = seed
        m["n_windows"] = len(sub)
        m["n_folds_used"] = n_used
        m["n_folds_total"] = len(ds_folds)
        m["coverage"] = round(coverage, 4)
        m["complete"] = bool(n_used == len(ds_folds) and coverage == 1.0)
        metric_rows.append(m)

        b = module.get_behavior_metrics(sub["truth"], sub["pred"])
        b["analysis_dataset"] = ds
        b["seed"] = seed
        b["complete"] = m["complete"].iloc[0]
        behavior_rows.append(b)

    return pd.concat(metric_rows, ignore_index=True), pd.concat(behavior_rows, ignore_index=True)


def write_model_outputs(name: str, out_root: Path, folds: pd.DataFrame, preds: pd.DataFrame,
                        by_seed: pd.DataFrame, behavior_by_seed: pd.DataFrame, settings: list[str]) -> pd.DataFrame:
    stats_dir = out_root / name / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    folds.to_csv(stats_dir / "fold_metrics.csv", index=False)
    if not preds.empty:
        preds.to_csv(stats_dir / "predictions.csv.gz", index=False, compression="gzip")
    by_seed.to_csv(stats_dir / "metrics_by_seed.csv", index=False)

    metric_cols = ["accuracy", "macro_recall", "macro_precision", "macro_f1"]

    def average_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=metric_cols + ["analysis_dataset"])
        out = df.groupby("analysis_dataset")[metric_cols].mean().reset_index()
        sd = df.groupby("analysis_dataset")["macro_f1"].std().rename("macro_f1_sd_seeds")
        info = df.groupby("analysis_dataset").agg(
            n_seeds=("seed", "nunique"), n_windows=("n_windows", "first"),
            n_folds_used=("n_folds_used", "first"), n_folds=("n_folds_total", "first"),
            coverage=("coverage", "first"),
        )
        out = out.join(sd, on="analysis_dataset").join(info, on="analysis_dataset")
        return out.sort_values("macro_f1", ascending=False)

    # Primary: datasets in which every individual could be the test unit, so every
    # window is predicted once. Sensitivity: also datasets with skipped folds.
    complete = by_seed[by_seed["complete"]] if not by_seed.empty else by_seed
    metrics_all = average_over_seeds(complete)
    metrics_all.to_csv(stats_dir / "metrics_all.csv", index=False)
    average_over_seeds(by_seed).to_csv(stats_dir / "metrics_all_incl_partial.csv", index=False)

    partial = by_seed[~by_seed["complete"]] if not by_seed.empty else by_seed
    partial.to_csv(stats_dir / "metrics_partial.csv", index=False)

    if not behavior_by_seed.empty:
        b = behavior_by_seed[behavior_by_seed["complete"]]
        b = (
            b.groupby(["analysis_dataset", "behavior"])
            .agg(support=("support", "first"), predicted=("predicted", "mean"), tp=("tp", "mean"),
                 recall=("recall", "mean"), precision=("precision", "mean"), f1=("f1", "mean"))
            .reset_index()
        )
        b.to_csv(stats_dir / "behavior_metrics_all.csv", index=False)

    skipped = folds[folds["status"] != "used"] if "status" in folds.columns else folds
    lines = settings + ["", f"Folds used: {int((folds['status'] == 'used').sum())} of {len(folds)}"]
    if not skipped.empty:
        lines += ["", "Skipped folds:"]
        lines += [f"  seed {r.seed} | {r.dataset} | test unit {getattr(r, 'test_unit', '')} | {r.status}" for r in skipped.itertuples()]
    if not partial.empty:
        lines += ["", "Datasets without complete LOIO coverage (excluded from metrics_all.csv):"]
        lines += [f"  seed {r.seed} | {r.analysis_dataset} | coverage {r.coverage}" for r in partial.itertuples()]
    (stats_dir / "run_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    log(f"[OK] {name}: outputs written to {stats_dir}")
    return metrics_all


def holdout_comparison(base: Path, out_root: Path, names: list[str]) -> None:
    rows = []
    for name in names:
        loio_file = out_root / name / "statistics" / "metrics_all.csv"
        holdout_file = base / "Models" / name / f"Within_{name}" / "statistics" / "metrics_all.csv"
        if not loio_file.exists() or not holdout_file.exists():
            continue
        loio = pd.read_csv(loio_file)[["analysis_dataset", "macro_f1"]].rename(columns={"macro_f1": "loio_macro_f1"})
        hold = pd.read_csv(holdout_file)[["analysis_dataset", "macro_f1"]].rename(columns={"macro_f1": "holdout_macro_f1"})
        both = hold.merge(loio, on="analysis_dataset", how="outer")
        both.insert(0, "model", name)
        rows.append(both)
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out["difference"] = out["loio_macro_f1"] - out["holdout_macro_f1"]
        out.to_csv(out_root / "comparison_holdout_vs_loio.csv", index=False)
        log(f"[OK] Comparison written to {out_root / 'comparison_holdout_vs_loio.csv'}")


# =========================================================
# Main
# =========================================================
def run_models_separately(args: argparse.Namespace) -> int:
    failed = []
    for name in args.models:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--models", name,
               "--seeds", *map(str, args.seeds), "--device", args.device]
        if args.base is not None:
            cmd += ["--base", str(args.base)]
        log(f"\n[INFO] ===== Starting {name} in a separate process =====")
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
            log(f"[ERROR] {name} failed; see the messages above. Continuing with the next model.")

    if failed:
        log(f"\n[ERROR] Failed models: {', '.join(failed)}")
    else:
        log("\n[OK] All models finished.")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Within-dataset leave-one-individual-out evaluation.")
    parser.add_argument("--base", type=Path, default=None, help="Analysis root folder (default: BASE of the model scripts)")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                        help="Device for CNN/ResNet; cpu is deterministic, mps is faster but not bit-reproducible")
    args = parser.parse_args()

    # Several models: run each in its own process. PyTorch (CNN, ResNet, HYDRA via
    # aeon) and LightGBM ship separate OpenMP runtimes, which can abort a shared
    # process on macOS ("OMP: Error #15").
    if len(args.models) > 1:
        return run_models_separately(args)

    device = None
    if any(MODEL_SPECS[m]["torch"] for m in args.models):
        import torch

        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        device = torch.device(args.device)
        if args.device == "cpu":
            torch.use_deterministic_algorithms(True, warn_only=True)

    data_cache: dict[str, tuple[pd.DataFrame, object]] = {}
    base = args.base

    for name in args.models:
        module = load_module(MODEL_SPECS[name]["file"])
        if base is None:
            base = Path(module.BASE)
        set_base(module, base)
        out_root = base / "Output_checks" / "Within_LOIO"

        kind = MODEL_SPECS[name]["input"]
        if kind not in data_cache:
            log(f"[INFO] Loading {kind} input...")
            data_cache[kind] = load_input(kind, base)
        all_dt, axis_cols = data_cache[kind]

        settings = [
            f"Model: {name} ({MODEL_SPECS[name]['file']}, {MODEL_SPECS[name]['fit']})",
            "Evaluation: within-dataset leave-one-individual-out, predictions pooled per dataset",
            f"Seeds: {', '.join(map(str, args.seeds))}",
            f"Device: {device if MODEL_SPECS[name]['torch'] else 'n/a'}",
            f"Training cap per behavior: {module.CAP_N_WITHIN_TRAIN_BEHAVIOR}",
            f"Minimum training windows per behavior: {module.MIN_N_WITHIN_DATASET_BEHAVIOR}",
            f"Windows loaded: {len(all_dt)}",
        ]

        fold_parts, pred_parts, seed_parts, behavior_parts = [], [], [], []
        for seed in args.seeds:
            folds, preds = run_loio(name, module, all_dt, axis_cols, seed, device)
            by_seed, behavior_by_seed = pooled_metrics(module, all_dt, folds, preds)
            fold_parts.append(folds)
            pred_parts.append(preds)
            seed_parts.append(by_seed)
            behavior_parts.append(behavior_by_seed)

        write_model_outputs(
            name,
            out_root,
            pd.concat(fold_parts, ignore_index=True),
            pd.concat(pred_parts) if any(not p.empty for p in pred_parts) else pd.DataFrame(),
            pd.concat(seed_parts, ignore_index=True),
            pd.concat(behavior_parts, ignore_index=True),
            settings,
        )

    holdout_comparison(base, base / "Output_checks" / "Within_LOIO", list(MODEL_SPECS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
