# check_splits_and_training.py
#
# Purpose:
# - Read-only diagnostics for the model outputs; no model is trained.
# - A: Replicate the within-dataset holdout split for the raw-window input
#      (CNN, ResNet, HYDRA, MultiRocket) and the feature input (RF, LGBM),
#      using the split functions and settings imported from the model scripts.
# - B: Check whether raw and feature inputs contain the same windows per
#      dataset and behavior, and the same within-dataset test individuals.
# - C: Summarize CNN/ResNet training histories (final loss, gradient steps)
#      per evaluation mode from the existing training_history.csv files.
#
# Usage:
#   python check_splits_and_training.py
#   python check_splits_and_training.py --base "/Volumes/Z Slim/11_05_2026_Data_Analysis"
#
# Outputs are written to <base>/Output_checks/splits_and_training/.

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Same value as BATCH_SIZE in 01_CNN.py and 02_ResNet.py.
DL_BATCH_SIZE = 256
DL_MODELS = ["CNN", "ResNet"]
DL_MODES = ["Global", "Within", "Inter", "Cross", "Pairwise"]


# =========================================================
# Module loading
# =========================================================
def load_module(file_name: str, alias: str) -> ModuleType:
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


# =========================================================
# A: Within-dataset split audit
# =========================================================
def audit_within_splits(module: ModuleType, all_dt: pd.DataFrame, source: str) -> pd.DataFrame:
    behaviors = list(module.KEEP_BEHAVIORS)
    min_n = module.MIN_N_WITHIN_DATASET_BEHAVIOR

    all_dt = all_dt.copy()
    all_dt["analysis_dataset"] = all_dt["analysis_dataset"].astype(str)
    all_dt["behavior"] = all_dt["behavior"].astype(str)

    counts = all_dt.groupby(["analysis_dataset", "behavior"]).size().reset_index(name="N")
    ready = (
        counts[counts["N"] >= min_n]
        .groupby("analysis_dataset")["behavior"].nunique()
    )
    ready = set(ready[ready == len(behaviors)].index)

    rows = []
    for ds in sorted(all_dt["analysis_dataset"].unique()):
        row: dict[str, object] = {"source": source, "dataset": ds}
        ds_dt = all_dt[all_dt["analysis_dataset"] == ds].copy()

        if ds not in ready:
            row["status"] = f"skipped: fewer than {min_n} windows in a behavior"
            rows.append(row)
            continue

        split_dt, _ = module.make_holdout_split(ds_dt, "analysis_dataset", module.TEST_FRAC, module.SEED)
        if split_dt is None:
            row["status"] = "skipped: no valid split unit"
            rows.append(row)
            continue

        train_dt = split_dt[~split_dt["in_test"]]
        test_dt = split_dt[split_dt["in_test"]]
        train_counts = train_dt.groupby("behavior").size()
        test_counts = test_dt.groupby("behavior").size()

        if not all(train_counts.get(b, 0) >= min_n for b in behaviors):
            status = "skipped: train set below minimum in a behavior"
        elif sum(test_counts.get(b, 0) >= 1 for b in behaviors) < 2:
            status = "skipped: test set has fewer than 2 behaviors"
        else:
            status = "used"

        units = split_dt["split_key"]
        test_units = sorted(test_dt["split_key"].unique())
        row.update(
            {
                "status": status,
                "split_var": split_dt["split_var_used"].iloc[0],
                "n_units": int(units[units != ""].nunique()),
                "n_test_units": len(test_units),
                "test_units": ";".join(test_units),
                "n_train_windows": len(train_dt),
                "n_test_windows": len(test_dt),
                "test_window_share": round(len(test_dt) / len(split_dt), 3),
            }
        )
        for b in behaviors:
            row[f"train_{b}"] = int(train_counts.get(b, 0))
            row[f"test_{b}"] = int(test_counts.get(b, 0))
        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# B: Raw vs feature input comparison
# =========================================================
def compare_window_counts(raw_dt: pd.DataFrame, feat_dt: pd.DataFrame) -> pd.DataFrame:
    def count(df: pd.DataFrame, name: str) -> pd.DataFrame:
        return (
            df.assign(analysis_dataset=df["analysis_dataset"].astype(str), behavior=df["behavior"].astype(str))
            .groupby(["analysis_dataset", "behavior"]).size().reset_index(name=name)
        )

    out = count(raw_dt, "n_raw").merge(count(feat_dt, "n_features"), how="outer", on=["analysis_dataset", "behavior"])
    out = out.fillna(0)
    out["difference"] = out["n_raw"] - out["n_features"]
    return out.sort_values(["analysis_dataset", "behavior"]).reset_index(drop=True)


def compare_test_units(raw_audit: pd.DataFrame, feat_audit: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "status", "split_var", "n_units", "test_units", "n_test_windows"]
    raw = raw_audit.reindex(columns=cols).add_suffix("_raw").rename(columns={"dataset_raw": "dataset"})
    feat = feat_audit.reindex(columns=cols).add_suffix("_features").rename(columns={"dataset_features": "dataset"})
    out = raw.merge(feat, how="outer", on="dataset")
    out["same_split_var"] = out["split_var_raw"] == out["split_var_features"]
    out["same_test_units"] = out["test_units_raw"] == out["test_units_features"]
    return out


# =========================================================
# C: Training histories (CNN, ResNet)
# =========================================================
def collect_training_histories(base: Path) -> pd.DataFrame:
    rows = []
    for model in DL_MODELS:
        for mode in DL_MODES:
            stats_root = base / "Models" / model / f"{mode}_{model}" / "statistics"
            if not stats_root.exists():
                continue
            for hist_file in sorted(stats_root.rglob("training_history.csv")):
                hist = pd.read_csv(hist_file)
                if hist.empty or "train_loss" not in hist.columns:
                    continue
                loss = hist["train_loss"].to_numpy(dtype=float)

                n_train = np.nan
                balance_file = hist_file.parent / "train_balance.csv"
                if balance_file.exists():
                    n_train = float(pd.read_csv(balance_file)["N"].sum())
                batches = math.ceil(n_train / DL_BATCH_SIZE) if np.isfinite(n_train) else np.nan

                tail_ref = loss[-6] if len(loss) >= 6 else loss[0]
                rows.append(
                    {
                        "model": model,
                        "mode": mode,
                        "run": str(hist_file.parent.relative_to(stats_root)),
                        "epochs": len(loss),
                        "n_train": n_train,
                        "gradient_steps": batches * len(loss) if np.isfinite(batches) else np.nan,
                        "loss_first": loss[0],
                        "loss_last": loss[-1],
                        # Relative loss change over the last five epochs; values far
                        # below zero indicate the loss was still decreasing.
                        "rel_change_last5": (loss[-1] - tail_ref) / tail_ref if tail_ref > 0 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def summarize_histories(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    return (
        runs.groupby(["model", "mode"])
        .agg(
            n_runs=("run", "size"),
            median_n_train=("n_train", "median"),
            median_gradient_steps=("gradient_steps", "median"),
            median_loss_last=("loss_last", "median"),
            max_loss_last=("loss_last", "max"),
            median_rel_change_last5=("rel_change_last5", "median"),
        )
        .reset_index()
    )


# =========================================================
# Main
# =========================================================
def main() -> int:
    raw_mod = load_module("03_HYDRA.py", "hydra_script")
    feat_mod = load_module("05_Random_Forest.py", "rf_script")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=raw_mod.BASE, help="Analysis root folder")
    parser.add_argument("--skip-data", action="store_true", help="Skip parts A and B (no parquet loading)")
    args = parser.parse_args()

    base = args.base
    for module in (raw_mod, feat_mod):
        set_base(module, base)

    out_dir = base / "Output_checks" / "splits_and_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = [f"Base: {base}", ""]

    if not args.skip_data:
        print("[INFO] Loading raw windows (as in CNN/ResNet/HYDRA/MultiRocket)...", flush=True)
        raw_dt, _ = raw_mod.load_raw_data()
        raw_dt, *_ = raw_mod.clean_raw_data(raw_dt)

        print("[INFO] Loading feature windows (as in RF/LGBM)...", flush=True)
        feat_dt, _ = feat_mod.load_feature_data()
        feat_dt = feat_mod.clean_feature_data(feat_dt)

        raw_audit = audit_within_splits(raw_mod, raw_dt, "raw")
        feat_audit = audit_within_splits(feat_mod, feat_dt, "features")
        audit = pd.concat([raw_audit, feat_audit], ignore_index=True)
        counts = compare_window_counts(raw_dt, feat_dt)
        units = compare_test_units(raw_audit, feat_audit)

        audit.to_csv(out_dir / "01_within_split_audit.csv", index=False)
        counts.to_csv(out_dir / "02_raw_vs_feature_window_counts.csv", index=False)
        units.to_csv(out_dir / "03_within_test_units_raw_vs_features.csv", index=False)

        report += [
            "=== A: Within-dataset split (raw input) ===",
            raw_audit.drop(columns=["source"]).to_string(index=False),
            "",
            "=== A: Within-dataset split (feature input) ===",
            feat_audit.drop(columns=["source"]).to_string(index=False),
            "",
            "=== B: Window counts raw vs features ===",
            f"Total raw windows: {len(raw_dt)} | total feature windows: {len(feat_dt)}",
            f"Dataset-behavior cells with a difference: {int((counts['difference'] != 0).sum())} of {len(counts)}",
            counts[counts["difference"] != 0].to_string(index=False) if (counts["difference"] != 0).any() else "None.",
            "",
            "=== B: Within test individuals raw vs features ===",
            units[["dataset", "split_var_raw", "split_var_features", "same_split_var", "same_test_units"]].to_string(index=False),
            "",
        ]

    runs = collect_training_histories(base)
    summary = summarize_histories(runs)
    runs.to_csv(out_dir / "04_training_history_runs.csv", index=False)
    summary.to_csv(out_dir / "05_training_history_summary.csv", index=False)

    report += ["=== C: CNN/ResNet training histories ==="]
    if runs.empty:
        report.append("No training_history.csv files found under <base>/Models/{CNN,ResNet}/.")
    else:
        report.append(summary.to_string(index=False))
        within = runs[runs["mode"] == "Within"].sort_values(["model", "run"])
        report += ["", "Within runs:", within.to_string(index=False)]

    report_text = "\n".join(report) + "\n"
    (out_dir / "check_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"[OK] Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
