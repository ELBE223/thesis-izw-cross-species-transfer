# 05_Random_Forest
# Author : Lucas Beseler
# Date   : 2026-06-07

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# =========================================================
# Paths
# =========================================================
BASE = Path("/Volumes/Z Slim/11_05_2026_Data_Analysis")
FEATURE_ROOT = BASE / "Data_Processed" / "data_processed_features"
OUT_ROOT = BASE / "Models" / "RF"

GLOBAL_ROOT = OUT_ROOT / "Global_RF"
WITHIN_ROOT = OUT_ROOT / "Within_RF"
INTER_ROOT = OUT_ROOT / "Inter_RF"
CROSS_ROOT = OUT_ROOT / "Cross_RF"
PAIRWISE_ROOT = OUT_ROOT / "Pairwise_RF"

# =========================================================
# Settings
# =========================================================
SEED = 42
EXPECTED_DATASET_COUNT = 12
KEEP_BEHAVIORS = ["Foraging", "Locomotion", "Resting"]
TEST_FRAC = 0.25

MIN_N_GLOBAL_DATASET_BEHAVIOR = 30
MIN_N_WITHIN_DATASET_BEHAVIOR = 30
MIN_N_INTER_TRAIN_BEHAVIOR = 30
MIN_N_INTER_TEST_BEHAVIOR = 5
MIN_N_CROSS_TRAIN_BEHAVIOR = 30
MIN_N_CROSS_TEST_BEHAVIOR = 5
MIN_N_PAIRWISE_TRAIN_BEHAVIOR = 30
MIN_N_PAIRWISE_TEST_BEHAVIOR = 5

CAP_N_GLOBAL_TRAIN_DATASET_BEHAVIOR = 250
CAP_N_WITHIN_TRAIN_BEHAVIOR = 250
CAP_N_INTER_TRAIN_BEHAVIOR = 250
CAP_N_CROSS_TRAIN_DATASET_BEHAVIOR = 250
CAP_N_PAIRWISE_TRAIN_BEHAVIOR = 250

TOP_N_VARIMP = 15

FEATURE_COLS = [
    "vedba_mean", "vedba_sd", "vedba_max",
    "odba_mean", "odba_sd",
    "sd_x", "sd_y", "sd_z",
    "rms_x", "rms_y", "rms_z",
    "iqr_x", "iqr_y", "iqr_z",
    "skew_x", "skew_y", "skew_z",
    "kurt_x", "kurt_y", "kurt_z",
    "zero_cross_x", "zero_cross_y", "zero_cross_z",
    "spec_dom_freq_mag", "spec_centroid_mag", "spec_entropy_mag",
]

INTER_FAMILIES = {
    "Fox": ("Fox_dataset_1", "Fox_dataset_2"),
    "Horse": ("Horse_dataset_1", "Horse_dataset_2"),
    "Raccoon": ("Raccoon_dataset_1", "Raccoon_dataset_2"),
    "Dog": ("Dog_dataset_1", "Dog_dataset_2"),
}

ROOT_READ_LOG_NAME = "read_log.txt"
ROOT_RF_OVERVIEW_NAME = "RF_overview.txt"
ROOT_COUNTS_NAME = "counts_by_dataset_species_behavior.csv"
ROOT_LOOKUP_NAME = "analysis_dataset_lookup.csv"
ROOT_DUPLICATE_NAME = "duplicate_window_report.csv"
ROOT_UNMATCHED_NAME = "unmatched_dataset_keys.csv"

TRAIN_BALANCE_NAME = "train_balance.csv"
TEST_BALANCE_NAME = "test_balance.csv"
CONFUSION_MATRIX_NAME = "confusion_matrix.csv"
OVERALL_METRICS_NAME = "overall_metrics.csv"
BEHAVIOR_METRICS_NAME = "behavior_metrics.csv"
VARIABLE_IMPORTANCE_NAME = "variable_importance.csv"
SUMMARY_NAME = "summary.txt"

DATASET_METRICS_NAME = "dataset_metrics.csv"
METRICS_ALL_NAME = "metrics_all.csv"
BEHAVIOR_METRICS_ALL_NAME = "behavior_metrics_all.csv"
VARIABLE_IMPORTANCE_ALL_NAME = "variable_importance_all.csv"

DATASET_METRICS_PLOT_NAME = "dataset_metrics.png"
SUMMARY_METRICS_PLOT_NAME = "summary_metrics.png"
VARIABLE_IMPORTANCE_PLOT_NAME = "variable_importance_top15.png"

# =========================================================
# Logging
# =========================================================
def log(msg: str) -> None:
    print(msg, flush=True)

# =========================================================
# Helpers
# =========================================================
def safe_str(x) -> str:
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    return str(x).strip()

def make_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    log(f"[OK] Directory ready: {path}")

def valid_parquet_files(root: Path) -> list[Path]:
    """Return only the final flat per-dataset parquet files.

    The final 07 output should contain exactly 12 direct parquet files.
    Recursive loading is intentionally avoided so old nested part files are not mixed in.
    """
    if not root.exists():
        return []
    return sorted(
        p for p in root.glob("*.parquet")
        if p.is_file() and not p.name.startswith("._") and not p.name.startswith(".__")
    )

def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")

def sample_cap(df: pd.DataFrame, n_cap: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= n_cap:
        return df.copy()
    idx = rng.choice(df.index.to_numpy(), size=n_cap, replace=False)
    return df.loc[idx].copy()

def cap_by_groups(
    df: pd.DataFrame,
    group_cols: list[str],
    n_cap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for _, sub in df.groupby(group_cols, dropna=False, sort=False):
        parts.append(sample_cap(sub.copy(), n_cap, rng))
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)

def choose_split_var(df: pd.DataFrame) -> Optional[str]:
    # Prefer individual-level split keys to reduce leakage between train and test.
    for col in ["individual_key", "individual_id", "individual_name", "name", "subject_key", "source_file"]:
        if col in df.columns:
            vals = df[col].fillna("").astype(str).str.strip()
            vals = vals[vals != ""]
            if vals.nunique() >= 2:
                return col
    return None

def make_holdout_split(df: pd.DataFrame, dataset_col: str, test_frac: float, seed: int) -> tuple[Optional[pd.DataFrame], list[str]]:
    rng = np.random.default_rng(seed)
    out_parts: list[pd.DataFrame] = []
    skip_log: list[str] = []

    for dataset_name, sub in df.groupby(dataset_col, dropna=False):
        sub = sub.copy()
        split_var = choose_split_var(sub)
        if split_var is None:
            skip_log.append(f"Skipped {dataset_name}: no valid split unit (individual_key/individual_id/individual_name/name/subject_key/source_file)")
            continue

        split_keys = (
            sub[split_var]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        split_keys = split_keys[split_keys != ""].drop_duplicates().reset_index(drop=True)

        if len(split_keys) < 2:
            skip_log.append(f"Skipped {dataset_name}: fewer than 2 unique {split_var} values")
            continue

        n_test = max(1, int(math.floor(len(split_keys) * test_frac)))
        perm = rng.permutation(len(split_keys))
        test_keys = set(split_keys.iloc[perm[:n_test]].tolist())

        sub["split_key"] = sub[split_var].fillna("").astype(str).str.strip()
        sub["in_test"] = sub["split_key"].isin(test_keys)
        sub["split_var_used"] = split_var
        out_parts.append(sub)

    if not out_parts:
        return None, skip_log

    out = pd.concat(out_parts, ignore_index=True)
    return out, skip_log


def build_confusion_df(truth: pd.Series, pred: pd.Series) -> pd.DataFrame:
    truth = pd.Series(truth, dtype="object")
    pred = pd.Series(pred, dtype="object")
    cm = pd.crosstab(truth, pred, dropna=False)
    cm = cm.reindex(index=KEEP_BEHAVIORS, columns=KEEP_BEHAVIORS, fill_value=0)

    out_rows = []
    for t in KEEP_BEHAVIORS:
        row_total = int(cm.loc[t].sum())
        for p in KEEP_BEHAVIORS:
            n = int(cm.loc[t, p])
            row_prop = (n / row_total) if row_total > 0 else np.nan
            label = f"{n}" if not np.isfinite(row_prop) else f"{n}\n{100 * row_prop:.1f}%"
            out_rows.append(
                {
                    "truth": t,
                    "pred": p,
                    "N": n,
                    "row_total": row_total,
                    "row_prop": row_prop,
                    "label": label,
                }
            )
    return pd.DataFrame(out_rows)

def metric_from_confusion(cm: pd.DataFrame) -> dict[str, float]:
    """Compute overall metrics with zero_division=0.

    Macro metrics are always averaged over the three harmonized behavior classes.
    Undefined precision, recall, or F1 values are set to 0.0.
    """
    cm = cm.reindex(index=KEEP_BEHAVIORS, columns=KEEP_BEHAVIORS, fill_value=0).astype(float)

    total = float(cm.to_numpy().sum())
    row_sums = cm.sum(axis=1).astype(float)
    col_sums = cm.sum(axis=0).astype(float)
    diag = pd.Series(np.diag(cm.to_numpy()), index=KEEP_BEHAVIORS, dtype=float)

    accuracy = float(diag.sum() / total) if total > 0 else np.nan

    recall_values: list[float] = []
    precision_values: list[float] = []
    f1_values: list[float] = []

    for behavior in KEEP_BEHAVIORS:
        tp = float(diag.get(behavior, 0.0))
        support = float(row_sums.get(behavior, 0.0))
        predicted = float(col_sums.get(behavior, 0.0))

        recall = tp / support if support > 0 else 0.0
        precision = tp / predicted if predicted > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        recall_values.append(recall)
        precision_values.append(precision)
        f1_values.append(f1)

    return {
        "accuracy": accuracy,
        "macro_recall": float(np.mean(recall_values)),
        "macro_precision": float(np.mean(precision_values)),
        "macro_f1": float(np.mean(f1_values)),
    }

def get_metrics(truth: pd.Series, pred: pd.Series) -> pd.DataFrame:
    cm = pd.crosstab(pd.Series(truth), pd.Series(pred), dropna=False)
    cm = cm.reindex(index=KEEP_BEHAVIORS, columns=KEEP_BEHAVIORS, fill_value=0)
    return pd.DataFrame([metric_from_confusion(cm)])

def get_behavior_metrics(truth: pd.Series, pred: pd.Series) -> pd.DataFrame:
    """Compute class-wise metrics with zero_division=0."""
    cm = pd.crosstab(pd.Series(truth), pd.Series(pred), dropna=False)
    cm = cm.reindex(index=KEEP_BEHAVIORS, columns=KEEP_BEHAVIORS, fill_value=0).astype(float)

    row_sums = cm.sum(axis=1).astype(float)
    col_sums = cm.sum(axis=0).astype(float)
    diag = pd.Series(np.diag(cm.to_numpy()), index=KEEP_BEHAVIORS, dtype=float)

    rows = []
    for behavior in KEEP_BEHAVIORS:
        tp = float(diag.get(behavior, 0.0))
        support = float(row_sums.get(behavior, 0.0))
        predicted = float(col_sums.get(behavior, 0.0))

        recall = tp / support if support > 0 else 0.0
        precision = tp / predicted if predicted > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append(
            {
                "behavior": behavior,
                "support": int(support),
                "predicted": int(predicted),
                "tp": int(tp),
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows)

def save_text_summary(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[OK] Text written: {path}")

def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log(f"[WARN] Could not read overview input {path}: {exc}")
        return pd.DataFrame()

def fmt_num(value) -> str:
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)

def compact_metric_table(dt: pd.DataFrame, id_cols: list[str]) -> list[str]:
    if dt.empty:
        return ["No metric rows available."]

    metric_cols = ["accuracy", "macro_recall", "macro_precision", "macro_f1"]
    cols = [c for c in id_cols if c in dt.columns] + [c for c in metric_cols if c in dt.columns]
    if not cols:
        return ["No displayable metric columns available."]

    show_dt = dt[cols].copy()
    for col in metric_cols:
        if col in show_dt.columns:
            show_dt[col] = show_dt[col].map(fmt_num)

    return show_dt.to_string(index=False).splitlines()

def append_mode_overview(
    lines: list[str],
    mode_name: str,
    metrics_file: Path,
    id_cols: list[str],
    summary_file: Optional[Path] = None,
) -> None:
    lines.extend(["", "-" * 72, mode_name, "-" * 72])

    metrics_dt = read_csv_if_exists(metrics_file)
    if metrics_dt.empty:
        lines.append(f"No metrics file found or no completed models: {metrics_file}")
    else:
        lines.append(f"Completed model rows: {len(metrics_dt)}")
        for metric in ["accuracy", "macro_recall", "macro_precision", "macro_f1"]:
            if metric in metrics_dt.columns:
                values = pd.to_numeric(metrics_dt[metric], errors="coerce")
                lines.append(
                    f"{metric}: mean={fmt_num(values.mean())}, "
                    f"min={fmt_num(values.min())}, max={fmt_num(values.max())}"
                )

        if "macro_f1" in metrics_dt.columns:
            ranked = metrics_dt.copy()
            ranked["macro_f1_numeric"] = pd.to_numeric(ranked["macro_f1"], errors="coerce")
            ranked = ranked.dropna(subset=["macro_f1_numeric"])
            if not ranked.empty:
                best = ranked.sort_values("macro_f1_numeric", ascending=False).iloc[0]
                worst = ranked.sort_values("macro_f1_numeric", ascending=True).iloc[0]
                label_cols = [c for c in id_cols if c in ranked.columns]
                best_label = " | ".join(str(best[c]) for c in label_cols) if label_cols else "best row"
                worst_label = " | ".join(str(worst[c]) for c in label_cols) if label_cols else "worst row"
                lines.append(f"Best macro_f1: {best_label} = {fmt_num(best['macro_f1_numeric'])}")
                lines.append(f"Worst macro_f1: {worst_label} = {fmt_num(worst['macro_f1_numeric'])}")

        lines.extend(["", "Metrics table:"])
        lines.extend(compact_metric_table(metrics_dt, id_cols))

    if summary_file is not None and summary_file.exists():
        lines.extend(["", "Status / skipped information:"])
        summary_lines = summary_file.read_text(encoding="utf-8").splitlines()
        keep = False
        kept_any = False
        for line in summary_lines:
            lower = line.lower()
            if "completed" in lower or "skipped" in lower or "split skips" in lower:
                keep = True
            if keep:
                lines.append(line)
                kept_any = True
        if not kept_any:
            lines.append("No skipped/status lines found in summary file.")

def save_rf_overview(all_dt: pd.DataFrame) -> None:
    out_file = OUT_ROOT / ROOT_RF_OVERVIEW_NAME

    lines: list[str] = [
        "RF overview",
        "===========",
        "This file is overwritten on every full RF pipeline run.",
        f"Output root: {OUT_ROOT}",
        "",
        "Input after cleaning",
        "--------------------",
        f"Rows: {len(all_dt)}",
        f"Datasets: {all_dt['analysis_dataset'].nunique() if 'analysis_dataset' in all_dt.columns else 'NA'}",
        f"Behaviors kept: {', '.join(KEEP_BEHAVIORS)}",
        f"Features: {len(FEATURE_COLS)}",
    ]

    if "analysis_dataset" in all_dt.columns and "behavior" in all_dt.columns:
        counts = (
            all_dt.groupby(["analysis_dataset", "behavior"], dropna=False)
            .size()
            .reset_index(name="N")
            .sort_values(["analysis_dataset", "behavior"])
        )
        lines.extend(["", "Rows by dataset and behavior:"])
        lines.extend(counts.to_string(index=False).splitlines())

    append_mode_overview(
        lines,
        "Global_RF",
        GLOBAL_ROOT / "statistics" / OVERALL_METRICS_NAME,
        [],
        GLOBAL_ROOT / "statistics" / SUMMARY_NAME,
    )
    append_mode_overview(
        lines,
        "Global_RF metrics by dataset",
        GLOBAL_ROOT / "statistics" / DATASET_METRICS_NAME,
        ["analysis_dataset"],
        None,
    )
    append_mode_overview(
        lines,
        "Within_RF",
        WITHIN_ROOT / "statistics" / METRICS_ALL_NAME,
        ["analysis_dataset"],
        WITHIN_ROOT / "statistics" / SUMMARY_NAME,
    )
    append_mode_overview(
        lines,
        "Inter_RF",
        INTER_ROOT / "statistics" / METRICS_ALL_NAME,
        ["family_id", "pair_id", "train_dataset", "test_dataset"],
        INTER_ROOT / "statistics" / SUMMARY_NAME,
    )
    append_mode_overview(
        lines,
        "Cross_RF",
        CROSS_ROOT / "statistics" / METRICS_ALL_NAME,
        ["test_dataset"],
        CROSS_ROOT / "statistics" / SUMMARY_NAME,
    )
    append_mode_overview(
        lines,
        "Pairwise_RF",
        PAIRWISE_ROOT / "statistics" / METRICS_ALL_NAME,
        ["pair_id", "train_dataset", "test_dataset", "train_family", "test_family"],
        PAIRWISE_ROOT / "statistics" / SUMMARY_NAME,
    )

    save_text_summary(out_file, lines)

def save_cm_plot(cm_dt: pd.DataFrame, out_file: Path, title: str) -> None:
    truths = KEEP_BEHAVIORS
    preds = KEEP_BEHAVIORS
    mat = np.full((len(truths), len(preds)), np.nan)
    labels = np.empty((len(truths), len(preds)), dtype=object)

    for i, t in enumerate(truths):
        for j, p in enumerate(preds):
            row = cm_dt[(cm_dt["truth"] == t) & (cm_dt["pred"] == p)].iloc[0]
            mat[i, j] = row["row_prop"] if pd.notna(row["row_prop"]) else 0.0
            labels[i, j] = row["label"]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, interpolation="nearest")
    ax.set_xticks(np.arange(len(preds)))
    ax.set_yticks(np.arange(len(truths)))
    ax.set_xticklabels(preds)
    ax.set_yticklabels(truths)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(truths)):
        for j in range(len(preds)):
            ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row proportion")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"[OK] Plot saved: {out_file}")

def save_varimp_plot(varimp_dt: pd.DataFrame, out_file: Path, title: str, top_n: int = TOP_N_VARIMP) -> None:
    if varimp_dt.empty:
        return
    plot_dt = varimp_dt.head(min(top_n, len(varimp_dt))).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_dt["feature"], plot_dt["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"[OK] Plot saved: {out_file}")

def save_behavior_metric_plot(metric_dt: pd.DataFrame, out_file: Path, title: str) -> None:
    plot_dt = metric_dt.copy()
    metrics = ["recall", "precision", "f1"]
    x = np.arange(len(plot_dt))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, plot_dt[metric].fillna(0.0), width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_dt["behavior"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"[OK] Plot saved: {out_file}")

def save_summary_metric_plot(dt: pd.DataFrame, id_col: str, out_file: Path, title: str) -> None:
    if dt.empty:
        return
    plot_dt = dt[[id_col, "accuracy", "macro_f1"]].copy()
    plot_dt = plot_dt.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    y = np.arange(len(plot_dt))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y - width / 2, plot_dt["accuracy"], height=width, label="accuracy")
    ax.barh(y + width / 2, plot_dt["macro_f1"], height=width, label="macro_f1")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_dt[id_col])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"[OK] Plot saved: {out_file}")

def infer_analysis_dataset(top_folder: str, species: str, dataset_id: str, dataset_folder: str) -> str:
    # With Script 07, each parquet file already represents one final dataset.
    # Therefore the dataset/file name is the safest analysis key.
    for value in [top_folder, dataset_id, dataset_folder]:
        value = safe_str(value)
        if value:
            value = re.sub(r"\.parquet$", "", value)
            value = re.sub(r"_features$", "", value)
            value = re.sub(r"_mr$", "", value)
            return value
    return "UNMAPPED"

def normalize_behavior_label(value) -> str:
    text = safe_str(value).lower()
    mapping = {
        "foraging": "Foraging",
        "locomotion": "Locomotion",
        "resting": "Resting",
    }
    return mapping.get(text, safe_str(value))

def add_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in [
        "dataset_name", "dataset_id", "dataset_group", "dataset_folder", "top_folder",
        "species", "individual_id", "individual_name", "individual_key",
        "name", "subject_key", "source_file", "behavior", "behavior_en", "main_class",
    ]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Final 07 files use dataset_name. Keeping this as the analysis dataset.
    df["analysis_dataset"] = df["dataset_name"]
    missing_dataset = df["analysis_dataset"] == ""
    if missing_dataset.any():
        df.loc[missing_dataset, "analysis_dataset"] = [
            infer_analysis_dataset(t, s, d, f)
            for t, s, d, f in zip(
                df.loc[missing_dataset, "top_folder"],
                df.loc[missing_dataset, "species"],
                df.loc[missing_dataset, "dataset_id"],
                df.loc[missing_dataset, "dataset_folder"],
            )
        ]

    # Here, a single capitalized behavior column expected by the RF code!
    behavior_source = df["behavior"].copy()
    behavior_source = behavior_source.mask(behavior_source == "", df["behavior_en"])
    behavior_source = behavior_source.mask(behavior_source == "", df["main_class"])
    df["behavior"] = behavior_source.map(normalize_behavior_label)

    df["analysis_family"] = df["analysis_dataset"].str.replace(r"_dataset_[0-9]+$", "", regex=True)
    df["dataset_key"] = np.where(
        df["dataset_group"] != "",
        df["dataset_group"] + "__" + df["dataset_id"],
        np.where(df["dataset_id"] != "", df["dataset_id"], df["analysis_dataset"]),
    )
    return df

def read_feature_file(path: Path) -> pd.DataFrame:
    table = pq.read_table(path)
    df = table.to_pandas()

    dataset_from_file = re.sub(r"\.parquet$", "", path.name)
    dataset_from_file = re.sub(r"_features$", "", dataset_from_file)
    dataset_from_file = re.sub(r"_mr$", "", dataset_from_file)

    if "dataset_name" not in df.columns:
        df["dataset_name"] = dataset_from_file
    else:
        ds = df["dataset_name"].fillna("").astype(str).str.strip()
        df["dataset_name"] = ds.mask(ds == "", dataset_from_file)

    # These columns keep me older helper functions/reporting compatible.
    df["top_folder"] = df["dataset_name"].fillna("").astype(str).str.strip()
    df["dataset_folder"] = dataset_from_file
    df["part_file"] = path.name
    df["full_feature_path"] = str(path)
    return df

def load_feature_data() -> tuple[pd.DataFrame, list[str]]:
    files = valid_parquet_files(FEATURE_ROOT)
    if not files:
        raise FileNotFoundError(f"No feature parquet files found in {FEATURE_ROOT}")
    if len(files) != EXPECTED_DATASET_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_DATASET_COUNT} final feature parquet files in {FEATURE_ROOT}, "
            f"found {len(files)}"
        )

    frames: list[pd.DataFrame] = []
    read_log: list[str] = []

    for i, file_path in enumerate(files, start=1):
        log("\n========================================")
        log(f"[INFO] File {i} of {len(files)}")
        log(f"[INFO] Input: {file_path}")
        try:
            df = read_feature_file(file_path)
            frames.append(df)
            read_log.append(f"OK | {file_path} | rows={len(df)}")
            log(f"[OK] Rows: {len(df)}")
        except Exception as exc:
            read_log.append(f"ERROR | {file_path} | {exc}")
            log(f"[ERROR] Could not read file: {exc}")

    if not frames:
        raise RuntimeError("No feature files could be read.")

    all_dt = pd.concat(frames, ignore_index=True)
    return all_dt, read_log

def clean_feature_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {', '.join(missing_features)}")

    df = add_analysis_columns(df)
    df = df[df["behavior"].isin(KEEP_BEHAVIORS)].copy()
    df = df[df["analysis_dataset"] != ""].copy()


    df = df.dropna(subset=FEATURE_COLS).copy()
    df["behavior"] = df["behavior"].astype(str)

    return df.reset_index(drop=True)

def save_duplicate_report(df: pd.DataFrame, out_file: Path) -> None:
    key_cols = [c for c in ["analysis_dataset", "dataset_folder", "source_file", "window_id", "window_start"] if c in df.columns]
    if len(key_cols) < 3:
        pd.DataFrame({"info": ["Duplicate check skipped: not enough key columns"]}).to_csv(out_file, index=False)
        return

    dup = df.groupby(key_cols, dropna=False).size().reset_index(name="n_rows")
    dup = dup[dup["n_rows"] > 1].sort_values("n_rows", ascending=False)
    dup.to_csv(out_file, index=False)
    log(f"[OK] CSV saved: {out_file}")

def fit_rf_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    X_train = train_df[FEATURE_COLS].to_numpy()
    X_test = test_df[FEATURE_COLS].to_numpy()
    y_train = train_df["behavior"].astype(str)
    y_test = test_df["behavior"].astype(str)

    # Standard setting: using sklearn defaults.
    # Only random_state and n_jobs are fixed for reproducibility and speed.
    model = RandomForestClassifier(
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = pd.Series(model.predict(X_test), index=test_df.index, dtype="object")
    truth = pd.Series(y_test.to_numpy(), index=test_df.index, dtype="object")

    cm_dt = build_confusion_df(truth, pred)
    overall_metrics = get_metrics(truth, pred)
    behavior_metrics = get_behavior_metrics(truth, pred)
    varimp_dt = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    model_settings = (
        "sklearn defaults; "
        f"n_estimators={model.n_estimators}; "
        f"max_features={model.max_features}; "
        f"min_samples_leaf={model.min_samples_leaf}; "
        f"class_weight={model.class_weight}; "
        f"random_state={model.random_state}; "
        f"n_jobs={model.n_jobs}"
    )

    return {
        "model": model,
        "pred": pred,
        "truth": truth,
        "cm_dt": cm_dt,
        "overall_metrics": overall_metrics,
        "behavior_metrics": behavior_metrics,
        "varimp_dt": varimp_dt,
        "model_settings": model_settings,
    }

def prepare_mode_dirs(root: Path) -> tuple[Path, Path]:
    plots_dir = root / "plots"
    stats_dir = root / "statistics"
    make_clean_dir(root)
    make_clean_dir(plots_dir)
    make_clean_dir(stats_dir)
    return plots_dir, stats_dir

def write_basic_outputs(
    fit: dict[str, object],
    plot_dir: Path,
    stat_dir: Path,
    title_prefix: str,
    train_balance: pd.DataFrame,
    test_balance: pd.DataFrame,
    summary_lines: list[str],
) -> None:
    pd.DataFrame(train_balance).to_csv(stat_dir / TRAIN_BALANCE_NAME, index=False)
    pd.DataFrame(test_balance).to_csv(stat_dir / TEST_BALANCE_NAME, index=False)
    pd.DataFrame(fit["cm_dt"]).to_csv(stat_dir / CONFUSION_MATRIX_NAME, index=False)
    pd.DataFrame(fit["overall_metrics"]).to_csv(stat_dir / OVERALL_METRICS_NAME, index=False)
    pd.DataFrame(fit["behavior_metrics"]).to_csv(stat_dir / BEHAVIOR_METRICS_NAME, index=False)
    pd.DataFrame(fit["varimp_dt"]).to_csv(stat_dir / VARIABLE_IMPORTANCE_NAME, index=False)

    save_cm_plot(pd.DataFrame(fit["cm_dt"]), plot_dir / "confusion_matrix.png", f"{title_prefix} | Confusion matrix")
    save_varimp_plot(pd.DataFrame(fit["varimp_dt"]), plot_dir / VARIABLE_IMPORTANCE_PLOT_NAME, f"{title_prefix} | Top 15 variable importance")
    save_behavior_metric_plot(pd.DataFrame(fit["behavior_metrics"]), plot_dir / "behavior_metrics.png", f"{title_prefix} | Behavior metrics")
    save_text_summary(stat_dir / SUMMARY_NAME, summary_lines)

def run_global_rf(all_dt: pd.DataFrame) -> None:
    log("\n================ GLOBAL RF ================")
    plots_dir, stats_dir = prepare_mode_dirs(GLOBAL_ROOT)

    global_valid = (
        all_dt.groupby(["analysis_dataset", "behavior"], dropna=False)
        .size()
        .reset_index(name="N")
    )
    global_keep = (
        global_valid[global_valid["N"] >= MIN_N_GLOBAL_DATASET_BEHAVIOR]
        .groupby("analysis_dataset")["behavior"]
        .nunique()
        .reset_index(name="n_behaviors")
    )
    global_keep = global_keep[global_keep["n_behaviors"] == len(KEEP_BEHAVIORS)]["analysis_dataset"].tolist()

    global_dt = all_dt[all_dt["analysis_dataset"].isin(global_keep)].copy()
    split_dt, skip_log = make_holdout_split(global_dt, "analysis_dataset", TEST_FRAC, SEED)
    if split_dt is None:
        raise RuntimeError("Global split failed. No datasets with valid split units.")

    train_dt = split_dt[split_dt["in_test"] == False].copy()
    test_dt = split_dt[split_dt["in_test"] == True].copy()

    train_bal = cap_by_groups(
        train_dt,
        ["analysis_dataset", "behavior"],
        CAP_N_GLOBAL_TRAIN_DATASET_BEHAVIOR,
        SEED,
    )

    fit = fit_rf_model(train_bal, test_dt)

    global_metrics_by_dataset = []
    for ds, sub in test_dt.groupby("analysis_dataset", dropna=False):
        truth = pd.Series(fit["truth"]).loc[sub.index]
        pred = pd.Series(fit["pred"]).loc[sub.index]
        row = get_metrics(truth, pred)
        row["analysis_dataset"] = ds
        global_metrics_by_dataset.append(row)

    global_metrics_by_dataset = pd.concat(global_metrics_by_dataset, ignore_index=True)
    global_metrics_by_dataset = global_metrics_by_dataset.sort_values(["macro_f1", "accuracy"], ascending=[False, False])

    train_balance = (
        train_bal.groupby(["analysis_dataset", "behavior"], dropna=False)
        .size().reset_index(name="N")
        .sort_values(["analysis_dataset", "behavior"])
    )
    test_balance = (
        test_dt.groupby(["analysis_dataset", "behavior"], dropna=False)
        .size().reset_index(name="N")
        .sort_values(["analysis_dataset", "behavior"])
    )

    global_metrics_by_dataset.to_csv(stats_dir / DATASET_METRICS_NAME, index=False)
    save_summary_metric_plot(global_metrics_by_dataset, "analysis_dataset", plots_dir / DATASET_METRICS_PLOT_NAME, "Global RF | Metrics by dataset")

    summary_lines = [
        "Mode: Global_RF",
        f"Rows after filters: {len(all_dt)}",
        f"Rows in global model: {len(split_dt)}",
        f"Train rows after balancing: {len(train_bal)}",
        f"Test rows: {len(test_dt)}",
        f"Features: {len(FEATURE_COLS)}",
        f"model_settings: {fit['model_settings']}",
        f"Global datasets used: {', '.join(sorted(global_dt['analysis_dataset'].unique()))}",
        "Split skips: none" if not skip_log else "Split skips:\n  - " + "\n  - ".join(skip_log),
        "",
        "Overall metrics:",
        f"  accuracy = {float(pd.DataFrame(fit['overall_metrics'])['accuracy'].iloc[0]):.4f}",
        f"  macro_recall = {float(pd.DataFrame(fit['overall_metrics'])['macro_recall'].iloc[0]):.4f}",
        f"  macro_precision = {float(pd.DataFrame(fit['overall_metrics'])['macro_precision'].iloc[0]):.4f}",
        f"  macro_f1 = {float(pd.DataFrame(fit['overall_metrics'])['macro_f1'].iloc[0]):.4f}",
    ]

    write_basic_outputs(
        fit=fit,
        plot_dir=plots_dir,
        stat_dir=stats_dir,
        title_prefix="Global RF",
        train_balance=train_balance,
        test_balance=test_balance,
        summary_lines=summary_lines,
    )

def run_within_rf(all_dt: pd.DataFrame) -> None:
    log("\n================ WITHIN RF ================")
    plots_dir, stats_dir = prepare_mode_dirs(WITHIN_ROOT)

    within_ready = (
        all_dt.groupby(["analysis_dataset", "behavior"], dropna=False)
        .size().reset_index(name="N")
    )
    within_ready = (
        within_ready[within_ready["N"] >= MIN_N_WITHIN_DATASET_BEHAVIOR]
        .groupby("analysis_dataset")["behavior"]
        .nunique()
        .reset_index(name="n_behaviors")
    )
    within_ready = within_ready[within_ready["n_behaviors"] == len(KEEP_BEHAVIORS)]["analysis_dataset"].tolist()

    metrics_all = []
    behavior_all = []
    varimp_all = []
    skip_log: list[str] = []

    for ds in sorted(within_ready):
        log(f"\n[INFO] Within dataset: {ds}")
        ds_plot = plots_dir / sanitize_name(ds)
        ds_stat = stats_dir / sanitize_name(ds)
        make_clean_dir(ds_plot)
        make_clean_dir(ds_stat)

        ds_dt = all_dt[all_dt["analysis_dataset"] == ds].copy()
        split_dt, split_skip = make_holdout_split(ds_dt, "analysis_dataset", TEST_FRAC, SEED)

        if split_dt is None:
            skip_log.append(f"Skipped {ds}: no valid split")
            continue

        train_dt = split_dt[split_dt["in_test"] == False].copy()
        test_dt = split_dt[split_dt["in_test"] == True].copy()

        train_counts = train_dt.groupby("behavior").size().to_dict()
        test_counts = test_dt.groupby("behavior").size().to_dict()

        if not all(train_counts.get(b, 0) >= MIN_N_WITHIN_DATASET_BEHAVIOR for b in KEEP_BEHAVIORS):
            skip_log.append(f"Skipped {ds}: train set missing behavior(s)")
            continue

        if sum(test_counts.get(b, 0) >= 1 for b in KEEP_BEHAVIORS) < 2:
            skip_log.append(f"Skipped {ds}: test set has fewer than 2 behaviors")
            continue

        train_bal = cap_by_groups(
            train_dt,
            ["behavior"],
            CAP_N_WITHIN_TRAIN_BEHAVIOR,
            SEED,
        )

        fit = fit_rf_model(train_bal, test_dt)

        train_balance = train_bal.groupby("behavior").size().reset_index(name="N").sort_values("behavior")
        test_balance = test_dt.groupby("behavior").size().reset_index(name="N").sort_values("behavior")

        metrics_dt = pd.DataFrame(fit["overall_metrics"]).copy()
        metrics_dt["analysis_dataset"] = ds

        behavior_dt = pd.DataFrame(fit["behavior_metrics"]).copy()
        behavior_dt["analysis_dataset"] = ds

        varimp_dt = pd.DataFrame(fit["varimp_dt"]).copy()
        varimp_dt["analysis_dataset"] = ds

        summary_lines = [
            "Mode: Within_RF",
            f"Dataset: {ds}",
            f"Rows: {len(split_dt)}",
            f"Train rows after balancing: {len(train_bal)}",
            f"Test rows: {len(test_dt)}",
                f"Features: {len(FEATURE_COLS)}",
            f"model_settings: {fit['model_settings']}",
            f"Split var used: {', '.join(sorted(split_dt['split_var_used'].dropna().astype(str).unique()))}",
            "Split skips: none" if not split_skip else "Split skips:\n  - " + "\n  - ".join(split_skip),
            "",
            "Overall metrics:",
            f"  accuracy = {float(metrics_dt['accuracy'].iloc[0]):.4f}",
            f"  macro_recall = {float(metrics_dt['macro_recall'].iloc[0]):.4f}",
            f"  macro_precision = {float(metrics_dt['macro_precision'].iloc[0]):.4f}",
            f"  macro_f1 = {float(metrics_dt['macro_f1'].iloc[0]):.4f}",
        ]

        write_basic_outputs(
            fit=fit,
            plot_dir=ds_plot,
            stat_dir=ds_stat,
            title_prefix=f"Within RF | {ds}",
            train_balance=train_balance,
            test_balance=test_balance,
            summary_lines=summary_lines,
        )

        metrics_all.append(metrics_dt)
        behavior_all.append(behavior_dt)
        varimp_all.append(varimp_dt)

    if metrics_all:
        metrics_all_df = pd.concat(metrics_all, ignore_index=True)
        behavior_all_df = pd.concat(behavior_all, ignore_index=True)
        varimp_all_df = pd.concat(varimp_all, ignore_index=True)

        metrics_all_df.to_csv(stats_dir / METRICS_ALL_NAME, index=False)
        behavior_all_df.to_csv(stats_dir / BEHAVIOR_METRICS_ALL_NAME, index=False)
        varimp_all_df.to_csv(stats_dir / VARIABLE_IMPORTANCE_ALL_NAME, index=False)
        save_summary_metric_plot(metrics_all_df, "analysis_dataset", plots_dir / SUMMARY_METRICS_PLOT_NAME, "Within RF | Accuracy and macro F1")

        summary_lines = [
            "Mode: Within_RF summary",
            "Datasets completed: " + ", ".join(sorted(metrics_all_df["analysis_dataset"].tolist())),
            "Skipped datasets: none" if not skip_log else "Skipped datasets:\n  - " + "\n  - ".join(skip_log),
        ]
        save_text_summary(stats_dir / SUMMARY_NAME, summary_lines)

def run_inter_rf(all_dt: pd.DataFrame) -> None:
    log("\n================ INTER RF ================")
    plots_dir, stats_dir = prepare_mode_dirs(INTER_ROOT)

    metrics_all = []
    behavior_all = []
    varimp_all = []
    skip_log: list[str] = []

    for family, (ds1, ds2) in INTER_FAMILIES.items():
        for train_ds, test_ds in [(ds1, ds2), (ds2, ds1)]:
            pair_id = f"{train_ds}__to__{test_ds}"
            log(f"\n[INFO] Inter pair: {pair_id}")

            pair_plot = plots_dir / sanitize_name(pair_id)
            pair_stat = stats_dir / sanitize_name(pair_id)
            make_clean_dir(pair_plot)
            make_clean_dir(pair_stat)

            train_dt = all_dt[all_dt["analysis_dataset"] == train_ds].copy()
            test_dt = all_dt[all_dt["analysis_dataset"] == test_ds].copy()

            if train_dt.empty or test_dt.empty:
                skip_log.append(f"Skipped {pair_id}: empty train or test dataset")
                continue

            train_counts = train_dt.groupby("behavior").size().to_dict()
            test_counts = test_dt.groupby("behavior").size().to_dict()

            if not all(train_counts.get(b, 0) >= MIN_N_INTER_TRAIN_BEHAVIOR for b in KEEP_BEHAVIORS):
                skip_log.append(f"Skipped {pair_id}: train set missing behavior(s)")
                continue

            if sum(test_counts.get(b, 0) >= MIN_N_INTER_TEST_BEHAVIOR for b in KEEP_BEHAVIORS) < 2:
                skip_log.append(f"Skipped {pair_id}: test set has fewer than 2 usable behaviors")
                continue

            train_bal = cap_by_groups(
                train_dt,
                ["behavior"],
                CAP_N_INTER_TRAIN_BEHAVIOR,
                SEED,
            )

            fit = fit_rf_model(train_bal, test_dt)

            train_balance = train_bal.groupby("behavior").size().reset_index(name="N").sort_values("behavior")
            test_balance = test_dt.groupby("behavior").size().reset_index(name="N").sort_values("behavior")

            metrics_dt = pd.DataFrame(fit["overall_metrics"]).copy()
            metrics_dt["family_id"] = family
            metrics_dt["pair_id"] = pair_id
            metrics_dt["train_dataset"] = train_ds
            metrics_dt["test_dataset"] = test_ds

            behavior_dt = pd.DataFrame(fit["behavior_metrics"]).copy()
            behavior_dt["family_id"] = family
            behavior_dt["pair_id"] = pair_id
            behavior_dt["train_dataset"] = train_ds
            behavior_dt["test_dataset"] = test_ds

            varimp_dt = pd.DataFrame(fit["varimp_dt"]).copy()
            varimp_dt["family_id"] = family
            varimp_dt["pair_id"] = pair_id
            varimp_dt["train_dataset"] = train_ds
            varimp_dt["test_dataset"] = test_ds

            summary_lines = [
                "Mode: Inter_RF",
                f"Family: {family}",
                f"Train dataset: {train_ds}",
                f"Test dataset: {test_ds}",
                f"Train rows after balancing: {len(train_bal)}",
                f"Test rows: {len(test_dt)}",
                        f"Features: {len(FEATURE_COLS)}",
                f"model_settings: {fit['model_settings']}",
                "",
                "Overall metrics:",
                f"  accuracy = {float(metrics_dt['accuracy'].iloc[0]):.4f}",
                f"  macro_recall = {float(metrics_dt['macro_recall'].iloc[0]):.4f}",
                f"  macro_precision = {float(metrics_dt['macro_precision'].iloc[0]):.4f}",
                f"  macro_f1 = {float(metrics_dt['macro_f1'].iloc[0]):.4f}",
            ]

            write_basic_outputs(
                fit=fit,
                plot_dir=pair_plot,
                stat_dir=pair_stat,
                title_prefix=f"Inter RF | {pair_id}",
                train_balance=train_balance,
                test_balance=test_balance,
                summary_lines=summary_lines,
            )

            metrics_all.append(metrics_dt)
            behavior_all.append(behavior_dt)
            varimp_all.append(varimp_dt)

    if metrics_all:
        metrics_all_df = pd.concat(metrics_all, ignore_index=True)
        behavior_all_df = pd.concat(behavior_all, ignore_index=True)
        varimp_all_df = pd.concat(varimp_all, ignore_index=True)

        metrics_all_df.to_csv(stats_dir / METRICS_ALL_NAME, index=False)
        behavior_all_df.to_csv(stats_dir / BEHAVIOR_METRICS_ALL_NAME, index=False)
        varimp_all_df.to_csv(stats_dir / VARIABLE_IMPORTANCE_ALL_NAME, index=False)
        save_summary_metric_plot(metrics_all_df, "pair_id", plots_dir / SUMMARY_METRICS_PLOT_NAME, "Inter RF | Accuracy and macro F1")

        summary_lines = [
            "Mode: Inter_RF summary",
            "Pairs completed: " + ", ".join(sorted(metrics_all_df["pair_id"].tolist())),
            "Skipped pairs: none" if not skip_log else "Skipped pairs:\n  - " + "\n  - ".join(skip_log),
        ]
        save_text_summary(stats_dir / SUMMARY_NAME, summary_lines)

def run_cross_rf(all_dt: pd.DataFrame) -> None:
    log("\n================ CROSS RF ================")
    plots_dir, stats_dir = prepare_mode_dirs(CROSS_ROOT)

    metrics_all = []
    behavior_all = []
    varimp_all = []
    skip_log: list[str] = []

    all_datasets = sorted(all_dt["analysis_dataset"].dropna().astype(str).unique().tolist())

    for test_ds in all_datasets:
        log(f"\n[INFO] Cross test dataset: {test_ds}")

        test_dt = all_dt[all_dt["analysis_dataset"] == test_ds].copy()
        train_dt = all_dt[all_dt["analysis_dataset"] != test_ds].copy()

        out_plot = plots_dir / sanitize_name(test_ds)
        out_stat = stats_dir / sanitize_name(test_ds)
        make_clean_dir(out_plot)
        make_clean_dir(out_stat)

        train_counts = train_dt.groupby("behavior").size().to_dict()
        test_counts = test_dt.groupby("behavior").size().to_dict()

        if not all(train_counts.get(b, 0) >= MIN_N_CROSS_TRAIN_BEHAVIOR for b in KEEP_BEHAVIORS):
            skip_log.append(f"Skipped {test_ds}: train pool missing behavior(s)")
            continue

        if sum(test_counts.get(b, 0) >= MIN_N_CROSS_TEST_BEHAVIOR for b in KEEP_BEHAVIORS) < 2:
            skip_log.append(f"Skipped {test_ds}: test dataset has fewer than 2 usable behaviors")
            continue

        train_bal = cap_by_groups(
            train_dt,
            ["analysis_dataset", "behavior"],
            CAP_N_CROSS_TRAIN_DATASET_BEHAVIOR,
            SEED,
        )

        fit = fit_rf_model(train_bal, test_dt)

        train_balance = (
            train_bal.groupby(["analysis_dataset", "behavior"], dropna=False)
            .size().reset_index(name="N")
            .sort_values(["analysis_dataset", "behavior"])
        )
        test_balance = test_dt.groupby("behavior").size().reset_index(name="N").sort_values("behavior")

        metrics_dt = pd.DataFrame(fit["overall_metrics"]).copy()
        metrics_dt["test_dataset"] = test_ds

        behavior_dt = pd.DataFrame(fit["behavior_metrics"]).copy()
        behavior_dt["test_dataset"] = test_ds

        varimp_dt = pd.DataFrame(fit["varimp_dt"]).copy()
        varimp_dt["test_dataset"] = test_ds

        summary_lines = [
            "Mode: Cross_RF",
            f"Test dataset: {test_ds}",
            f"Train datasets: {', '.join(sorted(train_bal['analysis_dataset'].unique()))}",
            f"Train rows after balancing: {len(train_bal)}",
            f"Test rows: {len(test_dt)}",
                f"Features: {len(FEATURE_COLS)}",
            f"model_settings: {fit['model_settings']}",
            "",
            "Overall metrics:",
            f"  accuracy = {float(metrics_dt['accuracy'].iloc[0]):.4f}",
            f"  macro_recall = {float(metrics_dt['macro_recall'].iloc[0]):.4f}",
            f"  macro_precision = {float(metrics_dt['macro_precision'].iloc[0]):.4f}",
            f"  macro_f1 = {float(metrics_dt['macro_f1'].iloc[0]):.4f}",
        ]

        write_basic_outputs(
            fit=fit,
            plot_dir=out_plot,
            stat_dir=out_stat,
            title_prefix=f"Cross RF | test={test_ds}",
            train_balance=train_balance,
            test_balance=test_balance,
            summary_lines=summary_lines,
        )

        metrics_all.append(metrics_dt)
        behavior_all.append(behavior_dt)
        varimp_all.append(varimp_dt)

    if metrics_all:
        metrics_all_df = pd.concat(metrics_all, ignore_index=True)
        behavior_all_df = pd.concat(behavior_all, ignore_index=True)
        varimp_all_df = pd.concat(varimp_all, ignore_index=True)

        metrics_all_df.to_csv(stats_dir / METRICS_ALL_NAME, index=False)
        behavior_all_df.to_csv(stats_dir / BEHAVIOR_METRICS_ALL_NAME, index=False)
        varimp_all_df.to_csv(stats_dir / VARIABLE_IMPORTANCE_ALL_NAME, index=False)
        save_summary_metric_plot(metrics_all_df, "test_dataset", plots_dir / SUMMARY_METRICS_PLOT_NAME, "Cross RF | Accuracy and macro F1")

        summary_lines = [
            "Mode: Cross_RF summary",
            "Datasets completed: " + ", ".join(sorted(metrics_all_df["test_dataset"].tolist())),
            "Skipped datasets: none" if not skip_log else "Skipped datasets:\n  - " + "\n  - ".join(skip_log),
        ]
        save_text_summary(stats_dir / SUMMARY_NAME, summary_lines)

def run_pairwise_rf(all_dt: pd.DataFrame) -> None:
    log("\n================ PAIRWISE RF ================")
    plots_dir, stats_dir = prepare_mode_dirs(PAIRWISE_ROOT)

    metrics_all = []
    behavior_all = []
    varimp_all = []
    skip_log: list[str] = []

    all_datasets = sorted(all_dt["analysis_dataset"].dropna().astype(str).unique().tolist())
    family_lookup = (
        all_dt[["analysis_dataset", "analysis_family"]]
        .drop_duplicates()
        .dropna(subset=["analysis_dataset"])
    )
    family_lookup = dict(zip(family_lookup["analysis_dataset"], family_lookup["analysis_family"]))

    for train_ds in all_datasets:
        for test_ds in all_datasets:
            if train_ds == test_ds:
                continue

            pair_id = f"{train_ds}__to__{test_ds}"
            log(f"\n[INFO] Pairwise pair: {pair_id}")

            pair_plot = plots_dir / sanitize_name(pair_id)
            pair_stat = stats_dir / sanitize_name(pair_id)
            make_clean_dir(pair_plot)
            make_clean_dir(pair_stat)

            train_dt = all_dt[all_dt["analysis_dataset"] == train_ds].copy()
            test_dt = all_dt[all_dt["analysis_dataset"] == test_ds].copy()

            if train_dt.empty or test_dt.empty:
                skip_log.append(f"Skipped {pair_id}: empty train or test dataset")
                continue

            train_counts = train_dt.groupby("behavior").size().to_dict()
            test_counts = test_dt.groupby("behavior").size().to_dict()

            if not all(train_counts.get(b, 0) >= MIN_N_PAIRWISE_TRAIN_BEHAVIOR for b in KEEP_BEHAVIORS):
                skip_log.append(f"Skipped {pair_id}: train set missing behavior(s)")
                continue

            if sum(test_counts.get(b, 0) >= MIN_N_PAIRWISE_TEST_BEHAVIOR for b in KEEP_BEHAVIORS) < 2:
                skip_log.append(f"Skipped {pair_id}: test set has fewer than 2 usable behaviors")
                continue

            train_bal = cap_by_groups(
                train_dt,
                ["behavior"],
                CAP_N_PAIRWISE_TRAIN_BEHAVIOR,
                SEED,
            )

            fit = fit_rf_model(train_bal, test_dt)

            train_balance = train_bal.groupby("behavior").size().reset_index(name="N").sort_values("behavior")
            test_balance = test_dt.groupby("behavior").size().reset_index(name="N").sort_values("behavior")

            metrics_dt = pd.DataFrame(fit["overall_metrics"]).copy()
            metrics_dt["pair_id"] = pair_id
            metrics_dt["train_dataset"] = train_ds
            metrics_dt["test_dataset"] = test_ds
            metrics_dt["train_family"] = family_lookup.get(train_ds, "")
            metrics_dt["test_family"] = family_lookup.get(test_ds, "")

            behavior_dt = pd.DataFrame(fit["behavior_metrics"]).copy()
            behavior_dt["pair_id"] = pair_id
            behavior_dt["train_dataset"] = train_ds
            behavior_dt["test_dataset"] = test_ds
            behavior_dt["train_family"] = family_lookup.get(train_ds, "")
            behavior_dt["test_family"] = family_lookup.get(test_ds, "")

            varimp_dt = pd.DataFrame(fit["varimp_dt"]).copy()
            varimp_dt["pair_id"] = pair_id
            varimp_dt["train_dataset"] = train_ds
            varimp_dt["test_dataset"] = test_ds
            varimp_dt["train_family"] = family_lookup.get(train_ds, "")
            varimp_dt["test_family"] = family_lookup.get(test_ds, "")

            summary_lines = [
                "Mode: Pairwise_RF",
                f"Pair: {pair_id}",
                f"Train dataset: {train_ds}",
                f"Test dataset: {test_ds}",
                f"Train family: {family_lookup.get(train_ds, '')}",
                f"Test family: {family_lookup.get(test_ds, '')}",
                f"Train rows after balancing: {len(train_bal)}",
                f"Test rows: {len(test_dt)}",
                        f"Features: {len(FEATURE_COLS)}",
                f"model_settings: {fit['model_settings']}",
                "",
                "Overall metrics:",
                f"  accuracy = {float(metrics_dt['accuracy'].iloc[0]):.4f}",
                f"  macro_recall = {float(metrics_dt['macro_recall'].iloc[0]):.4f}",
                f"  macro_precision = {float(metrics_dt['macro_precision'].iloc[0]):.4f}",
                f"  macro_f1 = {float(metrics_dt['macro_f1'].iloc[0]):.4f}",
            ]

            write_basic_outputs(
                fit=fit,
                plot_dir=pair_plot,
                stat_dir=pair_stat,
                title_prefix=f"Pairwise RF | {pair_id}",
                train_balance=train_balance,
                test_balance=test_balance,
                summary_lines=summary_lines,
            )

            metrics_all.append(metrics_dt)
            behavior_all.append(behavior_dt)
            varimp_all.append(varimp_dt)

    if metrics_all:
        metrics_all_df = pd.concat(metrics_all, ignore_index=True)
        behavior_all_df = pd.concat(behavior_all, ignore_index=True)
        varimp_all_df = pd.concat(varimp_all, ignore_index=True)

        metrics_all_df.to_csv(stats_dir / METRICS_ALL_NAME, index=False)
        behavior_all_df.to_csv(stats_dir / BEHAVIOR_METRICS_ALL_NAME, index=False)
        varimp_all_df.to_csv(stats_dir / VARIABLE_IMPORTANCE_ALL_NAME, index=False)
        save_summary_metric_plot(metrics_all_df, "pair_id", plots_dir / SUMMARY_METRICS_PLOT_NAME, "Pairwise RF | Accuracy and macro F1")

        summary_lines = [
            "Mode: Pairwise_RF summary",
            "Pairs completed: " + ", ".join(sorted(metrics_all_df["pair_id"].tolist())),
            "Skipped pairs: none" if not skip_log else "Skipped pairs:\n  - " + "\n  - ".join(skip_log),
        ]
        save_text_summary(stats_dir / SUMMARY_NAME, summary_lines)

def main() -> int:
    np.random.seed(SEED)

    for root in [OUT_ROOT, GLOBAL_ROOT, WITHIN_ROOT, INTER_ROOT, CROSS_ROOT, PAIRWISE_ROOT]:
        root.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] Feature root: {FEATURE_ROOT}")
    log(f"[INFO] Output root: {OUT_ROOT}")
    log("[INFO] Loading feature files...")
    all_dt, read_log = load_feature_data()

    read_log_path = OUT_ROOT / ROOT_READ_LOG_NAME
    read_log_path.write_text("\n".join(read_log), encoding="utf-8")
    log(f"[OK] Text written: {read_log_path}")

    log("[INFO] Cleaning feature table...")
    all_dt = clean_feature_data(all_dt)

    counts_dt = (
        all_dt.groupby(["analysis_dataset", "dataset_folder", "species", "behavior"], dropna=False)
        .size().reset_index(name="N")
        .sort_values(["analysis_dataset", "dataset_folder", "species", "behavior"])
    )
    counts_dt.to_csv(OUT_ROOT / ROOT_COUNTS_NAME, index=False)
    log(f"[OK] CSV saved: {OUT_ROOT / ROOT_COUNTS_NAME}")

    lookup_dt = (
        all_dt.groupby(
            ["analysis_dataset", "dataset_folder", "analysis_family", "species", "dataset_group", "dataset_id", "dataset_key"],
            dropna=False,
        )
        .agg(
            rows=("behavior", "size"),
            n_names=("name", pd.Series.nunique),
            n_subject_keys=("subject_key", pd.Series.nunique),
            n_source_files=("source_file", pd.Series.nunique),
        )
        .reset_index()
        .sort_values(["analysis_dataset", "dataset_folder"])
    )
    lookup_dt.to_csv(OUT_ROOT / ROOT_LOOKUP_NAME, index=False)
    log(f"[OK] CSV saved: {OUT_ROOT / ROOT_LOOKUP_NAME}")

    save_duplicate_report(all_dt, OUT_ROOT / ROOT_DUPLICATE_NAME)

    unmatched = all_dt[all_dt["analysis_dataset"] == "UNMAPPED"].copy()
    if not unmatched.empty:
        unmatched[["species", "dataset_group", "dataset_id", "dataset_folder"]].drop_duplicates().to_csv(
            OUT_ROOT / ROOT_UNMATCHED_NAME, index=False
        )
        raise RuntimeError("Some rows could not be mapped to analysis_dataset. Check unmatched_dataset_keys.csv")

    run_global_rf(all_dt)
    run_within_rf(all_dt)
    run_inter_rf(all_dt)
    run_cross_rf(all_dt)
    run_pairwise_rf(all_dt)
    save_rf_overview(all_dt)

    log("\n========================================")
    log("[DONE] RF pipeline finished.")
    log(f"[DONE] Output root: {OUT_ROOT}")
    log("========================================")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())