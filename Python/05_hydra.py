# 05_hydra.py
# Author : Lucas Beseler
# Date   : 2026-06-07

from __future__ import annotations

import gc
import math
import random
import re
import traceback
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from aeon.classification.convolution_based import HydraClassifier
    HYDRA_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    HydraClassifier = None  # type: ignore[assignment]
    HYDRA_IMPORT_ERROR = exc


# =========================================================
# Paths
# =========================================================
BASE = Path("/Volumes/Z Slim/11_05_2026_Data_Analysis")
RAW_ROOT = BASE / "Data_Processed" / "data_processed_final"
OUT_ROOT = BASE / "Models" / "HYDRA"

GLOBAL_ROOT = OUT_ROOT / "Global_HYDRA"
WITHIN_ROOT = OUT_ROOT / "Within_HYDRA"
INTER_ROOT = OUT_ROOT / "Inter_HYDRA"
CROSS_ROOT = OUT_ROOT / "Cross_HYDRA"
PAIRWISE_ROOT = OUT_ROOT / "Pairwise_HYDRA"
OVERVIEW_NAME = "HYDRA_overview.txt"


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

HYDRA_N_JOBS = 4
HYDRA_SETTINGS_NOTE = "aeon HydraClassifier defaults; only random_state and n_jobs are fixed"
PREDICT_CHUNK_SIZE = 2000

# Run switches
RUN_GLOBAL = True
RUN_WITHIN = True
RUN_INTER = True
RUN_CROSS = True
RUN_PAIRWISE = True

INTER_FAMILIES = {
    "Fox": ("Fox_dataset_1", "Fox_dataset_2"),
    "Horse": ("Horse_dataset_1", "Horse_dataset_2"),
    "Raccoon": ("Raccoon_dataset_1", "Raccoon_dataset_2"),
    "Dog":("Dog_dataset_1", "Dog_dataset_2"),
}


# =========================================================
# Logging
# =========================================================
def log(msg: str) -> None:
    print(msg, flush=True)


def log_exception(prefix: str, exc: Exception) -> None:
    log(f"[ERROR] {prefix}: {exc}")
    traceback.print_exc()


# =========================================================
# Helpers
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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

    """
    if not root.exists():
        return []
    return sorted(
        p for p in root.glob("*.parquet")
        if p.is_file() and not p.name.startswith("._") and not p.name.startswith(".__")
    )


def get_required_read_columns(path: Path) -> list[str]:
    schema_names = set(pq.ParquetFile(path).schema.names)

    keep_cols = [
        "dataset_name",
        "dataset_id",
        "dataset_group",
        "dataset_folder",
        "top_folder",
        "species",
        "individual_id",
        "individual_name",
        "individual_key",
        "name",
        "subject_key",
        "source_file",
        "behavior",
        "behavior_en",
        "main_class",
        "window_id",
        "window_start",
    ]
    keep_cols = [c for c in keep_cols if c in schema_names]

    axis_cols = sorted(
        c for c in schema_names
        if re.fullmatch(r"[xyz]_\d{3}", str(c))
    )

    return keep_cols + axis_cols

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
    for _, sub in df.groupby(group_cols, dropna=False, sort=False, observed=False):
        parts.append(sample_cap(sub.copy(), n_cap, rng))
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def choose_split_var(df: pd.DataFrame) -> Optional[str]:
    # Prefer individual-level split keys to reduce train/test leakage.
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

    for dataset_name, sub in df.groupby(dataset_col, dropna=False, observed=False):
        sub = sub.copy()
        split_var = choose_split_var(sub)
        if split_var is None:
            skip_log.append(f"Skipped {dataset_name}: no valid split unit (individual_key/individual_id/individual_name/name/subject_key/source_file)")
            continue

        split_keys = sub[split_var].fillna("").astype(str).str.strip()
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

    rows = []
    for t in KEEP_BEHAVIORS:
        row_total = int(cm.loc[t].sum())
        for p in KEEP_BEHAVIORS:
            n = int(cm.loc[t, p])
            row_prop = (n / row_total) if row_total > 0 else np.nan
            label = f"{n}" if not np.isfinite(row_prop) else f"{n}\n{100 * row_prop:.1f}%"
            rows.append(
                {
                    "truth": t,
                    "pred": p,
                    "N": n,
                    "row_total": row_total,
                    "row_prop": row_prop,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def metric_from_confusion(cm: pd.DataFrame) -> dict[str, float]:
    cm = cm.reindex(index=KEEP_BEHAVIORS, columns=KEEP_BEHAVIORS, fill_value=0)

    row_sums = cm.sum(axis=1).astype(float)
    col_sums = cm.sum(axis=0).astype(float)
    diag = pd.Series(np.diag(cm.to_numpy()), index=KEEP_BEHAVIORS, dtype=float)

    accuracy = float(diag.sum() / cm.to_numpy().sum()) if cm.to_numpy().sum() > 0 else np.nan
    recall = (diag / row_sums).replace([np.inf, -np.inf], np.nan)
    precision = (diag / col_sums).replace([np.inf, -np.inf], np.nan)
    f1 = (2 * precision * recall / (precision + recall)).replace([np.inf, -np.inf], np.nan)

    return {
        "accuracy": accuracy,
        "macro_recall": float(recall.mean(skipna=True)),
        "macro_precision": float(precision.mean(skipna=True)),
        "macro_f1": float(f1.mean(skipna=True)),
    }


def get_metrics(truth: pd.Series, pred: pd.Series) -> pd.DataFrame:
    cm = pd.crosstab(pd.Series(truth), pd.Series(pred), dropna=False)
    cm = cm.reindex(index=KEEP_BEHAVIORS, columns=KEEP_BEHAVIORS, fill_value=0)
    return pd.DataFrame([metric_from_confusion(cm)])


def get_behavior_metrics(truth: pd.Series, pred: pd.Series) -> pd.DataFrame:
    cm = pd.crosstab(pd.Series(truth), pd.Series(pred), dropna=False)
    cm = cm.reindex(index=KEEP_BEHAVIORS, columns=KEEP_BEHAVIORS, fill_value=0)

    row_sums = cm.sum(axis=1).astype(float)
    col_sums = cm.sum(axis=0).astype(float)
    diag = pd.Series(np.diag(cm.to_numpy()), index=KEEP_BEHAVIORS, dtype=float)

    recall = (diag / row_sums).replace([np.inf, -np.inf], np.nan)
    precision = (diag / col_sums).replace([np.inf, -np.inf], np.nan)
    f1 = (2 * precision * recall / (precision + recall)).replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame(
        {
            "behavior": KEEP_BEHAVIORS,
            "support": [int(row_sums.get(b, 0)) for b in KEEP_BEHAVIORS],
            "predicted": [int(col_sums.get(b, 0)) for b in KEEP_BEHAVIORS],
            "tp": [int(diag.get(b, 0)) for b in KEEP_BEHAVIORS],
            "recall": [recall.get(b, np.nan) for b in KEEP_BEHAVIORS],
            "precision": [precision.get(b, np.nan) for b in KEEP_BEHAVIORS],
            "f1": [f1.get(b, np.nan) for b in KEEP_BEHAVIORS],
        }
    )


def save_text_summary(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[OK] Text written: {path}")


def save_cm_plot(cm_dt: pd.DataFrame, out_file: Path, title: str) -> None:
    pivot = cm_dt.pivot(index="truth", columns="pred", values="row_prop").reindex(index=KEEP_BEHAVIORS, columns=KEEP_BEHAVIORS)
    labels = cm_dt.pivot(index="truth", columns="pred", values="label").reindex(index=KEEP_BEHAVIORS, columns=KEEP_BEHAVIORS)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(KEEP_BEHAVIORS)))
    ax.set_xticklabels(KEEP_BEHAVIORS)
    ax.set_yticks(range(len(KEEP_BEHAVIORS)))
    ax.set_yticklabels(KEEP_BEHAVIORS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(KEEP_BEHAVIORS)):
        for j in range(len(KEEP_BEHAVIORS)):
            txt = labels.iloc[i, j]
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Row proportion")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"[OK] Plot saved: {out_file}")


def save_behavior_metric_plot(metric_dt: pd.DataFrame, out_file: Path, title: str) -> None:
    plot_dt = metric_dt.copy()
    metrics = ["recall", "precision", "f1"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(plot_dt))
    width = 0.25

    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, plot_dt[metric].fillna(0).to_numpy(), width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_dt["behavior"].tolist())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"[OK] Plot saved: {out_file}")


def save_summary_metric_plot(df: pd.DataFrame, id_col: str, out_file: Path, title: str) -> None:
    plot_dt = df.copy()
    metrics = ["accuracy", "macro_recall", "macro_precision", "macro_f1"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(plot_dt))
    width = 0.20

    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, plot_dt[metric].fillna(0).to_numpy(), width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_dt[id_col].tolist(), rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"[OK] Plot saved: {out_file}")


def infer_analysis_dataset(top_folder: str, species: str, dataset_id: str, dataset_folder: str) -> str:
    # With the new final files, each parquet file already represents one dataset.
    # Therefore the dataset/file name is the safest analysis key.
    for value in [top_folder, dataset_id, dataset_folder]:
        value = safe_str(value)
        if value:
            value = re.sub(r"\.parquet$", "", value)
            value = re.sub(r"_windows$", "", value)
            value = re.sub(r"_final$", "", value)
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

    # New final files use dataset_name. Keep this as the analysis dataset.
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

    # Use one capitalized behavior column expected by the HYDRA code.
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


def validate_no_unmapped(df: pd.DataFrame, out_file: Optional[Path] = None) -> None:
    bad = df[df["analysis_dataset"] == "UNMAPPED"].copy()
    if bad.empty:
        return

    cols = [c for c in ["species", "dataset_group", "dataset_id", "dataset_folder", "top_folder", "source_file", "full_raw_path"] if c in bad.columns]
    preview = bad[cols].drop_duplicates().head(10)

    if out_file is not None:
        bad[cols].drop_duplicates().to_csv(out_file, index=False)
        log(f"[OK] CSV saved: {out_file}")

    log("[ERROR] Found rows with analysis_dataset = UNMAPPED.")
    log(f"[ERROR] UNMAPPED rows: {len(bad)}")
    if not preview.empty:
        log("[ERROR] First UNMAPPED combinations:")
        for row in preview.to_dict(orient="records"):
            parts = [f"{k}={safe_str(v)}" for k, v in row.items()]
            log("[ERROR]   - " + " | ".join(parts))

    raise ValueError(
        "UNMAPPED analysis_dataset detected. Check unmatched_dataset_keys.csv and review the dataset mapping."
    )

def detect_axis_cols(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    x_cols = sorted([c for c in df.columns if re.fullmatch(r"x_\d{3}", str(c))])
    y_cols = sorted([c for c in df.columns if re.fullmatch(r"y_\d{3}", str(c))])
    z_cols = sorted([c for c in df.columns if re.fullmatch(r"z_\d{3}", str(c))])

    if not x_cols or not y_cols or not z_cols:
        raise ValueError("Missing x_/y_/z_ columns in raw data.")
    if not (len(x_cols) == len(y_cols) == len(z_cols)):
        raise ValueError("x_/y_/z_ columns have different lengths.")
    return x_cols, y_cols, z_cols


def read_raw_file(path: Path) -> pd.DataFrame:
    read_cols = get_required_read_columns(path)
    table = pq.read_table(path, columns=read_cols)
    df = table.to_pandas()

    dataset_from_file = re.sub(r"\.parquet$", "", path.name)
    dataset_from_file = re.sub(r"_windows$", "", dataset_from_file)
    dataset_from_file = re.sub(r"_final$", "", dataset_from_file)
    dataset_from_file = re.sub(r"_mr$", "", dataset_from_file)

    if "dataset_name" not in df.columns:
        df["dataset_name"] = dataset_from_file
    else:
        ds = df["dataset_name"].fillna("").astype(str).str.strip()
        df["dataset_name"] = ds.mask(ds == "", dataset_from_file)

    # These columns keep older helper functions/reporting compatible.
    df["top_folder"] = df["dataset_name"].fillna("").astype(str).str.strip()
    df["dataset_folder"] = dataset_from_file
    df["part_file"] = path.name
    df["full_raw_path"] = str(path)
    return df


def load_raw_data() -> tuple[pd.DataFrame, list[str]]:
    files = valid_parquet_files(RAW_ROOT)
    if not files:
        raise FileNotFoundError(f"No raw parquet files found in {RAW_ROOT}")
    if len(files) != EXPECTED_DATASET_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_DATASET_COUNT} final raw parquet files in {RAW_ROOT}, "
            f"found {len(files)}"
        )

    frames: list[pd.DataFrame] = []
    read_log: list[str] = []

    for i, file_path in enumerate(files, start=1):
        log("\n========================================")
        log(f"[INFO] File {i} of {len(files)}")
        log(f"[INFO] Input: {file_path}")
        try:
            df = read_raw_file(file_path)
            frames.append(df)
            read_log.append(f"OK | {file_path} | rows={len(df)}")
            log(f"[OK] Rows: {len(df)}")
        except Exception as exc:
            read_log.append(f"ERROR | {file_path} | {exc}")
            log_exception(f"Could not read file {file_path}", exc)

    if not frames:
        raise RuntimeError("No raw files could be read successfully. See read_log.txt for details.")

    all_dt = pd.concat(frames, ignore_index=True, copy=False)
    del frames
    gc.collect()
    log(f"[INFO] Combined raw rows: {len(all_dt)}")
    return all_dt, read_log


def clean_raw_data(df: pd.DataFrame, unmatched_out_file: Optional[Path] = None) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    n_input = len(df)
    df = add_analysis_columns(df)

    df = df[df["behavior"].isin(KEEP_BEHAVIORS)].copy()
    log(f"[INFO] Rows after behavior filter: {len(df)} / {n_input}")

    df = df[df["analysis_dataset"] != ""].copy()
    log(f"[INFO] Rows after dataset-name filter: {len(df)}")

    validate_no_unmapped(df, unmatched_out_file)


    x_cols, y_cols, z_cols = detect_axis_cols(df)
    before_dropna = len(df)
    df = df.dropna(subset=x_cols + y_cols + z_cols).copy()
    log(f"[INFO] Rows after raw axis NA filter: {len(df)} (removed {before_dropna - len(df)})")

    if df.empty:
        raise RuntimeError("No usable rows left after raw data cleaning.")

    axis_cols = x_cols + y_cols + z_cols
    df.loc[:, axis_cols] = df.loc[:, axis_cols].astype(np.float32, copy=False)
    df["behavior"] = df["behavior"].astype(str)

    for col in [
        "behavior", "species", "analysis_dataset", "analysis_family", "dataset_name",
        "dataset_folder", "dataset_group", "dataset_id", "dataset_key", "top_folder",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    gc.collect()
    return df.reset_index(drop=True), x_cols, y_cols, z_cols


def get_metadata_view(df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "analysis_dataset", "analysis_family", "dataset_name", "species", "behavior",
        "dataset_folder", "dataset_group", "dataset_id", "dataset_key",
        "top_folder", "individual_id", "individual_name", "individual_key",
        "name", "subject_key", "source_file",
        "window_id", "window_start", "part_file", "full_raw_path",
    ]
    keep_cols = [col for col in meta_cols if col in df.columns]
    meta_dt = df.loc[:, keep_cols].copy()

    for col in [
        "analysis_dataset", "analysis_family", "dataset_name", "species", "behavior",
        "dataset_folder", "dataset_group", "dataset_id", "dataset_key",
        "top_folder", "individual_id", "individual_name", "individual_key",
        "name", "subject_key", "source_file", "part_file",
    ]:
        if col in meta_dt.columns:
            meta_dt[col] = meta_dt[col].astype(str)

    return meta_dt

def save_duplicate_report(df: pd.DataFrame, out_file: Path) -> None:
    key_cols = [c for c in ["analysis_dataset", "source_file", "window_id", "window_start", "part_file"] if c in df.columns]
    if not key_cols:
        pd.DataFrame({"note": ["No duplicate key columns found."]}).to_csv(out_file, index=False)
        log(f"[OK] CSV saved: {out_file}")
        return

    dup = (
        df.groupby(key_cols, dropna=False, observed=True)
        .size()
        .reset_index(name="N")
        .query("N > 1")
        .sort_values("N", ascending=False)
    )
    dup.to_csv(out_file, index=False)
    log(f"[OK] CSV saved: {out_file}")


# =========================================================
# Data prep
# =========================================================
def make_tensor_from_df(df: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str]) -> np.ndarray:
    n = len(df)
    t = len(x_cols)

    arr = np.empty((n, 3, t), dtype=np.float32)
    arr[:, 0, :] = df.loc[:, x_cols].to_numpy(dtype=np.float32, copy=False)
    arr[:, 1, :] = df.loc[:, y_cols].to_numpy(dtype=np.float32, copy=False)
    arr[:, 2, :] = df.loc[:, z_cols].to_numpy(dtype=np.float32, copy=False)

    return arr


def predict_in_chunks(
    model,
    test_df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    z_cols: list[str],
    chunk_size: int = PREDICT_CHUNK_SIZE,
) -> np.ndarray:
    pred_parts: list[np.ndarray] = []

    for start in range(0, len(test_df), chunk_size):
        stop = min(start + chunk_size, len(test_df))
        chunk_df = test_df.iloc[start:stop]
        X_chunk = make_tensor_from_df(chunk_df, x_cols, y_cols, z_cols)
        pred_parts.append(np.asarray(model.predict(X_chunk), dtype=object))
        del X_chunk, chunk_df
        gc.collect()

    if not pred_parts:
        return np.array([], dtype=object)

    return np.concatenate(pred_parts, axis=0)


# =========================================================
# Model
# =========================================================
def fit_hydra_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    z_cols: list[str],
) -> dict[str, object]:
    if HydraClassifier is None:
        raise ImportError(
            "aeon is required for HydraClassifier. Install aeon first."
        ) from HYDRA_IMPORT_ERROR

    if train_df.empty:
        raise ValueError("fit_hydra_model received an empty training table.")
    if test_df.empty:
        raise ValueError("fit_hydra_model received an empty test table.")

    log(f"[INFO] Building train tensor: {len(train_df)} rows")
    X_train = make_tensor_from_df(train_df, x_cols, y_cols, z_cols)
    y_train = train_df["behavior"].astype(str)
    y_test = test_df["behavior"].astype(str)

    model = HydraClassifier(
        n_jobs=HYDRA_N_JOBS,
        random_state=SEED,
    )
    log(f"[INFO] Starting HYDRA fit: train={len(train_df)}, test={len(test_df)}")
    model.fit(X_train, y_train.to_numpy())
    log("[OK] HYDRA fit finished")

    del X_train
    gc.collect()

    log(f"[INFO] Starting prediction on {len(test_df)} rows")
    pred = pd.Series(
        predict_in_chunks(model, test_df, x_cols, y_cols, z_cols),
        index=test_df.index,
        dtype="object",
    )
    truth = pd.Series(y_test.to_numpy(), index=test_df.index, dtype="object")
    log("[OK] Prediction finished")

    cm_dt = build_confusion_df(truth, pred)
    overall_metrics = get_metrics(truth, pred)
    behavior_metrics = get_behavior_metrics(truth, pred)

    del model
    gc.collect()

    return {
        "pred": pred,
        "truth": truth,
        "cm_dt": cm_dt,
        "overall_metrics": overall_metrics,
        "behavior_metrics": behavior_metrics,
    }


# =========================================================
# Outputs
# =========================================================
def prepare_mode_dirs(root: Path) -> tuple[Path, Path]:
    make_clean_dir(root)
    plots_dir = root / "plots"
    stats_dir = root / "statistics"
    make_clean_dir(plots_dir)
    make_clean_dir(stats_dir)
    return plots_dir, stats_dir


def get_enabled_modes() -> list[tuple[str, Path, callable]]:
    modes: list[tuple[str, Path, callable]] = []

    if RUN_GLOBAL:
        modes.append(("GLOBAL", GLOBAL_ROOT, run_global_hydra))
    if RUN_WITHIN:
        modes.append(("WITHIN", WITHIN_ROOT, run_within_hydra))
    if RUN_INTER:
        modes.append(("INTER", INTER_ROOT, run_inter_hydra))
    if RUN_CROSS:
        modes.append(("CROSS", CROSS_ROOT, run_cross_hydra))
    if RUN_PAIRWISE:
        modes.append(("PAIRWISE", PAIRWISE_ROOT, run_pairwise_hydra))

    return modes


def write_basic_outputs(
    cm_dt: pd.DataFrame,
    metrics_dt: pd.DataFrame,
    behavior_dt: pd.DataFrame,
    plot_root: Path,
    stat_root: Path,
    title_prefix: str,
) -> None:
    cm_dt.to_csv(stat_root / "confusion_matrix.csv", index=False)
    metrics_dt.to_csv(stat_root / "overall_metrics.csv", index=False)
    behavior_dt.to_csv(stat_root / "behavior_metrics.csv", index=False)

    save_cm_plot(cm_dt, plot_root / "confusion_matrix.png", f"{title_prefix} | Confusion matrix")
    save_behavior_metric_plot(behavior_dt, plot_root / "behavior_metrics.png", f"{title_prefix} | Behavior metrics")


# =========================================================
# Modes
# =========================================================
def run_global_hydra(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str]) -> None:
    log("\n================ GLOBAL HYDRA ================")
    plots_dir, stats_dir = prepare_mode_dirs(GLOBAL_ROOT)

    global_valid = (
        all_dt.groupby(["analysis_dataset", "behavior"], dropna=False, observed=False)
        .size()
        .reset_index(name="N")
    )
    global_keep = (
        global_valid[global_valid["N"] >= MIN_N_GLOBAL_DATASET_BEHAVIOR]
        .groupby("analysis_dataset", observed=False)["behavior"]
        .nunique()
        .reset_index(name="n_behaviors")
    )
    global_keep = global_keep[global_keep["n_behaviors"] == len(KEEP_BEHAVIORS)]["analysis_dataset"].tolist()

    global_dt = all_dt[all_dt["analysis_dataset"].isin(global_keep)]
    split_dt, skip_log = make_holdout_split(global_dt, "analysis_dataset", TEST_FRAC, SEED)
    if split_dt is None:
        raise RuntimeError("Global split failed. No datasets with valid split units.")

    train_dt = split_dt[split_dt["in_test"] == False]
    test_dt = split_dt[split_dt["in_test"] == True]

    train_bal = cap_by_groups(
        train_dt,
        ["analysis_dataset", "behavior"],
        CAP_N_GLOBAL_TRAIN_DATASET_BEHAVIOR,
        SEED,
    )

    fit = fit_hydra_model(train_bal, test_dt, x_cols, y_cols, z_cols)

    metrics_by_dataset = []
    for ds, sub in test_dt.groupby("analysis_dataset", dropna=False, observed=False):
        truth = pd.Series(fit["truth"]).loc[sub.index]
        pred = pd.Series(fit["pred"]).loc[sub.index]
        row = get_metrics(truth, pred)
        row["analysis_dataset"] = ds
        metrics_by_dataset.append(row)

    metrics_by_dataset = pd.concat(metrics_by_dataset, ignore_index=True)
    metrics_by_dataset = metrics_by_dataset.sort_values(["macro_f1", "accuracy"], ascending=[False, False])
    metrics_by_dataset.to_csv(stats_dir / "metrics_by_analysis_dataset.csv", index=False)

    train_balance = (
        train_bal.groupby(["analysis_dataset", "behavior"], dropna=False, observed=False)
        .size().reset_index(name="N")
        .sort_values(["analysis_dataset", "behavior"])
    )
    test_balance = (
        test_dt.groupby(["analysis_dataset", "behavior"], dropna=False, observed=False)
        .size().reset_index(name="N")
        .sort_values(["analysis_dataset", "behavior"])
    )
    train_balance.to_csv(stats_dir / "train_balance.csv", index=False)
    test_balance.to_csv(stats_dir / "test_balance.csv", index=False)

    write_basic_outputs(
        fit["cm_dt"],
        fit["overall_metrics"],
        fit["behavior_metrics"],
        plots_dir,
        stats_dir,
        "Global HYDRA",
    )
    save_summary_metric_plot(metrics_by_dataset, "analysis_dataset", plots_dir / "dataset_metrics.png", "Global HYDRA | Metrics by dataset")

    lines = [
        "Mode: Global_HYDRA",
        f"Rows after filters: {len(all_dt)}",
        f"Rows in global model: {len(split_dt)}",
        f"Train rows after balancing: {len(train_bal)}",
        f"Test rows: {len(test_dt)}",
        f"Hydra settings: {HYDRA_SETTINGS_NOTE}",
        f"Hydra n_jobs: {HYDRA_N_JOBS}",
        f"Global datasets used: {', '.join(sorted(global_dt['analysis_dataset'].unique()))}",
        "Split skips: none" if not skip_log else "Split skips:\n  - " + "\n  - ".join(skip_log),
        "",
        "Overall metrics:",
    ]
    lines.extend(pd.DataFrame(fit["overall_metrics"]).to_string(index=False).splitlines())
    save_text_summary(stats_dir / "summary.txt", lines)

    del global_dt, split_dt, train_dt, test_dt, train_bal, fit, metrics_by_dataset, train_balance, test_balance
    gc.collect()


def run_within_hydra(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str]) -> None:
    log("\n================ WITHIN HYDRA ================")
    plots_dir, stats_dir = prepare_mode_dirs(WITHIN_ROOT)

    within_ready = (
        all_dt.groupby(["analysis_dataset", "behavior"], dropna=False, observed=False)
        .size().reset_index(name="N")
    )
    within_ready = (
        within_ready[within_ready["N"] >= MIN_N_WITHIN_DATASET_BEHAVIOR]
        .groupby("analysis_dataset", observed=False)["behavior"]
        .nunique()
        .reset_index(name="n_behaviors")
    )
    within_ready = within_ready[within_ready["n_behaviors"] == len(KEEP_BEHAVIORS)]["analysis_dataset"].tolist()

    metrics_all = []
    skip_log: list[str] = []

    for ds in sorted(within_ready):
        log(f"\n[INFO] Within dataset: {ds}")
        ds_plot = plots_dir / sanitize_name(ds)
        ds_stat = stats_dir / sanitize_name(ds)
        make_clean_dir(ds_plot)
        make_clean_dir(ds_stat)

        ds_dt = all_dt[all_dt["analysis_dataset"] == ds]
        split_dt, split_skip = make_holdout_split(ds_dt, "analysis_dataset", TEST_FRAC, SEED)
        if split_dt is None:
            skip_log.append(f"Skipped {ds}: no valid split")
            continue

        train_dt = split_dt[split_dt["in_test"] == False]
        test_dt = split_dt[split_dt["in_test"] == True]

        train_counts = train_dt.groupby("behavior", observed=False).size().to_dict()
        test_counts = test_dt.groupby("behavior", observed=False).size().to_dict()

        if not all(train_counts.get(b, 0) >= MIN_N_WITHIN_DATASET_BEHAVIOR for b in KEEP_BEHAVIORS):
            skip_log.append(f"Skipped {ds}: train set missing behavior(s)")
            continue

        if sum(test_counts.get(b, 0) >= 1 for b in KEEP_BEHAVIORS) < 2:
            skip_log.append(f"Skipped {ds}: test set has fewer than 2 behaviors")
            continue

        train_bal = cap_by_groups(train_dt, ["behavior"], CAP_N_WITHIN_TRAIN_BEHAVIOR, SEED)

        try:
            fit = fit_hydra_model(train_bal, test_dt, x_cols, y_cols, z_cols)
        except Exception as exc:
            msg = f"Skipped {ds}: {exc}"
            skip_log.append(msg)
            log_exception(msg, exc)
            continue

        metrics_dt = pd.DataFrame(fit["overall_metrics"]).copy()
        metrics_dt["analysis_dataset"] = ds
        metrics_all.append(metrics_dt)

        train_balance = train_bal.groupby("behavior", observed=False).size().reset_index(name="N").sort_values("behavior")
        test_balance = test_dt.groupby("behavior", observed=False).size().reset_index(name="N").sort_values("behavior")
        train_balance.to_csv(ds_stat / "train_balance.csv", index=False)
        test_balance.to_csv(ds_stat / "test_balance.csv", index=False)

        write_basic_outputs(
            fit["cm_dt"],
            fit["overall_metrics"],
            fit["behavior_metrics"],
            ds_plot,
            ds_stat,
            f"Within HYDRA | {ds}",
        )

        lines = [
            "Mode: Within_HYDRA",
            f"Dataset: {ds}",
            f"Rows: {len(split_dt)}",
            f"Train rows after balancing: {len(train_bal)}",
            f"Test rows: {len(test_dt)}",
                f"Hydra settings: {HYDRA_SETTINGS_NOTE}",
                f"Hydra n_jobs: {HYDRA_N_JOBS}",
            f"Split var used: {', '.join(sorted(split_dt['split_var_used'].dropna().astype(str).unique()))}",
            "Split skips: none" if not split_skip else "Split skips:\n  - " + "\n  - ".join(split_skip),
            "",
            "Overall metrics:",
        ]
        lines.extend(metrics_dt[["accuracy", "macro_recall", "macro_precision", "macro_f1"]].to_string(index=False).splitlines())
        save_text_summary(ds_stat / "summary.txt", lines)

        del ds_dt, split_dt, train_dt, test_dt, train_bal, fit, metrics_dt, train_balance, test_balance
        gc.collect()

    if metrics_all:
        summary_dt = pd.concat(metrics_all, ignore_index=True).sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        summary_dt.to_csv(stats_dir / "metrics_all.csv", index=False)
        save_summary_metric_plot(summary_dt, "analysis_dataset", plots_dir / "summary_metrics.png", "Within HYDRA | Summary metrics")

        lines = ["Mode: Within_HYDRA", "", "Summary metrics:"]
        lines.extend(summary_dt.to_string(index=False).splitlines())
        if skip_log:
            lines.append("")
            lines.append("Skip log:")
            lines.extend(skip_log)
        save_text_summary(stats_dir / "summary.txt", lines)
        gc.collect()


def run_inter_hydra(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str]) -> None:
    log("\n================ INTER HYDRA ================")
    plots_dir, stats_dir = prepare_mode_dirs(INTER_ROOT)

    metrics_all = []
    skip_log: list[str] = []

    for family, (ds1, ds2) in INTER_FAMILIES.items():
        for train_ds, test_ds in [(ds1, ds2), (ds2, ds1)]:
            pair_id = f"{train_ds}__to__{test_ds}"
            log(f"\n[INFO] Inter pair: {pair_id}")

            pair_plot = plots_dir / sanitize_name(pair_id)
            pair_stat = stats_dir / sanitize_name(pair_id)
            make_clean_dir(pair_plot)
            make_clean_dir(pair_stat)

            train_dt = all_dt[all_dt["analysis_dataset"] == train_ds]
            test_dt = all_dt[all_dt["analysis_dataset"] == test_ds]

            if train_dt.empty or test_dt.empty:
                skip_log.append(f"Skipped {pair_id}: empty train or test dataset")
                continue

            train_counts = train_dt.groupby("behavior", observed=False).size().to_dict()
            test_counts = test_dt.groupby("behavior", observed=False).size().to_dict()

            if not all(train_counts.get(b, 0) >= MIN_N_INTER_TRAIN_BEHAVIOR for b in KEEP_BEHAVIORS):
                skip_log.append(f"Skipped {pair_id}: train set missing behavior(s)")
                continue

            if sum(test_counts.get(b, 0) >= MIN_N_INTER_TEST_BEHAVIOR for b in KEEP_BEHAVIORS) < 2:
                skip_log.append(f"Skipped {pair_id}: test set has fewer than 2 usable behaviors")
                continue

            train_bal = cap_by_groups(train_dt, ["behavior"], CAP_N_INTER_TRAIN_BEHAVIOR, SEED)

            try:
                fit = fit_hydra_model(train_bal, test_dt, x_cols, y_cols, z_cols)
            except Exception as exc:
                msg = f"Skipped {pair_id}: {exc}"
                skip_log.append(msg)
                log_exception(msg, exc)
                continue

            metrics_dt = pd.DataFrame(fit["overall_metrics"]).copy()
            metrics_dt["family_id"] = family
            metrics_dt["pair_id"] = pair_id
            metrics_dt["train_dataset"] = train_ds
            metrics_dt["test_dataset"] = test_ds
            metrics_all.append(metrics_dt)

            train_balance = train_bal.groupby("behavior", observed=False).size().reset_index(name="N").sort_values("behavior")
            test_balance = test_dt.groupby("behavior", observed=False).size().reset_index(name="N").sort_values("behavior")
            train_balance.to_csv(pair_stat / "train_balance.csv", index=False)
            test_balance.to_csv(pair_stat / "test_balance.csv", index=False)

            write_basic_outputs(
                fit["cm_dt"],
                fit["overall_metrics"],
                fit["behavior_metrics"],
                pair_plot,
                pair_stat,
                f"Inter HYDRA | {pair_id}",
            )

            lines = [
                "Mode: Inter_HYDRA",
                f"Family: {family}",
                f"Train dataset: {train_ds}",
                f"Test dataset: {test_ds}",
                f"Train rows after balancing: {len(train_bal)}",
                f"Test rows: {len(test_dt)}",
                f"Hydra settings: {HYDRA_SETTINGS_NOTE}",
                f"Hydra n_jobs: {HYDRA_N_JOBS}",
                "",
                "Overall metrics:",
            ]
            lines.extend(pd.DataFrame(fit["overall_metrics"]).to_string(index=False).splitlines())
            save_text_summary(pair_stat / "summary.txt", lines)

            del train_dt, test_dt, train_bal, fit, metrics_dt, train_balance, test_balance
            gc.collect()

    if metrics_all:
        summary_dt = pd.concat(metrics_all, ignore_index=True).sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        summary_dt.to_csv(stats_dir / "metrics_all.csv", index=False)
        save_summary_metric_plot(summary_dt, "pair_id", plots_dir / "summary_metrics.png", "Inter HYDRA | Summary metrics")

        lines = ["Mode: Inter_HYDRA", "", "Summary metrics:"]
        lines.extend(summary_dt.to_string(index=False).splitlines())
        if skip_log:
            lines.append("")
            lines.append("Skip log:")
            lines.extend(skip_log)
        save_text_summary(stats_dir / "summary.txt", lines)
        gc.collect()


def run_cross_hydra(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str]) -> None:
    log("\n================ CROSS HYDRA ================")
    plots_dir, stats_dir = prepare_mode_dirs(CROSS_ROOT)

    metrics_all = []
    skip_log: list[str] = []
    all_datasets = sorted(all_dt["analysis_dataset"].dropna().astype(str).unique().tolist())

    for test_ds in all_datasets:
        log(f"\n[INFO] Cross test dataset: {test_ds}")

        test_dt = all_dt[all_dt["analysis_dataset"] == test_ds]
        train_dt = all_dt[all_dt["analysis_dataset"] != test_ds]

        out_plot = plots_dir / sanitize_name(test_ds)
        out_stat = stats_dir / sanitize_name(test_ds)
        make_clean_dir(out_plot)
        make_clean_dir(out_stat)

        train_counts = train_dt.groupby("behavior", observed=False).size().to_dict()
        test_counts = test_dt.groupby("behavior", observed=False).size().to_dict()

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

        try:
            fit = fit_hydra_model(train_bal, test_dt, x_cols, y_cols, z_cols)
        except Exception as exc:
            msg = f"Skipped {test_ds}: {exc}"
            skip_log.append(msg)
            log_exception(msg, exc)
            continue

        metrics_dt = pd.DataFrame(fit["overall_metrics"]).copy()
        metrics_dt["test_dataset"] = test_ds
        metrics_all.append(metrics_dt)

        train_balance = (
            train_bal.groupby(["analysis_dataset", "behavior"], dropna=False, observed=False)
            .size().reset_index(name="N")
            .sort_values(["analysis_dataset", "behavior"])
        )
        test_balance = test_dt.groupby("behavior", observed=False).size().reset_index(name="N").sort_values("behavior")
        train_balance.to_csv(out_stat / "train_balance.csv", index=False)
        test_balance.to_csv(out_stat / "test_balance.csv", index=False)

        write_basic_outputs(
            fit["cm_dt"],
            fit["overall_metrics"],
            fit["behavior_metrics"],
            out_plot,
            out_stat,
            f"Cross HYDRA | {test_ds}",
        )

        lines = [
            "Mode: Cross_HYDRA",
            f"Test dataset: {test_ds}",
            f"Train datasets: {', '.join(sorted(train_bal['analysis_dataset'].unique()))}",
            f"Train rows after balancing: {len(train_bal)}",
            f"Test rows: {len(test_dt)}",
                f"Hydra settings: {HYDRA_SETTINGS_NOTE}",
                f"Hydra n_jobs: {HYDRA_N_JOBS}",
            "",
            "Overall metrics:",
        ]
        lines.extend(pd.DataFrame(fit["overall_metrics"]).to_string(index=False).splitlines())
        save_text_summary(out_stat / "summary.txt", lines)

        del train_dt, test_dt, train_bal, fit, metrics_dt, train_balance, test_balance
        gc.collect()

    if metrics_all:
        summary_dt = pd.concat(metrics_all, ignore_index=True).sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        summary_dt.to_csv(stats_dir / "metrics_all.csv", index=False)
        save_summary_metric_plot(summary_dt, "test_dataset", plots_dir / "summary_metrics.png", "Cross HYDRA | Summary metrics")

        lines = ["Mode: Cross_HYDRA", "", "Summary metrics:"]
        lines.extend(summary_dt.to_string(index=False).splitlines())
        if skip_log:
            lines.append("")
            lines.append("Skip log:")
            lines.extend(skip_log)
        save_text_summary(stats_dir / "summary.txt", lines)
        gc.collect()


def run_pairwise_hydra(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str]) -> None:
    log("\n================ PAIRWISE HYDRA ================")
    plots_dir, stats_dir = prepare_mode_dirs(PAIRWISE_ROOT)

    metrics_all = []
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

            train_dt = all_dt[all_dt["analysis_dataset"] == train_ds]
            test_dt = all_dt[all_dt["analysis_dataset"] == test_ds]

            pair_plot = plots_dir / sanitize_name(pair_id)
            pair_stat = stats_dir / sanitize_name(pair_id)
            make_clean_dir(pair_plot)
            make_clean_dir(pair_stat)

            if train_dt.empty or test_dt.empty:
                skip_log.append(f"Skipped {pair_id}: empty train or test dataset")
                continue

            train_counts = train_dt.groupby("behavior", observed=False).size().to_dict()
            test_counts = test_dt.groupby("behavior", observed=False).size().to_dict()

            if not all(train_counts.get(b, 0) >= MIN_N_PAIRWISE_TRAIN_BEHAVIOR for b in KEEP_BEHAVIORS):
                skip_log.append(f"Skipped {pair_id}: train set missing behavior(s)")
                continue

            if sum(test_counts.get(b, 0) >= MIN_N_PAIRWISE_TEST_BEHAVIOR for b in KEEP_BEHAVIORS) < 2:
                skip_log.append(f"Skipped {pair_id}: test set has fewer than 2 usable behaviors")
                continue

            train_bal = cap_by_groups(train_dt, ["behavior"], CAP_N_PAIRWISE_TRAIN_BEHAVIOR, SEED)

            try:
                fit = fit_hydra_model(train_bal, test_dt, x_cols, y_cols, z_cols)
            except Exception as exc:
                msg = f"Skipped {pair_id}: {exc}"
                skip_log.append(msg)
                log_exception(msg, exc)
                continue

            metrics_dt = pd.DataFrame(fit["overall_metrics"]).copy()
            metrics_dt["pair_id"] = pair_id
            metrics_dt["train_dataset"] = train_ds
            metrics_dt["test_dataset"] = test_ds
            metrics_dt["train_family"] = family_lookup.get(train_ds, "")
            metrics_dt["test_family"] = family_lookup.get(test_ds, "")
            metrics_all.append(metrics_dt)

            train_balance = train_bal.groupby("behavior", observed=False).size().reset_index(name="N").sort_values("behavior")
            test_balance = test_dt.groupby("behavior", observed=False).size().reset_index(name="N").sort_values("behavior")
            train_balance.to_csv(pair_stat / "train_balance.csv", index=False)
            test_balance.to_csv(pair_stat / "test_balance.csv", index=False)

            write_basic_outputs(
                fit["cm_dt"],
                fit["overall_metrics"],
                fit["behavior_metrics"],
                pair_plot,
                pair_stat,
                f"Pairwise HYDRA | {pair_id}",
            )

            lines = [
                "Mode: Pairwise_HYDRA",
                f"Pair: {pair_id}",
                f"Train dataset: {train_ds}",
                f"Test dataset: {test_ds}",
                f"Train family: {family_lookup.get(train_ds, '')}",
                f"Test family: {family_lookup.get(test_ds, '')}",
                f"Train rows after balancing: {len(train_bal)}",
                f"Test rows: {len(test_dt)}",
                f"Hydra settings: {HYDRA_SETTINGS_NOTE}",
                f"Hydra n_jobs: {HYDRA_N_JOBS}",
                "",
                "Overall metrics:",
            ]
            lines.extend(pd.DataFrame(fit["overall_metrics"]).to_string(index=False).splitlines())
            save_text_summary(pair_stat / "summary.txt", lines)

            del train_dt, test_dt, train_bal, fit, metrics_dt, train_balance, test_balance
            gc.collect()

    if metrics_all:
        summary_dt = pd.concat(metrics_all, ignore_index=True).sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        summary_dt.to_csv(stats_dir / "metrics_all.csv", index=False)
        save_summary_metric_plot(summary_dt, "pair_id", plots_dir / "summary_metrics.png", "Pairwise HYDRA | Summary metrics")

        lines = ["Mode: Pairwise_HYDRA", "", "Summary metrics:"]
        lines.extend(summary_dt.to_string(index=False).splitlines())
        if skip_log:
            lines.append("")
            lines.append("Skip log:")
            lines.extend(skip_log)
        save_text_summary(stats_dir / "summary.txt", lines)
        gc.collect()


def format_metric_value(value) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def add_metric_overview(
    lines: list[str],
    title: str,
    csv_path: Path,
    id_cols: list[str],
    max_rows: Optional[int] = None,
) -> None:
    lines.append("")
    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)

    if not csv_path.exists():
        lines.append(f"Status: not available ({csv_path})")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        lines.append("Status: file exists, but contains no rows.")
        return

    metric_cols = [c for c in ["accuracy", "macro_recall", "macro_precision", "macro_f1"] if c in df.columns]
    lines.append(f"File: {csv_path}")
    lines.append(f"Rows: {len(df)}")

    for metric in metric_cols:
        values = pd.to_numeric(df[metric], errors="coerce")
        lines.append(
            f"{metric}: mean={format_metric_value(values.mean())} | "
            f"min={format_metric_value(values.min())} | "
            f"max={format_metric_value(values.max())}"
        )

    if "macro_f1" in df.columns:
        values = pd.to_numeric(df["macro_f1"], errors="coerce")
        valid = df.loc[values.notna()].copy()
        if not valid.empty:
            best = valid.loc[pd.to_numeric(valid["macro_f1"], errors="coerce").idxmax()]
            worst = valid.loc[pd.to_numeric(valid["macro_f1"], errors="coerce").idxmin()]

            def row_label(row: pd.Series) -> str:
                parts = [str(row[c]) for c in id_cols if c in row.index]
                return " | ".join(parts) if parts else "overall"

            lines.append(f"Best macro_f1: {row_label(best)} = {format_metric_value(best['macro_f1'])}")
            lines.append(f"Worst macro_f1: {row_label(worst)} = {format_metric_value(worst['macro_f1'])}")

    show_cols = [c for c in id_cols + metric_cols if c in df.columns]
    if show_cols:
        out_df = df[show_cols].copy()
        sort_cols = [c for c in ["macro_f1", "accuracy"] if c in out_df.columns]
        if sort_cols:
            out_df = out_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        if max_rows is not None:
            out_df = out_df.head(max_rows)
        lines.append("")
        lines.extend(out_df.to_string(index=False).splitlines())


def add_text_file_excerpt(lines: list[str], title: str, text_path: Path) -> None:
    lines.append("")
    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)

    if not text_path.exists():
        lines.append(f"Status: not available ({text_path})")
        return

    content = text_path.read_text(encoding="utf-8", errors="replace").splitlines()
    keep = [line for line in content if "Skipped" in line or "Skip log" in line or line.startswith("Mode:")]
    if not keep:
        lines.append("No skip notes found.")
        return
    lines.extend(keep)


def write_hydra_overview(
    all_dt: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    z_cols: list[str],
) -> None:
    lines: list[str] = []
    lines.append("HYDRA overview")
    lines.append("=" * 72)
    lines.append(f"Output root: {OUT_ROOT}")
    lines.append(f"Raw input root: {RAW_ROOT}")
    lines.append(f"Rows after cleaning: {len(all_dt)}")
    lines.append(f"Datasets: {all_dt['analysis_dataset'].nunique()}")
    lines.append(f"Behaviors kept: {', '.join(KEEP_BEHAVIORS)}")
    lines.append(f"Raw axis length: x={len(x_cols)}, y={len(y_cols)}, z={len(z_cols)}")
    lines.append("")
    lines.append("HYDRA and run settings")
    lines.append("-" * 72)
    lines.append(f"Seed: {SEED}")
    lines.append(f"Test fraction: {TEST_FRAC}")
    lines.append(f"HYDRA settings: {HYDRA_SETTINGS_NOTE}")
    lines.append(f"HYDRA n_jobs: {HYDRA_N_JOBS}")
    lines.append(f"Prediction chunk size: {PREDICT_CHUNK_SIZE}")
    lines.append(f"RUN_GLOBAL={RUN_GLOBAL}")
    lines.append(f"RUN_WITHIN={RUN_WITHIN}")
    lines.append(f"RUN_INTER={RUN_INTER}")
    lines.append(f"RUN_CROSS={RUN_CROSS}")
    lines.append(f"RUN_PAIRWISE={RUN_PAIRWISE}")

    counts = (
        all_dt.groupby(["analysis_dataset", "behavior"], dropna=False, observed=False)
        .size()
        .reset_index(name="N")
        .sort_values(["analysis_dataset", "behavior"])
    )
    lines.append("")
    lines.append("Rows by dataset and behavior")
    lines.append("-" * 72)
    lines.extend(counts.to_string(index=False).splitlines())

    add_metric_overview(
        lines,
        "Global_HYDRA overall metrics",
        GLOBAL_ROOT / "statistics" / "overall_metrics.csv",
        [],
    )
    add_metric_overview(
        lines,
        "Global_HYDRA metrics by dataset",
        GLOBAL_ROOT / "statistics" / "metrics_by_analysis_dataset.csv",
        ["analysis_dataset"],
    )
    add_metric_overview(
        lines,
        "Within_HYDRA metrics",
        WITHIN_ROOT / "statistics" / "metrics_all.csv",
        ["analysis_dataset"],
    )
    add_metric_overview(
        lines,
        "Inter_HYDRA metrics",
        INTER_ROOT / "statistics" / "metrics_all.csv",
        ["family_id", "pair_id", "train_dataset", "test_dataset"],
    )
    add_metric_overview(
        lines,
        "Cross_HYDRA metrics",
        CROSS_ROOT / "statistics" / "metrics_all.csv",
        ["test_dataset"],
    )
    add_metric_overview(
        lines,
        "Pairwise_HYDRA metrics",
        PAIRWISE_ROOT / "statistics" / "metrics_all.csv",
        ["pair_id", "train_dataset", "test_dataset"],
    )

    add_text_file_excerpt(lines, "Within_HYDRA skip notes", WITHIN_ROOT / "statistics" / "summary.txt")
    add_text_file_excerpt(lines, "Inter_HYDRA skip notes", INTER_ROOT / "statistics" / "summary.txt")
    add_text_file_excerpt(lines, "Cross_HYDRA skip notes", CROSS_ROOT / "statistics" / "summary.txt")
    add_text_file_excerpt(lines, "Pairwise_HYDRA skip notes", PAIRWISE_ROOT / "statistics" / "summary.txt")

    save_text_summary(OUT_ROOT / OVERVIEW_NAME, lines)


# =========================================================
# Main
# =========================================================
def main() -> int:
    try:
        set_seed(SEED)

        if HydraClassifier is None:
            raise ImportError(
                "aeon is not available. Install aeon before running 05_hydra.py."
            ) from HYDRA_IMPORT_ERROR

        enabled_modes = get_enabled_modes()
        if not enabled_modes:
            raise ValueError(
                "No run mode is enabled. Set at least one of RUN_GLOBAL/RUN_WITHIN/RUN_INTER/RUN_CROSS/RUN_PAIRWISE to True."
            )

        log("[INFO] Enabled modes: " + ", ".join(name for name, _, _ in enabled_modes))
        log("[INFO] Loading raw files...")
        all_dt, read_log = load_raw_data()
        make_clean_dir(OUT_ROOT)
        for _, root, _ in enabled_modes:
            make_clean_dir(root)
        save_text_summary(OUT_ROOT / "read_log.txt", read_log)
        save_text_summary(
            OUT_ROOT / "run_config.txt",
            [
                f"RUN_GLOBAL={RUN_GLOBAL}",
                f"RUN_WITHIN={RUN_WITHIN}",
                f"RUN_INTER={RUN_INTER}",
                f"RUN_CROSS={RUN_CROSS}",
                f"RUN_PAIRWISE={RUN_PAIRWISE}",
                f"BASE={BASE}",
                f"RAW_ROOT={RAW_ROOT}",
                f"OUT_ROOT={OUT_ROOT}",
                f"EXPECTED_DATASET_COUNT={EXPECTED_DATASET_COUNT}",
                f"CAP_N_GLOBAL_TRAIN_DATASET_BEHAVIOR={CAP_N_GLOBAL_TRAIN_DATASET_BEHAVIOR}",
                f"CAP_N_WITHIN_TRAIN_BEHAVIOR={CAP_N_WITHIN_TRAIN_BEHAVIOR}",
                f"CAP_N_INTER_TRAIN_BEHAVIOR={CAP_N_INTER_TRAIN_BEHAVIOR}",
                f"CAP_N_CROSS_TRAIN_DATASET_BEHAVIOR={CAP_N_CROSS_TRAIN_DATASET_BEHAVIOR}",
                f"CAP_N_PAIRWISE_TRAIN_BEHAVIOR={CAP_N_PAIRWISE_TRAIN_BEHAVIOR}",
            ],
        )

        log("[INFO] Cleaning raw table...")
        all_dt, x_cols, y_cols, z_cols = clean_raw_data(all_dt, OUT_ROOT / "unmatched_dataset_keys.csv")

        log("[INFO] Creating metadata-only tables for reports...")
        meta_dt = get_metadata_view(all_dt)

        counts_dt = (
            meta_dt.groupby(["analysis_dataset", "species", "behavior"], dropna=False, observed=True)
            .size().reset_index(name="N")
            .sort_values(["analysis_dataset", "species", "behavior"])
        )
        counts_dt.to_csv(OUT_ROOT / "counts_by_dataset_species_behavior.csv", index=False)
        log(f"[OK] CSV saved: {OUT_ROOT / 'counts_by_dataset_species_behavior.csv'}")

        lookup_group_cols = [
            col for col in [
                "analysis_dataset", "dataset_name", "dataset_folder", "analysis_family",
                "species", "dataset_group", "dataset_id", "dataset_key",
            ]
            if col in meta_dt.columns
        ]
        lookup_dt = (
            meta_dt.groupby(lookup_group_cols, dropna=False, observed=True)
            .agg(
                rows=("behavior", "size"),
                n_individual_keys=("individual_key", pd.Series.nunique),
                n_individual_ids=("individual_id", pd.Series.nunique),
                n_names=("name", pd.Series.nunique),
                n_subject_keys=("subject_key", pd.Series.nunique),
                n_source_files=("source_file", pd.Series.nunique),
            )
            .reset_index()
            .sort_values(["analysis_dataset", "dataset_folder"])
        )
        lookup_dt.to_csv(OUT_ROOT / "analysis_dataset_lookup.csv", index=False)
        log(f"[OK] CSV saved: {OUT_ROOT / 'analysis_dataset_lookup.csv'}")

        save_duplicate_report(meta_dt, OUT_ROOT / "duplicate_window_report.csv")

        del counts_dt, lookup_dt, meta_dt
        gc.collect()

        for mode_name, _, mode_func in enabled_modes:
            log(f"[INFO] Starting mode: {mode_name}")
            mode_func(all_dt, x_cols, y_cols, z_cols)
            gc.collect()
            log(f"[OK] Finished mode: {mode_name}")

        write_hydra_overview(all_dt, x_cols, y_cols, z_cols)

        log("[DONE] HYDRA pipeline finished.")
        return 0

    except Exception as exc:
        log_exception("Fatal error", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
