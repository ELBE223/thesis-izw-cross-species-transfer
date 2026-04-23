#!/usr/bin/env python3
from __future__ import annotations

import gc
import math
import random
import re
import traceback
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from aeon.classification.convolution_based import MultiRocketClassifier
    MULTIROCKET_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    MultiRocketClassifier = None  # type: ignore[assignment]
    MULTIROCKET_IMPORT_ERROR = exc


# =========================================================
# Paths
# =========================================================
BASE = Path("/Volumes/Z Slim/07_04_2026_Data_Analysis")
RAW_ROOT = BASE / "Output" / "Pre" / "raw_by_dataset"
OUT_ROOT = BASE / "Output" / "MultiRocket"

GLOBAL_ROOT = OUT_ROOT / "Global_MultiRocket"
WITHIN_ROOT = OUT_ROOT / "Within_MultiRocket"
INTER_ROOT = OUT_ROOT / "Inter_MultiRocket"
CROSS_ROOT = OUT_ROOT / "Cross_MultiRocket"
PAIRWISE_ROOT = OUT_ROOT / "Pairwise_MultiRocket"


# =========================================================
# Settings
# =========================================================
SEED = 42
KEEP_BEHAVIORS = ["Foraging", "Locomotion", "Resting"]
REMOVE_PADDED = False
TEST_FRAC = 0.25

MIN_N_GLOBAL_DATASET_BEHAVIOR = 30
MIN_N_WITHIN_DATASET_BEHAVIOR = 30
MIN_N_INTER_TRAIN_BEHAVIOR = 30
MIN_N_INTER_TEST_BEHAVIOR = 5
MIN_N_CROSS_TRAIN_BEHAVIOR = 30
MIN_N_CROSS_TEST_BEHAVIOR = 5
MIN_N_PAIRWISE_TRAIN_BEHAVIOR = 30
MIN_N_PAIRWISE_TEST_BEHAVIOR = 5

CAP_N_GLOBAL_TRAIN_DATASET_BEHAVIOR = 500
CAP_N_WITHIN_TRAIN_BEHAVIOR = 500
CAP_N_INTER_TRAIN_BEHAVIOR = 500
CAP_N_CROSS_TRAIN_DATASET_BEHAVIOR = 500
CAP_N_PAIRWISE_TRAIN_BEHAVIOR = 500

MULTIROCKET_N_KERNELS = 5000
MULTIROCKET_MAX_DILATIONS_PER_KERNEL = 16
MULTIROCKET_N_FEATURES_PER_KERNEL = 4
MULTIROCKET_N_JOBS = 4
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
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*.parquet")
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
    for _, sub in df.groupby(group_cols, dropna=False, sort=False, observed=False):
        parts.append(sample_cap(sub.copy(), n_cap, rng))
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def choose_split_var(df: pd.DataFrame) -> Optional[str]:
    for col in ["name", "subject_key", "source_file"]:
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
            skip_log.append(f"Skipped {dataset_name}: no valid split unit (name/subject_key/source_file)")
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


def compute_class_weights(y: pd.Series) -> dict[str, float]:
    counts = y.value_counts()
    total = counts.sum()
    n_classes = len(counts)
    return {cls: float(total / (n_classes * n)) for cls, n in counts.items() if n > 0}


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
    top_folder = safe_str(top_folder)
    species = safe_str(species)
    dataset_id = safe_str(dataset_id)
    dataset_folder = safe_str(dataset_folder)

    if top_folder:
        return top_folder

    key = (dataset_id or dataset_folder).lower()

    if species == "Bison bonasus":
        return "Bison"
    if species == "Canis lupus familiaris":
        return "Dog"
    if species == "Giraffa camelopardalis":
        return "Giraffe"
    if species == "Erinaceus europaeus":
        return "Hedgehog"
    if species == "Equus ferus przewalskii" or key.startswith("horse_dataset_1"):
        return "Horse_dataset_1"
    if species == "Equus caballus" or key.startswith("horse_dataset_2"):
        return "Horse_dataset_2"
    if species == "Vulpes vulpes":
        return "Fox_dataset_1" if "fox_dataset_1" in key else "Fox_dataset_2"
    if species == "Procyon lotor":
        return "Raccoon_dataset_1" if "raccoon_dataset_1" in key else "Raccoon_dataset_2"
    return "UNMAPPED"


def add_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["top_folder", "dataset_folder", "dataset_id", "dataset_group", "species", "name", "subject_key", "source_file", "behavior"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["analysis_dataset"] = [
        infer_analysis_dataset(t, s, d, f)
        for t, s, d, f in zip(df["top_folder"], df["species"], df["dataset_id"], df["dataset_folder"])
    ]

    df["analysis_family"] = df["analysis_dataset"].str.replace(r"_dataset_[0-9]+$", "", regex=True)
    df["dataset_key"] = np.where(
        df["dataset_group"] != "",
        df["dataset_group"] + "__" + df["dataset_id"],
        np.where(df["dataset_id"] != "", df["dataset_id"], df["dataset_folder"]),
    )
    return df


def validate_no_unmapped(df: pd.DataFrame) -> None:
    bad = df[df["analysis_dataset"] == "UNMAPPED"].copy()
    if bad.empty:
        return

    cols = [c for c in ["species", "dataset_id", "dataset_folder", "top_folder", "source_file", "full_raw_path"] if c in bad.columns]
    preview = bad[cols].drop_duplicates().head(10)

    log("[ERROR] Found rows with analysis_dataset = UNMAPPED.")
    log(f"[ERROR] UNMAPPED rows: {len(bad)}")
    if not preview.empty:
        log("[ERROR] First UNMAPPED combinations:")
        for row in preview.to_dict(orient="records"):
            parts = [f"{k}={safe_str(v)}" for k, v in row.items()]
            log("[ERROR]   - " + " | ".join(parts))

    raise ValueError(
        "UNMAPPED analysis_dataset detected. Check top_folder/species/dataset_id/dataset_folder mapping before training."
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
    table = pq.read_table(path)
    df = table.to_pandas()
    rel = path.relative_to(RAW_ROOT)
    parts = rel.parts
    top_folder = parts[0] if len(parts) >= 3 else ""
    dataset_folder = parts[1] if len(parts) >= 3 else path.parent.name
    df["top_folder"] = top_folder
    df["dataset_folder"] = dataset_folder
    df["part_file"] = path.name
    df["full_raw_path"] = str(path)
    return df


def load_raw_data() -> tuple[pd.DataFrame, list[str]]:
    files = valid_parquet_files(RAW_ROOT)
    if not files:
        raise FileNotFoundError(f"No raw parquet files found in {RAW_ROOT}")

    frames: list[pd.DataFrame] = []
    read_log: list[str] = []

    for i, file_path in enumerate(files, start=1):
        log("\n========================================")
        log(f"[INFO] File {i} of {len(files)}")
        log(f"[INFO] Input: {file_path}")
        try:
            df = read_raw_file(file_path)
            frames.append(df)
            read_log.append(f"OK | {file_path}")
            log(f"[OK] Rows: {len(df)}")
        except Exception as exc:
            read_log.append(f"ERROR | {file_path} | {exc}")
            log_exception(f"Could not read file {file_path}", exc)

    if not frames:
        raise RuntimeError("No raw files could be read successfully. See read_log.txt for details.")

    all_dt = pd.concat(frames, ignore_index=True)
    log(f"[INFO] Combined raw rows: {len(all_dt)}")
    return all_dt, read_log


def clean_raw_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    n_input = len(df)
    df = add_analysis_columns(df)

    df = df[df["behavior"].isin(KEEP_BEHAVIORS)].copy()
    log(f"[INFO] Rows after behavior filter: {len(df)} / {n_input}")

    df = df[df["species"] != ""].copy()
    log(f"[INFO] Rows after species filter: {len(df)}")

    validate_no_unmapped(df)

    if REMOVE_PADDED and "padded_end" in df.columns:
        before_padded = len(df)
        padded = df["padded_end"]
        if padded.dtype == bool:
            keep_mask = ~padded.fillna(False)
        else:
            keep_mask = padded.fillna(False).astype(str).str.lower().isin(["false", "0", ""]) | padded.isna()
        df = df[keep_mask].copy()
        log(f"[INFO] Rows after padded filter: {len(df)} (removed {before_padded - len(df)})")

    x_cols, y_cols, z_cols = detect_axis_cols(df)
    before_dropna = len(df)
    df = df.dropna(subset=x_cols + y_cols + z_cols).copy()
    log(f"[INFO] Rows after raw axis NA filter: {len(df)} (removed {before_dropna - len(df)})")

    if df.empty:
        raise RuntimeError("No usable rows left after raw data cleaning.")

    axis_cols = x_cols + y_cols + z_cols
    df.loc[:, axis_cols] = df.loc[:, axis_cols].astype(np.float32, copy=False)
    df["behavior"] = df["behavior"].astype(str)

    for col in ["behavior", "species", "analysis_dataset", "analysis_family", "dataset_folder", "dataset_group", "dataset_id", "dataset_key", "top_folder"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    gc.collect()
    return df.reset_index(drop=True), x_cols, y_cols, z_cols


def get_metadata_view(df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "analysis_dataset", "analysis_family", "species", "behavior",
        "dataset_folder", "dataset_group", "dataset_id", "dataset_key",
        "top_folder", "name", "subject_key", "source_file",
        "window_id", "window_start", "part_file", "full_raw_path",
    ]
    keep_cols = [col for col in meta_cols if col in df.columns]
    meta_dt = df.loc[:, keep_cols].copy()

    for col in [
        "analysis_dataset", "analysis_family", "species", "behavior",
        "dataset_folder", "dataset_group", "dataset_id", "dataset_key",
        "top_folder", "name", "subject_key", "source_file", "part_file",
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
def fit_multirocket_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    z_cols: list[str],
) -> dict[str, object]:
    if MultiRocketClassifier is None:
        raise ImportError(
            "aeon is required for MultiRocketClassifier. Install aeon first."
        ) from MULTIROCKET_IMPORT_ERROR

    if train_df.empty:
        raise ValueError("fit_multirocket_model received an empty training table.")
    if test_df.empty:
        raise ValueError("fit_multirocket_model received an empty test table.")

    log(f"[INFO] Building train tensor: {len(train_df)} rows")
    X_train = make_tensor_from_df(train_df, x_cols, y_cols, z_cols)
    y_train = train_df["behavior"].astype(str)
    y_test = test_df["behavior"].astype(str)

    class_weights = compute_class_weights(y_train)

    model = MultiRocketClassifier(
        n_kernels=MULTIROCKET_N_KERNELS,
        max_dilations_per_kernel=MULTIROCKET_MAX_DILATIONS_PER_KERNEL,
        n_features_per_kernel=MULTIROCKET_N_FEATURES_PER_KERNEL,
        class_weight=class_weights,
        n_jobs=MULTIROCKET_N_JOBS,
        random_state=SEED,
    )
    log(f"[INFO] Starting MultiRocket fit: train={len(train_df)}, test={len(test_df)}")
    model.fit(X_train, y_train.to_numpy())
    log("[OK] MultiRocket fit finished")

    del X_train
    gc.collect()

    log(f"[INFO] Starting prediction on {len(test_df)} rows")
    pred = pd.Series(
        predict_in_chunks(model, test_df, x_cols, y_cols, z_cols),
        index=test_df.index,
        dtype="object",
    )
    log("[OK] Prediction finished")
    truth = pd.Series(y_test.to_numpy(), index=test_df.index, dtype="object")

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


def get_enabled_modes() -> list[tuple[str, Path, Callable[..., None]]]:
    modes: list[tuple[str, Path, Callable[..., None]]] = []

    if RUN_GLOBAL:
        modes.append(("GLOBAL", GLOBAL_ROOT, run_global_multirocket))
    if RUN_WITHIN:
        modes.append(("WITHIN", WITHIN_ROOT, run_within_multirocket))
    if RUN_INTER:
        modes.append(("INTER", INTER_ROOT, run_inter_multirocket))
    if RUN_CROSS:
        modes.append(("CROSS", CROSS_ROOT, run_cross_multirocket))
    if RUN_PAIRWISE:
        modes.append(("PAIRWISE", PAIRWISE_ROOT, run_pairwise_multirocket))

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
def run_global_multirocket(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str]) -> None:
    log("\n================ GLOBAL MULTIROCKET ================")
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

    fit = fit_multirocket_model(train_bal, test_dt, x_cols, y_cols, z_cols)

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
        "Global MultiRocket",
    )
    save_summary_metric_plot(metrics_by_dataset, "analysis_dataset", plots_dir / "dataset_metrics.png", "Global MultiRocket | Metrics by dataset")

    lines = [
        "Mode: Global_MULTIROCKET",
        f"Rows after filters: {len(all_dt)}",
        f"Rows in global model: {len(split_dt)}",
        f"Train rows after balancing: {len(train_bal)}",
        f"Test rows: {len(test_dt)}",
        f"remove_padded: {REMOVE_PADDED}",
        f"MultiRocket n_kernels: {MULTIROCKET_N_KERNELS}",
        f"MultiRocket max_dilations_per_kernel: {MULTIROCKET_MAX_DILATIONS_PER_KERNEL}",
        f"MultiRocket n_features_per_kernel: {MULTIROCKET_N_FEATURES_PER_KERNEL}",
        f"MultiRocket n_jobs: {MULTIROCKET_N_JOBS}",
        f"Global datasets used: {', '.join(sorted(global_dt['analysis_dataset'].unique()))}",
        "Split skips: none" if not skip_log else "Split skips:\n  - " + "\n  - ".join(skip_log),
        "",
        "Overall metrics:",
    ]
    lines.extend(pd.DataFrame(fit["overall_metrics"]).to_string(index=False).splitlines())
    save_text_summary(stats_dir / "summary.txt", lines)

    del global_dt, split_dt, train_dt, test_dt, train_bal, fit, metrics_by_dataset, train_balance, test_balance
    plt.close("all")
    gc.collect()


def run_within_multirocket(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str]) -> None:
    log("\n================ WITHIN MULTIROCKET ================")
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

        ds_dt = all_dt[all_dt["analysis_dataset"] == ds].copy()
        split_dt, split_skip = make_holdout_split(ds_dt, "analysis_dataset", TEST_FRAC, SEED)
        if split_dt is None:
            skip_log.append(f"Skipped {ds}: no valid split")
            continue

        train_dt = split_dt[split_dt["in_test"] == False].copy()
        test_dt = split_dt[split_dt["in_test"] == True].copy()

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
            fit = fit_multirocket_model(train_bal, test_dt, x_cols, y_cols, z_cols)
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
            f"Within MultiRocket | {ds}",
        )

        lines = [
            "Mode: Within_MULTIROCKET",
            f"Dataset: {ds}",
            f"Rows: {len(split_dt)}",
            f"Train rows after balancing: {len(train_bal)}",
            f"Test rows: {len(test_dt)}",
            f"remove_padded: {REMOVE_PADDED}",
            f"MultiRocket n_kernels: {MULTIROCKET_N_KERNELS}",
            f"MultiRocket max_dilations_per_kernel: {MULTIROCKET_MAX_DILATIONS_PER_KERNEL}",
            f"MultiRocket n_features_per_kernel: {MULTIROCKET_N_FEATURES_PER_KERNEL}",
            f"MultiRocket n_jobs: {MULTIROCKET_N_JOBS}",
            f"Split var used: {', '.join(sorted(split_dt['split_var_used'].dropna().astype(str).unique()))}",
            "Split skips: none" if not split_skip else "Split skips:\n  - " + "\n  - ".join(split_skip),
            "",
            "Overall metrics:",
        ]
        lines.extend(metrics_dt[["accuracy", "macro_recall", "macro_precision", "macro_f1"]].to_string(index=False).splitlines())
        save_text_summary(ds_stat / "summary.txt", lines)

        del ds_dt, split_dt, train_dt, test_dt, train_bal, fit, metrics_dt, train_balance, test_balance
        plt.close("all")
        gc.collect()

    if metrics_all:
        summary_dt = pd.concat(metrics_all, ignore_index=True).sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        summary_dt.to_csv(stats_dir / "metrics_all.csv", index=False)
        save_summary_metric_plot(summary_dt, "analysis_dataset", plots_dir / "summary_metrics.png", "Within MultiRocket | Summary metrics")

        lines = ["Mode: Within_MULTIROCKET", "", "Summary metrics:"]
        lines.extend(summary_dt.to_string(index=False).splitlines())
        if skip_log:
            lines.append("")
            lines.append("Skip log:")
            lines.extend(skip_log)
        save_text_summary(stats_dir / "summary.txt", lines)


def run_inter_multirocket(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str]) -> None:
    log("\n================ INTER MULTIROCKET ================")
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

            train_dt = all_dt[all_dt["analysis_dataset"] == train_ds].copy()
            test_dt = all_dt[all_dt["analysis_dataset"] == test_ds].copy()

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
                fit = fit_multirocket_model(train_bal, test_dt, x_cols, y_cols, z_cols)
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
                f"Inter MultiRocket | {pair_id}",
            )

            lines = [
                "Mode: Inter_MULTIROCKET",
                f"Family: {family}",
                f"Train dataset: {train_ds}",
                f"Test dataset: {test_ds}",
                f"Train rows after balancing: {len(train_bal)}",
                f"Test rows: {len(test_dt)}",
                f"remove_padded: {REMOVE_PADDED}",
                f"MultiRocket n_kernels: {MULTIROCKET_N_KERNELS}",
                f"MultiRocket max_dilations_per_kernel: {MULTIROCKET_MAX_DILATIONS_PER_KERNEL}",
                f"MultiRocket n_features_per_kernel: {MULTIROCKET_N_FEATURES_PER_KERNEL}",
                f"MultiRocket n_jobs: {MULTIROCKET_N_JOBS}",
                "",
                "Overall metrics:",
            ]
            lines.extend(pd.DataFrame(fit["overall_metrics"]).to_string(index=False).splitlines())
            save_text_summary(pair_stat / "summary.txt", lines)

            del train_dt, test_dt, train_bal, fit, metrics_dt, train_balance, test_balance
            plt.close("all")
            gc.collect()

    if metrics_all:
        summary_dt = pd.concat(metrics_all, ignore_index=True).sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        summary_dt.to_csv(stats_dir / "metrics_all.csv", index=False)
        save_summary_metric_plot(summary_dt, "pair_id", plots_dir / "summary_metrics.png", "Inter MultiRocket | Summary metrics")

        lines = ["Mode: Inter_MULTIROCKET", "", "Summary metrics:"]
        lines.extend(summary_dt.to_string(index=False).splitlines())
        if skip_log:
            lines.append("")
            lines.append("Skip log:")
            lines.extend(skip_log)
        save_text_summary(stats_dir / "summary.txt", lines)


def run_cross_multirocket(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str]) -> None:
    log("\n================ CROSS MULTIROCKET ================")
    plots_dir, stats_dir = prepare_mode_dirs(CROSS_ROOT)

    metrics_all = []
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
            fit = fit_multirocket_model(train_bal, test_dt, x_cols, y_cols, z_cols)
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
            f"Cross MultiRocket | {test_ds}",
        )

        lines = [
            "Mode: Cross_MULTIROCKET",
            f"Test dataset: {test_ds}",
            f"Train datasets: {', '.join(sorted(train_bal['analysis_dataset'].unique()))}",
            f"Train rows after balancing: {len(train_bal)}",
            f"Test rows: {len(test_dt)}",
            f"remove_padded: {REMOVE_PADDED}",
            f"MultiRocket n_kernels: {MULTIROCKET_N_KERNELS}",
            f"MultiRocket max_dilations_per_kernel: {MULTIROCKET_MAX_DILATIONS_PER_KERNEL}",
            f"MultiRocket n_features_per_kernel: {MULTIROCKET_N_FEATURES_PER_KERNEL}",
            f"MultiRocket n_jobs: {MULTIROCKET_N_JOBS}",
            "",
            "Overall metrics:",
        ]
        lines.extend(pd.DataFrame(fit["overall_metrics"]).to_string(index=False).splitlines())
        save_text_summary(out_stat / "summary.txt", lines)

        del train_dt, test_dt, train_bal, fit, metrics_dt, train_balance, test_balance
        plt.close("all")
        gc.collect()

    if metrics_all:
        summary_dt = pd.concat(metrics_all, ignore_index=True).sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        summary_dt.to_csv(stats_dir / "metrics_all.csv", index=False)
        save_summary_metric_plot(summary_dt, "test_dataset", plots_dir / "summary_metrics.png", "Cross MultiRocket | Summary metrics")

        lines = ["Mode: Cross_MULTIROCKET", "", "Summary metrics:"]
        lines.extend(summary_dt.to_string(index=False).splitlines())
        if skip_log:
            lines.append("")
            lines.append("Skip log:")
            lines.extend(skip_log)
        save_text_summary(stats_dir / "summary.txt", lines)


def run_pairwise_multirocket(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str]) -> None:
    log("\n================ PAIRWISE MULTIROCKET ================")
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

            train_dt = all_dt[all_dt["analysis_dataset"] == train_ds].copy()
            test_dt = all_dt[all_dt["analysis_dataset"] == test_ds].copy()

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
                fit = fit_multirocket_model(train_bal, test_dt, x_cols, y_cols, z_cols)
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
                f"Pairwise MultiRocket | {pair_id}",
            )

            lines = [
                "Mode: Pairwise_MULTIROCKET",
                f"Pair: {pair_id}",
                f"Train dataset: {train_ds}",
                f"Test dataset: {test_ds}",
                f"Train family: {family_lookup.get(train_ds, '')}",
                f"Test family: {family_lookup.get(test_ds, '')}",
                f"Train rows after balancing: {len(train_bal)}",
                f"Test rows: {len(test_dt)}",
                f"remove_padded: {REMOVE_PADDED}",
                f"MultiRocket n_kernels: {MULTIROCKET_N_KERNELS}",
                f"MultiRocket max_dilations_per_kernel: {MULTIROCKET_MAX_DILATIONS_PER_KERNEL}",
                f"MultiRocket n_features_per_kernel: {MULTIROCKET_N_FEATURES_PER_KERNEL}",
                f"MultiRocket n_jobs: {MULTIROCKET_N_JOBS}",
                "",
                "Overall metrics:",
            ]
            lines.extend(pd.DataFrame(fit["overall_metrics"]).to_string(index=False).splitlines())
            save_text_summary(pair_stat / "summary.txt", lines)

            del train_dt, test_dt, train_bal, fit, metrics_dt, train_balance, test_balance
            plt.close("all")
            gc.collect()

    if metrics_all:
        summary_dt = pd.concat(metrics_all, ignore_index=True).sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        summary_dt.to_csv(stats_dir / "metrics_all.csv", index=False)
        save_summary_metric_plot(summary_dt, "pair_id", plots_dir / "summary_metrics.png", "Pairwise MultiRocket | Summary metrics")

        lines = ["Mode: Pairwise_MULTIROCKET", "", "Summary metrics:"]
        lines.extend(summary_dt.to_string(index=False).splitlines())
        if skip_log:
            lines.append("")
            lines.append("Skip log:")
            lines.extend(skip_log)
        save_text_summary(stats_dir / "summary.txt", lines)


# =========================================================
# Main
# =========================================================
def main() -> int:
    try:
        set_seed(SEED)

        if MultiRocketClassifier is None:
            raise ImportError(
                "aeon is not available. Install aeon before running this script."
            ) from MULTIROCKET_IMPORT_ERROR

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
            ],
        )

        log("[INFO] Cleaning raw table...")
        all_dt, x_cols, y_cols, z_cols = clean_raw_data(all_dt)

        log("[INFO] Creating metadata-only tables for reports...")
        meta_dt = get_metadata_view(all_dt)

        counts_dt = (
            meta_dt.groupby(["analysis_dataset", "species", "behavior"], dropna=False, observed=True)
            .size().reset_index(name="N")
            .sort_values(["analysis_dataset", "species", "behavior"])
        )
        counts_dt.to_csv(OUT_ROOT / "counts_by_dataset_species_behavior.csv", index=False)
        log(f"[OK] CSV saved: {OUT_ROOT / 'counts_by_dataset_species_behavior.csv'}")

        lookup_dt = (
            meta_dt.groupby(
                ["analysis_dataset", "dataset_folder", "analysis_family", "species", "dataset_group", "dataset_id", "dataset_key"],
                dropna=False,
                observed=True,
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
        lookup_dt.to_csv(OUT_ROOT / "analysis_dataset_lookup.csv", index=False)
        log(f"[OK] CSV saved: {OUT_ROOT / 'analysis_dataset_lookup.csv'}")

        save_duplicate_report(meta_dt, OUT_ROOT / "duplicate_window_report.csv")

        del counts_dt, lookup_dt, meta_dt
        gc.collect()

        for mode_name, _, run_fn in enabled_modes:
            log(f"[INFO] Starting mode: {mode_name}")
            run_fn(all_dt, x_cols, y_cols, z_cols)

        log("[DONE] MULTIROCKET pipeline finished.")
        return 0

    except Exception as exc:
        log_exception("Fatal error", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
