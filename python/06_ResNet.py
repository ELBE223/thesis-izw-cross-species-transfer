#!/usr/bin/env python3
from __future__ import annotations

import math
import random
import re
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Paths
# =========================================================
BASE = Path("/Volumes/Z Slim/07_04_2026_Data_Analysis")
RAW_ROOT = BASE / "Output" / "Pre" / "raw_by_dataset"
OUT_ROOT = BASE / "Output" / "ResNet"

GLOBAL_ROOT = OUT_ROOT / "Global_ResNet"
WITHIN_ROOT = OUT_ROOT / "Within_ResNet"
INTER_ROOT = OUT_ROOT / "Inter_ResNet"
CROSS_ROOT = OUT_ROOT / "Cross_ResNet"
PAIRWISE_ROOT = OUT_ROOT / "Pairwise_ResNet"


# =========================================================
# Settings
# =========================================================
SEED = 42
KEEP_BEHAVIORS = ["Foraging", "Locomotion", "Resting"]
LABEL_TO_INT = {label: i for i, label in enumerate(KEEP_BEHAVIORS)}
INT_TO_LABEL = {i: label for label, i in LABEL_TO_INT.items()}

REMOVE_PADDED = False
TEST_FRAC = 0.25
VAL_FRAC = 0.20

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

BATCH_SIZE = 256
EPOCHS = 30
PATIENCE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25

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
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    for _, sub in df.groupby(group_cols, dropna=False, sort=False):
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

    for dataset_name, sub in df.groupby(dataset_col, dropna=False):
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


def save_history_plot(history_df: pd.DataFrame, out_file: Path, title: str) -> None:
    if history_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
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
        raise RuntimeError("No raw files could be read successfully. See resnet_read_log.txt for details.")

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

    df["behavior"] = df["behavior"].astype(str)
    return df.reset_index(drop=True), x_cols, y_cols, z_cols


def save_duplicate_report(df: pd.DataFrame, out_file: Path) -> None:
    key_cols = [c for c in ["analysis_dataset", "source_file", "window_id", "window_start", "part_file"] if c in df.columns]
    if not key_cols:
        pd.DataFrame({"note": ["No duplicate key columns found."]}).to_csv(out_file, index=False)
        log(f"[OK] CSV saved: {out_file}")
        return

    dup = (
        df.groupby(key_cols, dropna=False)
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
    x = df[x_cols].to_numpy(dtype=np.float32)
    y = df[y_cols].to_numpy(dtype=np.float32)
    z = df[z_cols].to_numpy(dtype=np.float32)
    arr = np.stack([x, y, z], axis=1)
    return arr.astype(np.float32)


def standardize_train_test(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test, mean.squeeze(), std.squeeze()


def encode_labels(y: pd.Series) -> np.ndarray:
    return y.map(LABEL_TO_INT).to_numpy(dtype=np.int64)


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# =========================================================
# Model
# =========================================================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class TimeSeriesResNet(nn.Module):
    def __init__(self, n_classes: int = 3, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(64, 64, kernel_size=7, dropout=dropout),
            ResidualBlock1D(64, 128, kernel_size=5, dropout=dropout),
            ResidualBlock1D(128, 128, kernel_size=3, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.classifier(x)


def find_validation_unsupported_datasets(
    df: pd.DataFrame,
    dataset_col: str,
) -> tuple[list[str], list[str]]:
    del df, dataset_col
    return [], []


def build_validation_split(train_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    del seed
    if train_df.empty:
        raise ValueError("Validation split failed: training data is empty.")
    return train_df.copy(), train_df.iloc[0:0].copy(), ["Validation disabled for alignment with HYDRA/MultiRocket."]


def fit_resnet_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    z_cols: list[str],
    device: torch.device,
) -> dict[str, object]:
    if train_df.empty:
        raise ValueError("fit_resnet_model received an empty training table.")
    if test_df.empty:
        raise ValueError("fit_resnet_model received an empty test table.")

    train_main_df = train_df.copy()
    val_df = train_df.iloc[0:0].copy()
    del val_df
    val_split_notes = ["Validation disabled for alignment with HYDRA/MultiRocket."]

    log(
        f"[INFO] Training split ready | train={len(train_main_df)} | "
        f"test={len(test_df)} | validation=disabled"
    )

    X_train = make_tensor_from_df(train_main_df, x_cols, y_cols, z_cols)
    X_test = make_tensor_from_df(test_df, x_cols, y_cols, z_cols)

    y_train = encode_labels(train_main_df["behavior"])
    y_test = encode_labels(test_df["behavior"])

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    mean_vec = mean.squeeze()
    std_vec = std.squeeze()

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(WindowDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = TimeSeriesResNet(n_classes=len(KEEP_BEHAVIORS), dropout=DROPOUT).to(device)

    class_weights = compute_class_weights(train_main_df["behavior"])
    weight_tensor = torch.tensor(
        [class_weights.get(label, 1.0) for label in KEEP_BEHAVIORS],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history_rows = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(yb)
            train_n += len(yb)

        train_loss = train_loss_sum / max(train_n, 1)
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": np.nan})
        log(f"[INFO] Epoch {epoch:02d} | train_loss={train_loss:.4f}")

    model.eval()
    pred_list = []
    true_list = []
    prob_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(axis=1)

            prob_list.append(probs)
            pred_list.append(pred)
            true_list.append(yb.numpy())

    y_pred_int = np.concatenate(pred_list)
    y_true_int = np.concatenate(true_list)
    y_prob = np.concatenate(prob_list)

    truth = pd.Series([INT_TO_LABEL[int(i)] for i in y_true_int], index=test_df.index, dtype="object")
    pred = pd.Series([INT_TO_LABEL[int(i)] for i in y_pred_int], index=test_df.index, dtype="object")

    cm_dt = build_confusion_df(truth, pred)
    overall_metrics = get_metrics(truth, pred)
    behavior_metrics = get_behavior_metrics(truth, pred)

    return {
        "model": model,
        "pred": pred,
        "truth": truth,
        "prob": y_prob,
        "cm_dt": cm_dt,
        "overall_metrics": overall_metrics,
        "behavior_metrics": behavior_metrics,
        "history_df": pd.DataFrame(history_rows),
        "norm_mean": mean_vec,
        "norm_std": std_vec,
        "train_main_n": len(train_main_df),
        "val_n": 0,
        "test_n": len(test_df),
        "val_split_notes": val_split_notes,
        "val_split_vars": [],
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


def write_basic_outputs(
    cm_dt: pd.DataFrame,
    metrics_dt: pd.DataFrame,
    behavior_dt: pd.DataFrame,
    history_df: pd.DataFrame,
    plot_root: Path,
    stat_root: Path,
    title_prefix: str,
) -> None:
    cm_dt.to_csv(stat_root / "confusion_matrix.csv", index=False)
    metrics_dt.to_csv(stat_root / "overall_metrics.csv", index=False)
    behavior_dt.to_csv(stat_root / "behavior_metrics.csv", index=False)
    history_df.to_csv(stat_root / "training_history.csv", index=False)

    save_cm_plot(cm_dt, plot_root / "confusion_matrix.png", f"{title_prefix} | Confusion matrix")
    save_behavior_metric_plot(behavior_dt, plot_root / "behavior_metrics.png", f"{title_prefix} | Behavior metrics")
    save_history_plot(history_df, plot_root / "training_history.png", f"{title_prefix} | Loss")


# =========================================================
# Modes
# =========================================================
def run_global_resnet(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str], device: torch.device) -> None:
    log("\n================ GLOBAL RESNET ================")
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

    fit = fit_resnet_model(train_bal, test_dt, x_cols, y_cols, z_cols, device)

    metrics_by_dataset = []
    for ds, sub in test_dt.groupby("analysis_dataset", dropna=False):
        truth = pd.Series(fit["truth"]).loc[sub.index]
        pred = pd.Series(fit["pred"]).loc[sub.index]
        row = get_metrics(truth, pred)
        row["analysis_dataset"] = ds
        metrics_by_dataset.append(row)

    metrics_by_dataset = pd.concat(metrics_by_dataset, ignore_index=True)
    metrics_by_dataset = metrics_by_dataset.sort_values(["macro_f1", "accuracy"], ascending=[False, False])
    metrics_by_dataset.to_csv(stats_dir / "metrics_by_analysis_dataset.csv", index=False)

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
    train_balance.to_csv(stats_dir / "train_balance.csv", index=False)
    test_balance.to_csv(stats_dir / "test_balance.csv", index=False)

    write_basic_outputs(
        fit["cm_dt"],
        fit["overall_metrics"],
        fit["behavior_metrics"],
        fit["history_df"],
        plots_dir,
        stats_dir,
        "Global ResNet",
    )
    save_summary_metric_plot(metrics_by_dataset, "analysis_dataset", plots_dir / "dataset_metrics.png", "Global ResNet | Metrics by dataset")

    summary_lines = [
        "Mode: Global_ResNet",
        f"Device: {device}",
        f"Train rows after balancing: {len(train_bal)}",
        f"Train rows used for fitting: {fit['train_main_n']}",
        f"Validation rows: {fit['val_n']}",
        f"Validation mode: disabled (aligned to HYDRA/MultiRocket)",
        f"Validation split vars: {', '.join(fit['val_split_vars']) if fit['val_split_vars'] else 'n/a'}",
        f"Test rows: {len(test_dt)}",
        f"Epochs max: {EPOCHS}",
        f"Batch size: {BATCH_SIZE}",
        f"Learning rate: {LEARNING_RATE}",
        f"Weight decay: {WEIGHT_DECAY}",
        "",
        "Overall metrics:",
    ]
    summary_lines.extend(pd.DataFrame(fit["overall_metrics"]).to_string(index=False).splitlines())
    summary_lines.append("")
    summary_lines.append("Metrics by dataset:")
    summary_lines.extend(metrics_by_dataset.to_string(index=False).splitlines())
    if skip_log:
        summary_lines.append("")
        summary_lines.append("Split notes:")
        summary_lines.extend(skip_log)
    save_text_summary(stats_dir / "resnet_summary.txt", summary_lines)


def run_within_resnet(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str], device: torch.device) -> None:
    log("\n================ WITHIN RESNET ================")
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
    skip_log = []

    for ds in sorted(within_ready):
        log(f"\n[INFO] Within dataset: {ds}")
        ds_plot = plots_dir / sanitize_name(ds)
        ds_stat = stats_dir / sanitize_name(ds)
        make_clean_dir(ds_plot)
        make_clean_dir(ds_stat)

        ds_dt = all_dt[all_dt["analysis_dataset"] == ds].copy()
        split_dt, _ = make_holdout_split(ds_dt, "analysis_dataset", TEST_FRAC, SEED)
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

        train_bal = cap_by_groups(train_dt, ["behavior"], CAP_N_WITHIN_TRAIN_BEHAVIOR, SEED)

        try:
            fit = fit_resnet_model(train_bal, test_dt, x_cols, y_cols, z_cols, device)
        except Exception as exc:
            msg = f"Skipped {ds}: {exc}"
            skip_log.append(msg)
            log_exception(msg, exc)
            continue

        metrics_dt = pd.DataFrame(fit["overall_metrics"]).copy()
        metrics_dt["analysis_dataset"] = ds
        metrics_all.append(metrics_dt)

        train_balance = train_bal.groupby("behavior").size().reset_index(name="N").sort_values("behavior")
        test_balance = test_dt.groupby("behavior").size().reset_index(name="N").sort_values("behavior")
        train_balance.to_csv(ds_stat / "train_balance.csv", index=False)
        test_balance.to_csv(ds_stat / "test_balance.csv", index=False)

        write_basic_outputs(
            fit["cm_dt"],
            fit["overall_metrics"],
            fit["behavior_metrics"],
            fit["history_df"],
            ds_plot,
            ds_stat,
            f"Within ResNet | {ds}",
        )

        summary_lines = [
            "Mode: Within_ResNet",
            f"Dataset: {ds}",
            f"Device: {device}",
            f"Train rows after balancing: {len(train_bal)}",
            f"Train rows used for fitting: {fit['train_main_n']}",
            f"Validation rows: {fit['val_n']}",
            f"Validation mode: disabled (aligned to HYDRA/MultiRocket)",
            f"Validation split vars: {', '.join(fit['val_split_vars']) if fit['val_split_vars'] else 'n/a'}",
            f"Test rows: {len(test_dt)}",
            "",
            "Overall metrics:",
        ]
        summary_lines.extend(pd.DataFrame(fit["overall_metrics"]).to_string(index=False).splitlines())
        save_text_summary(ds_stat / "resnet_summary.txt", summary_lines)

    if metrics_all:
        summary_dt = pd.concat(metrics_all, ignore_index=True).sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        summary_dt.to_csv(stats_dir / "within_summary_metrics.csv", index=False)
        save_summary_metric_plot(summary_dt, "analysis_dataset", plots_dir / "within_summary_metrics.png", "Within ResNet | Summary metrics")

        lines = ["Mode: Within_ResNet", "", "Summary metrics:"]
        lines.extend(summary_dt.to_string(index=False).splitlines())
        if skip_log:
            lines.append("")
            lines.append("Skip log:")
            lines.extend(skip_log)
        save_text_summary(stats_dir / "within_summary.txt", lines)


def run_inter_resnet(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str], device: torch.device) -> None:
    log("\n================ INTER RESNET ================")
    plots_dir, stats_dir = prepare_mode_dirs(INTER_ROOT)

    metrics_all = []
    skip_log = []

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

            train_bal = cap_by_groups(train_dt, ["behavior"], CAP_N_INTER_TRAIN_BEHAVIOR, SEED)

            try:
                fit = fit_resnet_model(train_bal, test_dt, x_cols, y_cols, z_cols, device)
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

            train_balance = train_bal.groupby("behavior").size().reset_index(name="N").sort_values("behavior")
            test_balance = test_dt.groupby("behavior").size().reset_index(name="N").sort_values("behavior")
            train_balance.to_csv(pair_stat / "train_balance.csv", index=False)
            test_balance.to_csv(pair_stat / "test_balance.csv", index=False)

            write_basic_outputs(
                fit["cm_dt"],
                fit["overall_metrics"],
                fit["behavior_metrics"],
                fit["history_df"],
                pair_plot,
                pair_stat,
                f"Inter ResNet | {pair_id}",
            )

            summary_lines = [
                "Mode: Inter_ResNet",
                f"Family: {family}",
                f"Train dataset: {train_ds}",
                f"Test dataset: {test_ds}",
                f"Device: {device}",
                f"Train rows after balancing: {len(train_bal)}",
                f"Train rows used for fitting: {fit['train_main_n']}",
                f"Validation rows: {fit['val_n']}",
                f"Validation mode: disabled (aligned to HYDRA/MultiRocket)",
                f"Validation split vars: {', '.join(fit['val_split_vars']) if fit['val_split_vars'] else 'n/a'}",
                f"Test rows: {len(test_dt)}",
                "",
                "Overall metrics:",
            ]
            summary_lines.extend(pd.DataFrame(fit["overall_metrics"]).to_string(index=False).splitlines())
            save_text_summary(pair_stat / "resnet_summary.txt", summary_lines)

    if metrics_all:
        summary_dt = pd.concat(metrics_all, ignore_index=True).sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        summary_dt.to_csv(stats_dir / "inter_summary_metrics.csv", index=False)
        save_summary_metric_plot(summary_dt, "pair_id", plots_dir / "inter_summary_metrics.png", "Inter ResNet | Summary metrics")

        lines = ["Mode: Inter_ResNet", "", "Summary metrics:"]
        lines.extend(summary_dt.to_string(index=False).splitlines())
        if skip_log:
            lines.append("")
            lines.append("Skip log:")
            lines.extend(skip_log)
        save_text_summary(stats_dir / "inter_summary.txt", lines)


def run_cross_resnet(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str], device: torch.device) -> None:
    log("\n================ CROSS RESNET ================")
    plots_dir, stats_dir = prepare_mode_dirs(CROSS_ROOT)

    metrics_all = []
    skip_log = []
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

        try:
            fit = fit_resnet_model(train_bal, test_dt, x_cols, y_cols, z_cols, device)
        except Exception as exc:
            msg = f"Skipped {test_ds}: {exc}"
            skip_log.append(msg)
            log_exception(msg, exc)
            continue

        metrics_dt = pd.DataFrame(fit["overall_metrics"]).copy()
        metrics_dt["test_dataset"] = test_ds
        metrics_all.append(metrics_dt)

        train_balance = (
            train_bal.groupby(["analysis_dataset", "behavior"], dropna=False)
            .size().reset_index(name="N")
            .sort_values(["analysis_dataset", "behavior"])
        )
        test_balance = test_dt.groupby("behavior").size().reset_index(name="N").sort_values("behavior")
        train_balance.to_csv(out_stat / "train_balance.csv", index=False)
        test_balance.to_csv(out_stat / "test_balance.csv", index=False)

        write_basic_outputs(
            fit["cm_dt"],
            fit["overall_metrics"],
            fit["behavior_metrics"],
            fit["history_df"],
            out_plot,
            out_stat,
            f"Cross ResNet | {test_ds}",
        )

        summary_lines = [
            "Mode: Cross_ResNet",
            f"Test dataset: {test_ds}",
            f"Device: {device}",
            f"Train rows after balancing: {len(train_bal)}",
            f"Train rows used for fitting: {fit['train_main_n']}",
            f"Validation rows: {fit['val_n']}",
            f"Validation mode: disabled (aligned to HYDRA/MultiRocket)",
            f"Validation split vars: {', '.join(fit['val_split_vars']) if fit['val_split_vars'] else 'n/a'}",
            f"Test rows: {len(test_dt)}",
            "",
            "Overall metrics:",
        ]
        summary_lines.extend(pd.DataFrame(fit["overall_metrics"]).to_string(index=False).splitlines())
        save_text_summary(out_stat / "resnet_summary.txt", summary_lines)

    if metrics_all:
        summary_dt = pd.concat(metrics_all, ignore_index=True).sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        summary_dt.to_csv(stats_dir / "cross_summary_metrics.csv", index=False)
        save_summary_metric_plot(summary_dt, "test_dataset", plots_dir / "cross_summary_metrics.png", "Cross ResNet | Summary metrics")

        lines = ["Mode: Cross_ResNet", "", "Summary metrics:"]
        lines.extend(summary_dt.to_string(index=False).splitlines())
        if skip_log:
            lines.append("")
            lines.append("Skip log:")
            lines.extend(skip_log)
        save_text_summary(stats_dir / "cross_summary.txt", lines)


def run_pairwise_resnet(all_dt: pd.DataFrame, x_cols: list[str], y_cols: list[str], z_cols: list[str], device: torch.device) -> None:
    log("\n================ PAIRWISE RESNET ================")
    plots_dir, stats_dir = prepare_mode_dirs(PAIRWISE_ROOT)

    metrics_all = []
    skip_log = []
    all_datasets = sorted(all_dt["analysis_dataset"].dropna().astype(str).unique().tolist())

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

            try:
                fit = fit_resnet_model(train_bal, test_dt, x_cols, y_cols, z_cols, device)
            except Exception as exc:
                msg = f"Skipped {pair_id}: {exc}"
                skip_log.append(msg)
                log_exception(msg, exc)
                continue

            metrics_dt = pd.DataFrame(fit["overall_metrics"]).copy()
            metrics_dt["pair_id"] = pair_id
            metrics_dt["train_dataset"] = train_ds
            metrics_dt["test_dataset"] = test_ds
            metrics_all.append(metrics_dt)

            train_balance = train_bal.groupby("behavior").size().reset_index(name="N").sort_values("behavior")
            test_balance = test_dt.groupby("behavior").size().reset_index(name="N").sort_values("behavior")
            train_balance.to_csv(pair_stat / "train_balance.csv", index=False)
            test_balance.to_csv(pair_stat / "test_balance.csv", index=False)

            write_basic_outputs(
                fit["cm_dt"],
                fit["overall_metrics"],
                fit["behavior_metrics"],
                fit["history_df"],
                pair_plot,
                pair_stat,
                f"Pairwise ResNet | {pair_id}",
            )

            summary_lines = [
                "Mode: Pairwise_ResNet",
                f"Train dataset: {train_ds}",
                f"Test dataset: {test_ds}",
                f"Device: {device}",
                f"Train rows after balancing: {len(train_bal)}",
                f"Train rows used for fitting: {fit['train_main_n']}",
                f"Validation rows: {fit['val_n']}",
                f"Validation mode: disabled (aligned to HYDRA/MultiRocket)",
                f"Validation split vars: {', '.join(fit['val_split_vars']) if fit['val_split_vars'] else 'n/a'}",
                f"Test rows: {len(test_dt)}",
                "",
                "Overall metrics:",
            ]
            summary_lines.extend(pd.DataFrame(fit["overall_metrics"]).to_string(index=False).splitlines())
            save_text_summary(pair_stat / "resnet_summary.txt", summary_lines)

    if metrics_all:
        summary_dt = pd.concat(metrics_all, ignore_index=True).sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        summary_dt.to_csv(stats_dir / "pairwise_summary_metrics.csv", index=False)
        save_summary_metric_plot(summary_dt, "pair_id", plots_dir / "pairwise_summary_metrics.png", "Pairwise ResNet | Summary metrics")

        lines = ["Mode: Pairwise_ResNet", "", "Summary metrics:"]
        lines.extend(summary_dt.to_string(index=False).splitlines())
        if skip_log:
            lines.append("")
            lines.append("Skip log:")
            lines.extend(skip_log)
        save_text_summary(stats_dir / "pairwise_summary.txt", lines)


# =========================================================
# Main
# =========================================================
def main() -> int:
    try:
        set_seed(SEED)
        device = get_device()
        log(f"[INFO] Device: {device}")
        log("[INFO] Loading raw files...")

        all_dt, read_log = load_raw_data()
        make_clean_dir(OUT_ROOT)
        for root in [GLOBAL_ROOT, WITHIN_ROOT, INTER_ROOT, CROSS_ROOT, PAIRWISE_ROOT]:
            make_clean_dir(root)
        save_text_summary(OUT_ROOT / "resnet_read_log.txt", read_log)

        log("[INFO] Cleaning raw table...")
        all_dt, x_cols, y_cols, z_cols = clean_raw_data(all_dt)

        counts_dt = (
            all_dt.groupby(["analysis_dataset", "species", "behavior"], dropna=False)
            .size().reset_index(name="N")
            .sort_values(["analysis_dataset", "species", "behavior"])
        )
        counts_dt.to_csv(OUT_ROOT / "resnet_counts_by_dataset_species_behavior.csv", index=False)
        log(f"[OK] CSV saved: {OUT_ROOT / 'resnet_counts_by_dataset_species_behavior.csv'}")

        lookup_dt = (
            all_dt[["analysis_dataset", "analysis_family", "species", "dataset_id", "dataset_folder", "top_folder"]]
            .drop_duplicates()
            .sort_values(["analysis_dataset", "species", "dataset_folder"])
        )
        lookup_dt.to_csv(OUT_ROOT / "resnet_analysis_dataset_lookup.csv", index=False)
        log(f"[OK] CSV saved: {OUT_ROOT / 'resnet_analysis_dataset_lookup.csv'}")

        save_duplicate_report(all_dt, OUT_ROOT / "resnet_duplicate_window_report.csv")

        mode_failures: list[str] = []
        mode_runs = [
            ("Global_ResNet", run_global_resnet),
            ("Within_ResNet", run_within_resnet),
            ("Inter_ResNet", run_inter_resnet),
            ("Cross_ResNet", run_cross_resnet),
            ("Pairwise_ResNet", run_pairwise_resnet),
        ]

        for mode_name, mode_func in mode_runs:
            try:
                mode_func(all_dt, x_cols, y_cols, z_cols, device)
            except Exception as exc:
                mode_failures.append(f"{mode_name}: {exc}")
                log_exception(f"{mode_name} failed", exc)

        log("\n========================================")
        if mode_failures:
            log("[ERROR] ResNet pipeline finished with errors.")
            for item in mode_failures:
                log(f"[ERROR]   - {item}")
            log(f"[DONE] Output root: {OUT_ROOT}")
            log(f"[DONE] remove_padded = {REMOVE_PADDED}")
            log("========================================")
            return 1

        log("[DONE] ResNet pipeline finished.")
        log(f"[DONE] Output root: {OUT_ROOT}")
        log(f"[DONE] remove_padded = {REMOVE_PADDED}")
        log("========================================")
        return 0

    except Exception as exc:
        log_exception("ResNet pipeline aborted", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
