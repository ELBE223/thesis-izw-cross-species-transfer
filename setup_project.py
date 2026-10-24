# setup_project.py
#
# Purpose:
# - Creates the folder structure of the paper project, one folder per dataset
#   in raw and processed data, plus starter files (README, .gitignore,
#   dataset registry, decision log, requirements).
# - Creates a Python virtual environment (.venv) and installs requirements.txt.
# - Safe to run again: existing folders and files are never overwritten.
#
# Usage (macOS Terminal; the venv uses the Python that runs this script):
#   python3.11 setup_project.py                       # default project folder
#   python3.11 setup_project.py "/path/to/project"    # other project folder
#   python3.11 setup_project.py --no-venv             # folders and files only
#   python3.11 setup_project.py --no-install          # venv without packages
#
# Afterwards: source "/Volumes/Z Slim/2026_10_24_Paper/.venv/bin/activate"

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path

DEFAULT_ROOT = Path("/Volumes/Z Slim/2026_10_24_Paper")

# Same dataset names as in the thesis, so results stay comparable.
DATASETS = {
    "Bison_dataset_1": ("Bison", "Bison bonasus"),
    "Cattle_dataset_1": ("Cattle", "Bos taurus"),
    "Dog_dataset_1": ("Dog", "Canis lupus familiaris"),
    "Dog_dataset_2": ("Dog", "Canis lupus familiaris"),
    "Fox_dataset_1": ("Fox", "Vulpes vulpes"),
    "Fox_dataset_2": ("Fox", "Vulpes vulpes"),
    "Giraffe_dataset_1": ("Giraffe", "Giraffa camelopardalis"),
    "Hedgehog_dataset_1": ("Hedgehog", "Erinaceus europaeus"),
    "Horse_dataset_1": ("Horse", "Equus ferus przewalskii"),
    "Horse_dataset_2": ("Horse", "Equus caballus"),
    "Raccoon_dataset_1": ("Raccoon", "Procyon lotor"),
    "Raccoon_dataset_2": ("Raccoon", "Procyon lotor"),
}

DOC_DIR = "00_documentation"
RAW_DIR = "01_raw_data"
PROCESSED_DIR = "02_processed_data"
CODE_DIR = "code"
VENV_DIR = ".venv"

FOLDERS = [
    DOC_DIR,
    f"{DOC_DIR}/dataset_reports",
    RAW_DIR,
    PROCESSED_DIR,
    CODE_DIR,
] + [f"{RAW_DIR}/{ds}" for ds in DATASETS] + [f"{PROCESSED_DIR}/{ds}" for ds in DATASETS]

# Unpinned on purpose; the installed versions are written to requirements-lock.txt.
REQUIREMENTS = [
    "numpy",
    "pandas",
    "pyarrow",    # parquet
    "openpyxl",   # Excel files
    "scipy",      # MATLAB .mat files
]


# =========================================================
# Starter files
# =========================================================
README = f"""# Cross-species transfer – paper project

## Structure

- `{DOC_DIR}/` – documentation: dataset registry, dataset reports, decision log
- `{RAW_DIR}/<dataset>/` – original data as received; read only, never modified
- `{PROCESSED_DIR}/<dataset>/` – harmonized data, created by the code only
- `{CODE_DIR}/` – all code
- `{VENV_DIR}/` – Python environment (`source {VENV_DIR}/bin/activate`)

## Rules

- Raw data are never edited; every change happens in code.
- `{RAW_DIR}/` and `{PROCESSED_DIR}/` are not in git: the raw data cannot be shared,
  and the processed data can be recreated from the code.
- Every processing decision is recorded in `{DOC_DIR}/decisions.md`.
- Dataset names match the thesis (`<Species>_dataset_<n>`).
"""

DECISIONS = """# Decision log

One entry per decision: what was decided, why, and which datasets are affected.

## Template

### YYYY-MM-DD – Short title
- Datasets: all / <dataset>
- Decision:
- Reason:
- Alternatives considered:
"""

GITIGNORE = f"""# Data: raw data cannot be shared, processed data are recreated by the code
{RAW_DIR}/
{PROCESSED_DIR}/

# Python
{VENV_DIR}/
__pycache__/
*.pyc
.ipynb_checkpoints/

# macOS
.DS_Store
._*
"""


def dataset_registry() -> str:
    header = "dataset_id,species,scientific_name,source_reference,doi,license,publicly_available,notes"
    rows = [f"{ds},{common},{latin},,,,," for ds, (common, latin) in DATASETS.items()]
    return "\n".join([header] + rows) + "\n"


# =========================================================
# Helpers
# =========================================================
def log(msg: str) -> None:
    print(msg, flush=True)


def write_if_missing(path: Path, content: str) -> None:
    if path.exists():
        log(f"[SKIP] {path.name} exists")
        return
    path.write_text(content, encoding="utf-8")
    log(f"[OK] {path.name} created")


def venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def create_venv(venv_dir: Path) -> None:
    if (venv_dir / "pyvenv.cfg").exists():
        log(f"[SKIP] venv exists: {venv_dir}")
        return
    log(f"[INFO] Creating venv with Python {sys.version.split()[0]} ...")
    try:
        venv.EnvBuilder(with_pip=True, symlinks=sys.platform != "win32").create(venv_dir)
    except OSError as exc:
        # External drives formatted as exFAT do not support symlinks.
        log(f"[WARN] Symlinks not supported ({exc}); creating the venv with copies.")
        shutil.rmtree(venv_dir, ignore_errors=True)
        venv.EnvBuilder(with_pip=True, symlinks=False, clear=True).create(venv_dir)
    log(f"[OK] venv created: {venv_dir}")


def install_requirements(root: Path, venv_dir: Path) -> None:
    python = str(venv_python(venv_dir))
    log("[INFO] Installing requirements ...")
    subprocess.run([python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([python, "-m", "pip", "install", "-r", str(root / "requirements.txt")], check=True)
    frozen = subprocess.run([python, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    (root / "requirements-lock.txt").write_text(frozen, encoding="utf-8")
    log("[OK] Installed versions written to requirements-lock.txt")


# =========================================================
# Main
# =========================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Create the paper project structure and venv.")
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT, help=f"Project folder (default: {DEFAULT_ROOT})")
    parser.add_argument("--no-venv", action="store_true", help="Do not create a venv")
    parser.add_argument("--no-install", action="store_true", help="Create the venv without installing packages")
    args = parser.parse_args()

    root = args.root.expanduser()

    # Stop if the external drive is not mounted, instead of creating the path elsewhere.
    if not root.parent.is_dir():
        log(f"[ERROR] Parent folder not found: {root.parent} (is the drive connected?)")
        return 1

    if sys.version_info[:2] != (3, 11):
        log(f"[WARN] Running Python {sys.version.split()[0]}; the thesis analyses used Python 3.11.")

    log(f"[INFO] Project folder: {root}")
    for folder in FOLDERS:
        (root / folder).mkdir(parents=True, exist_ok=True)
    log(f"[OK] {len(FOLDERS)} folders present ({len(DATASETS)} datasets)")

    write_if_missing(root / "README.md", README)
    write_if_missing(root / ".gitignore", GITIGNORE)
    write_if_missing(root / "requirements.txt", "\n".join(REQUIREMENTS) + "\n")
    write_if_missing(root / DOC_DIR / "decisions.md", DECISIONS)
    write_if_missing(root / DOC_DIR / "datasets.csv", dataset_registry())

    if not args.no_venv:
        venv_dir = root / VENV_DIR
        create_venv(venv_dir)
        # Also on re-runs, so a failed or new installation is completed.
        if not args.no_install:
            install_requirements(root, venv_dir)

    log("\n[OK] Done. Next: copy each raw dataset into its folder in " + RAW_DIR)
    if not args.no_venv:
        log(f'     Activate the venv with: source "{root / VENV_DIR / "bin" / "activate"}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
