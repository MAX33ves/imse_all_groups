from __future__ import annotations

import os
import sys
from pathlib import Path


# Path setup is centralized so every step writes to the same project folders.
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
RAW_DIR = WORKSPACE_ROOT / "Measurement_Campaign"
VENDOR_DIR = WORKSPACE_ROOT / "_analysis" / "vendor"

if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if hasattr(os, "add_dll_directory") and (VENDOR_DIR / "bin").exists():
    os.add_dll_directory(str(VENDOR_DIR / "bin"))

OUT_DIR = PROJECT_ROOT / "03_outputs"
TABLE_DIR = OUT_DIR / "tables"
MODEL_DIR = OUT_DIR / "models"
FIG_EDA_DIR = OUT_DIR / "figures_training_pool_eda"
FIG_MODEL_DIR = OUT_DIR / "figures_training_pool_ffnn"
REPORT_DIR = PROJECT_ROOT / "04_report"
LOG_DIR = PIPELINE_ROOT / "logs"

BIKES = ("FAT", "ISY", "MTB")
GROUPS = tuple(f"G{i:02d}" for i in range(1, 7))

# New project rule:
# All observed P1-P4 runs are local training data. The teacher's hidden dataset is
# the real external test set, so we only estimate local performance by CV.
LOCAL_DATA_ROLE = "training_pool"
VALIDATION_METHOD = "leave_one_group_out_cv"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
BLUE = {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"}
GOLD = {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"}
ORANGE = {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"}
OLIVE = {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"}
PINK = {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"}


# Labels transcribed from Measurement Details.pdf.
# Tuple format: (P number, pressure bar, rider weight kg, ride time seconds).
GROUP_DATA: dict[str, dict[str, list[tuple[str, float, float, float]]]] = {
    "G01": {
        "FAT": [("P1", 0.4, 91, 7.5), ("P2", 0.8, 91, 7.4), ("P3", 0.8, 100, 5.74), ("P4", 0.6, 100, 6.31)],
        "ISY": [("P1", 1.0, 91, 8.25), ("P2", 3.0, 91, 7.37), ("P3", 3.0, 100, 6.19), ("P4", 2.0, 100, 5.57)],
        "MTB": [("P1", 2.0, 91, 7.32), ("P2", 3.0, 91, 6.88), ("P3", 3.0, 100, 6.17), ("P4", 1.0, 100, 4.96)],
    },
    "G02": {
        "FAT": [("P1", 0.4, 63, 4.75), ("P2", 0.8, 63, 5.8), ("P3", 0.8, 106, 6.04), ("P4", 0.6, 106, 6.41)],
        "ISY": [("P1", 1.0, 63, 6.7), ("P2", 3.0, 63, 6.1), ("P3", 3.0, 106, 5.31), ("P4", 2.0, 106, 5.64)],
        "MTB": [("P1", 2.0, 63, 5.4), ("P2", 3.0, 63, 5.7), ("P3", 3.0, 106, 4.9), ("P4", 1.0, 106, 5.15)],
    },
    "G03": {
        "FAT": [("P1", 0.4, 120, 8.13), ("P2", 0.8, 120, 7.32), ("P3", 0.8, 109, 6.0), ("P4", 0.6, 109, 6.45)],
        "ISY": [("P1", 1.0, 120, 7.66), ("P2", 3.0, 120, 7.92), ("P3", 3.0, 109, 5.3), ("P4", 2.0, 109, 5.36)],
        "MTB": [("P1", 2.0, 120, 7.01), ("P2", 3.0, 120, 6.89), ("P3", 3.0, 109, 4.5), ("P4", 1.0, 109, 6.0)],
    },
    "G04": {
        "FAT": [("P1", 0.4, 73, 6.59), ("P2", 0.8, 73, 7.0), ("P3", 0.8, 75, 6.34), ("P4", 0.6, 75, 7.23)],
        "ISY": [("P1", 1.0, 73, 7.08), ("P2", 3.0, 73, 6.61), ("P3", 3.0, 75, 6.03), ("P4", 2.0, 75, 5.67)],
        "MTB": [("P1", 2.0, 73, 6.15), ("P2", 3.0, 73, 6.91), ("P3", 3.0, 75, 5.68), ("P4", 1.0, 75, 6.01)],
    },
    "G05": {
        "FAT": [("P1", 0.4, 63, 5.24), ("P2", 0.8, 63, 6.6), ("P3", 0.8, 79, 6.05), ("P4", 0.6, 79, 6.48)],
        "ISY": [("P1", 1.0, 63, 7.07), ("P2", 3.0, 63, 6.8), ("P3", 3.0, 79, 6.03), ("P4", 2.0, 79, 4.87)],
        "MTB": [("P1", 2.0, 63, 6.21), ("P2", 3.0, 63, 6.35), ("P3", 3.0, 79, 5.48), ("P4", 1.0, 79, 5.87)],
    },
    "G06": {
        "FAT": [("P1", 0.4, 67, 7.52), ("P2", 0.8, 67, 7.57), ("P3", 0.8, 82, 6.96), ("P4", 0.6, 82, 6.48)],
        "ISY": [("P1", 1.0, 67, 7.6), ("P2", 3.0, 67, 7.62), ("P3", 3.0, 82, 6.38), ("P4", 2.0, 82, 5.84)],
        "MTB": [("P1", 2.0, 67, 5.38), ("P2", 3.0, 67, 6.91), ("P3", 3.0, 82, 5.67), ("P4", 1.0, 82, 5.41)],
    },
}


def ensure_dirs(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
