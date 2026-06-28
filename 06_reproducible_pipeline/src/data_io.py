from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from project_config import RAW_DIR

import numpy as np
import pandas as pd


def parse_file_key(path: Path) -> tuple[str, str, str] | None:
    """Parse group, bike, and P number from a raw file path."""
    parts = [part.upper() for part in path.parts]
    group = next((part for part in parts if re.fullmatch(r"G0?[1-6]", part)), None)
    if group is None:
        return None
    group = f"G{int(group[1:]):02d}"

    name = path.stem.upper()
    if "MTB" in name:
        bike = "MTB"
    elif "ISY" in name:
        bike = "ISY"
    elif "FAT" in name:
        bike = "FAT"
    else:
        return None

    match = re.search(r"P([1-4])$", name)
    if match is None:
        return None
    return group, bike, f"P{match.group(1)}"


def read_sagemotion_csv(path: Path) -> pd.DataFrame:
    """Read a Sagemotion CSV and coerce numeric sensor columns."""
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        header = next(csv.reader(handle))
    data_cols = [col for col in header if not col.strip().startswith("{")]
    df = pd.read_csv(path, usecols=list(range(len(data_cols))), encoding="utf-8-sig")
    df.columns = data_cols
    for col in data_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def median_sample_rate(times_ms: pd.Series) -> float:
    """Estimate sample rate from the median positive time step."""
    values = times_ms.dropna().to_numpy(dtype=float)
    if values.size < 3:
        return float("nan")
    diffs = np.diff(values) / 1000.0
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan")
    return 1.0 / float(np.median(diffs))


def duration_s(times_ms: pd.Series) -> float:
    values = times_ms.dropna().to_numpy(dtype=float)
    if values.size < 2:
        return float("nan")
    return float((values[-1] - values[0]) / 1000.0)


def build_inventory(labels: pd.DataFrame) -> pd.DataFrame:
    """Create a file-level quality inventory for CSV and XLS raw files."""
    label_map = {(row.group, row.bike, row.p_number): row for row in labels.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for path in sorted(RAW_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".xls"}:
            continue
        key = parse_file_key(path)
        row: dict[str, Any] = {
            "file": str(path.relative_to(RAW_DIR)),
            "source": "sagemotion" if path.suffix.lower() == ".csv" else "phyphox",
            "size_bytes": path.stat().st_size,
            "parse_status": "unparsed_filename" if key is None else "ok",
        }
        if key is not None:
            group, bike, p_number = key
            label = label_map.get((group, bike, p_number))
            row.update({"group": group, "bike": bike, "p_number": p_number, "run_id": f"{group}_{bike}_{p_number}", "has_label": label is not None})
            if label is not None:
                row.update(
                    {
                        "pressure_bar": float(label.pressure_bar),
                        "dataset_role": label.dataset_role,
                        "rider_weight_kg": float(label.rider_weight_kg),
                        "table_ride_time_s": float(label.table_ride_time_s),
                    }
                )
        try:
            if path.suffix.lower() == ".csv":
                df = read_sagemotion_csv(path)
                row.update(
                    {
                        "n_rows": int(len(df)),
                        "n_columns": int(len(df.columns)),
                        "duration_s": duration_s(df["Sampletime_1"]),
                        "sample_rate_hz": median_sample_rate(df["Sampletime_1"]),
                        "null_cells": int(df.isna().sum().sum()),
                    }
                )
        except Exception as exc:
            row["parse_status"] = f"read_error:{type(exc).__name__}"
        rows.append(row)
    return pd.DataFrame(rows)
