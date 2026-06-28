from __future__ import annotations

from pathlib import Path
from typing import Any

from project_config import BIKES, LOCAL_DATA_ROLE, RAW_DIR

import numpy as np
import pandas as pd

from data_io import parse_file_key, read_sagemotion_csv
from signal_features import (
    acc_magnitude,
    aggregate_sensor_pair,
    crop_active_window,
    gyro_magnitude,
    sensor_features,
    window_starts,
)


def extract_window_features(labels: pd.DataFrame, window_s: float = 1.0, overlap: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract one row of features per active riding window."""
    label_map = {(row.group, row.bike, row.p_number): row for row in labels.itertuples(index=False)}
    feature_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []

    for path in sorted(RAW_DIR.rglob("*.csv")):
        if "Sagemotion Sensor Data" not in str(path.parent):
            continue
        key = parse_file_key(path)
        if key is None or key not in label_map:
            continue

        group, bike, p_number = key
        label = label_map[key]
        df = read_sagemotion_csv(path)
        start, end, fs, _energy = crop_active_window(df, float(label.table_ride_time_s))
        run_id = f"{group}_{bike}_{p_number}"

        crop_rows.append(
            {
                "run_id": run_id,
                "group": group,
                "bike": bike,
                "p_number": p_number,
                "source_file": str(path.relative_to(RAW_DIR)),
                "sample_rate_hz": fs,
                "raw_n_rows": int(len(df)),
                "active_start_idx": start,
                "active_end_idx": end,
                "active_n_rows": end - start,
                "active_duration_s": float((end - start) / fs) if np.isfinite(fs) and fs > 0 else np.nan,
                "table_ride_time_s": float(label.table_ride_time_s),
            }
        )

        active = df.iloc[start:end].reset_index(drop=True)
        window_n = max(12, int(round(window_s * fs))) if np.isfinite(fs) and fs > 0 else len(active)
        step_n = max(1, int(round(window_n * (1.0 - overlap))))

        for window_idx, w_start in enumerate(window_starts(len(active), window_n, step_n)):
            w_end = min(len(active), w_start + window_n)
            part = active.iloc[w_start:w_end].reset_index(drop=True)

            # Feature extraction is based on magnitudes, not raw axes. This makes
            # the features less sensitive to sensor orientation.
            acc1 = sensor_features(acc_magnitude(part, "1"), fs, "acc1")
            acc2 = sensor_features(acc_magnitude(part, "2"), fs, "acc2")
            gyro1 = sensor_features(gyro_magnitude(part, "1"), fs, "gyro1")
            gyro2 = sensor_features(gyro_magnitude(part, "2"), fs, "gyro2")
            feature_values = aggregate_sensor_pair([acc1, acc2])
            feature_values.update(aggregate_sensor_pair([gyro1, gyro2]))

            feature_rows.append(
                {
                    "group": group,
                    "group_id": int(group[1:]),
                    "run_id": run_id,
                    "window_id": f"{run_id}_w{window_idx:03d}",
                    "bike": bike,
                    "p_number": p_number,
                    "p_order": int(p_number[1:]),
                    "dataset_role": LOCAL_DATA_ROLE,
                    "pressure_bar": float(label.pressure_bar),
                    "rider_weight_kg": float(label.rider_weight_kg),
                    "table_ride_time_s": float(label.table_ride_time_s),
                    "sample_rate_hz": fs,
                    "window_start_idx": int(start + w_start),
                    "window_end_idx": int(start + w_end),
                    "window_s": float((w_end - w_start) / fs) if np.isfinite(fs) and fs > 0 else np.nan,
                    **feature_values,
                }
            )
    return pd.DataFrame(feature_rows), pd.DataFrame(crop_rows)


def signal_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return numeric signal columns while excluding metadata and labels."""
    meta_cols = {
        "group_id",
        "p_order",
        "pressure_bar",
        "rider_weight_kg",
        "table_ride_time_s",
        "sample_rate_hz",
        "window_start_idx",
        "window_end_idx",
        "window_s",
    }
    return [
        col
        for col in features.columns
        if pd.api.types.is_numeric_dtype(features[col])
        and col not in meta_cols
        and not col.startswith("bike_")
    ]


def make_feature_matrix(features: pd.DataFrame, include_weight: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Build model-ready X from signal features plus bike one-hot encoding."""
    signal_cols = signal_feature_columns(features)
    bike_dummies = pd.get_dummies(features["bike"], prefix="bike", dtype=float)
    for bike in BIKES:
        col = f"bike_{bike}"
        if col not in bike_dummies:
            bike_dummies[col] = 0.0

    frames = [features[signal_cols].copy(), bike_dummies[[f"bike_{bike}" for bike in BIKES]]]
    if include_weight:
        frames.append(features[["rider_weight_kg"]].copy())
    X = pd.concat(frames, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X.astype(float), list(X.columns)


def build_run_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate selected window features to one row per run for EDA."""
    key_cols = ["group", "run_id", "bike", "p_number", "p_order", "dataset_role", "pressure_bar", "rider_weight_kg"]
    numeric_cols = signal_feature_columns(features)
    selected = [col for col in numeric_cols if col.endswith("_rms_mean") or col.endswith("_energy_per_s_mean")]
    if not selected:
        selected = numeric_cols[:12]
    return features.groupby(key_cols, as_index=False).agg(
        n_windows=("window_id", "count"),
        sample_rate_hz=("sample_rate_hz", "median"),
        window_s=("window_s", "median"),
        **{f"{col}_median": (col, "median") for col in selected},
    )


def write_feature_tables(labels: pd.DataFrame, inventory: pd.DataFrame, features: pd.DataFrame, crops: pd.DataFrame, run_summary: pd.DataFrame, feature_names: list[str], table_dir: Path) -> None:
    labels.to_csv(table_dir / "training_pool_labels.csv", index=False)
    inventory.to_csv(table_dir / "training_pool_raw_file_inventory.csv", index=False)
    features.to_csv(table_dir / "training_pool_window_features.csv", index=False)
    crops.to_csv(table_dir / "training_pool_active_window_summary.csv", index=False)
    run_summary.to_csv(table_dir / "training_pool_run_feature_summary.csv", index=False)
    pd.DataFrame({"feature_name": feature_names}).to_csv(table_dir / "training_pool_candidate_input_features.csv", index=False)
