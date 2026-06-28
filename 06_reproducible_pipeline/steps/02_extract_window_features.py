from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from data_io import build_inventory
from feature_pipeline import build_run_summary, extract_window_features, make_feature_matrix, write_feature_tables
from labels import labels_frame
from project_config import MODEL_DIR, TABLE_DIR, ensure_dirs
from step_logging import run_logged_action


def main() -> None:
    def action(emit):
        ensure_dirs(TABLE_DIR, MODEL_DIR)

        # 1) Read labels from previous step if available; otherwise rebuild them.
        labels_path = TABLE_DIR / "training_pool_labels.csv"
        labels = pd.read_csv(labels_path) if labels_path.exists() else labels_frame()
        inventory_path = TABLE_DIR / "training_pool_raw_file_inventory.csv"
        inventory = pd.read_csv(inventory_path) if inventory_path.exists() else build_inventory(labels)

        # 2) Crop active riding windows and extract one row per 1-second window.
        features, crops = extract_window_features(labels, window_s=1.0, overlap=0.5)

        # 3) Build model input matrix only to record the candidate input columns.
        # Rider weight is now part of the model input pool.
        _, feature_names = make_feature_matrix(features, include_weight=True)
        run_summary = build_run_summary(features)

        # 4) Persist all intermediate tables. These are the main audit trail.
        write_feature_tables(labels, inventory, features, crops, run_summary, feature_names, TABLE_DIR)

        summary = {
            "n_labeled_runs": int(labels["run_id"].nunique()),
            "n_window_rows": int(len(features)),
            "n_active_window_rows": int(len(crops)),
            "n_input_features": int(len(feature_names)),
            "window_seconds": 1.0,
            "overlap": 0.5,
            "cleaning_steps": [
                "convert sensor columns to numeric",
                "replace inf with NaN",
                "interpolate missing signal values",
                "fill remaining missing values with median",
                "use acceleration magnitude and gyro magnitude",
                "band-pass filter when sample rate allows",
            ],
        }
        (MODEL_DIR / "training_pool_feature_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        emit(f"Window feature rows: {summary['n_window_rows']}")
        emit(f"Input features     : {summary['n_input_features']}")
        emit(f"Run summaries      : {run_summary['run_id'].nunique()}")

    run_logged_action("02_extract_window_features", action)


if __name__ == "__main__":
    main()
