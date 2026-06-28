from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_io import build_inventory
from labels import labels_frame
from project_config import MODEL_DIR, TABLE_DIR, ensure_dirs
from step_logging import run_logged_action


def main() -> None:
    def action(emit):
        ensure_dirs(TABLE_DIR, MODEL_DIR)

        # 1) Build labels from Measurement Details, not from P number alone.
        labels = labels_frame()

        # 2) Build file inventory and lightweight data-quality checks.
        inventory = build_inventory(labels)

        # 3) Persist outputs for later steps.
        labels.to_csv(TABLE_DIR / "training_pool_labels.csv", index=False)
        inventory.to_csv(TABLE_DIR / "training_pool_raw_file_inventory.csv", index=False)

        source_counts = inventory.groupby(["source", "parse_status"]).size().reset_index(name="count")
        summary = {
            "scope": "All observed G01-G06 P1-P4 runs are local training data.",
            "n_labeled_runs": int(labels["run_id"].nunique()),
            "n_groups": int(labels["group"].nunique()),
            "n_bikes": int(labels["bike"].nunique()),
            "n_raw_files": int(len(inventory)),
            "source_status_counts": source_counts.to_dict(orient="records"),
        }
        (MODEL_DIR / "training_pool_data_inventory_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        emit(f"Labeled runs       : {summary['n_labeled_runs']}")
        emit(f"Raw files          : {summary['n_raw_files']}")
        for row in source_counts.itertuples(index=False):
            emit(f"{row.source:10} {row.parse_status:18} {row.count}")

    run_logged_action("01_build_labels_inventory", action)


if __name__ == "__main__":
    main()
