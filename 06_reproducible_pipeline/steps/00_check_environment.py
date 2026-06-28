from __future__ import annotations

import importlib
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_config import GROUPS, PIPELINE_ROOT, RAW_DIR
from step_logging import assert_exists, run_logged_action


def main() -> None:
    def action(emit):
        # 1) Check that all raw group folders are present.
        assert_exists(RAW_DIR, "Measurement_Campaign raw data directory")
        for group in GROUPS:
            group_dir = RAW_DIR / group
            assert_exists(group_dir, f"{group} directory")
            assert_exists(group_dir / "Sagemotion Sensor Data", f"{group} Sagemotion directory")
            assert_exists(group_dir / "PhyPhox Data", f"{group} PhyPhox directory")

        # 2) Count raw files. We expect 6 groups * 3 bikes * 4 runs = 72 files per source.
        csv_count = len(list(RAW_DIR.rglob("*.csv")))
        xls_count = len(list(RAW_DIR.rglob("*.xls")))
        emit(f"Raw data directory : {RAW_DIR}")
        emit(f"Groups checked     : {', '.join(GROUPS)}")
        emit(f"Sagemotion CSV     : {csv_count}")
        emit(f"PhyPhox XLS        : {xls_count}")
        if csv_count != 72:
            raise RuntimeError(f"Expected 72 Sagemotion CSV files, found {csv_count}.")
        if xls_count != 72:
            raise RuntimeError(f"Expected 72 PhyPhox XLS files, found {xls_count}.")

        # 3) Check project modules and Python packages.
        required_modules = [
            "project_config.py",
            "labels.py",
            "data_io.py",
            "signal_features.py",
            "feature_pipeline.py",
            "modeling.py",
            "plotting.py",
        ]
        for module in required_modules:
            assert_exists(SRC_DIR / module, f"Source module {module}")
        emit(f"Source modules OK  : {len(required_modules)}")

        missing = []
        for package in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn"]:
            try:
                importlib.import_module(package)
            except Exception:
                missing.append(package)
        if missing:
            raise RuntimeError(f"Missing Python packages: {', '.join(missing)}")
        emit("Python packages OK")
        emit(f"Pipeline root      : {PIPELINE_ROOT}")

    run_logged_action("00_check_environment", action)


if __name__ == "__main__":
    main()
