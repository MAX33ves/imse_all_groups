from __future__ import annotations

import subprocess
import sys
from pathlib import Path


STEP_DIR = Path(__file__).resolve().parent
STEPS = [
    "00_check_environment.py",
    "01_build_labels_inventory.py",
    "02_extract_window_features.py",
    "03_make_eda_figures.py",
    "04_group_cv_model_selection.py",
    "05_train_final_model.py",
    "06_check_outputs.py",
    "07_teacher_review_checks.py",
]


def main() -> None:
    for step in STEPS:
        print("\n" + "=" * 72)
        print(f"RUNNING {step}")
        print("=" * 72)
        result = subprocess.run([sys.executable, str(STEP_DIR / step)], cwd=str(STEP_DIR.parent))
        if result.returncode != 0:
            raise SystemExit(f"{step} failed with exit code {result.returncode}")
    print("\nAll training-pool steps completed.")


if __name__ == "__main__":
    main()
