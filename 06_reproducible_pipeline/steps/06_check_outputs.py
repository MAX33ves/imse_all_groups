from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from project_config import FIG_EDA_DIR, FIG_MODEL_DIR, MODEL_DIR, REPORT_DIR, TABLE_DIR
from step_logging import assert_exists, run_logged_action


def main() -> None:
    def action(emit):
        required = [
            TABLE_DIR / "training_pool_labels.csv",
            TABLE_DIR / "training_pool_raw_file_inventory.csv",
            TABLE_DIR / "training_pool_window_features.csv",
            TABLE_DIR / "training_pool_run_feature_summary.csv",
            TABLE_DIR / "training_pool_candidate_feature_target_correlations.csv",
            TABLE_DIR / "training_pool_final_input_feature_rationale.csv",
            TABLE_DIR / "training_pool_final_input_correlation_matrix.csv",
            TABLE_DIR / "training_pool_ffnn_model_comparison.csv",
            TABLE_DIR / "training_pool_ffnn_selected_cv_predictions.csv",
            TABLE_DIR / "training_pool_ffnn_final_model_training_fit_predictions.csv",
            MODEL_DIR / "training_pool_ffnn_selection_summary.json",
            MODEL_DIR / "training_pool_ffnn_final_model_summary.json",
            MODEL_DIR / "training_pool_ffnn_final_model.pkl",
            REPORT_DIR / "training_pool_data_processing_report_bilingual.md",
            REPORT_DIR / "training_pool_ffnn_cv_model_report_bilingual.md",
            REPORT_DIR / "training_pool_ffnn_final_model_report_bilingual.md",
            FIG_EDA_DIR / "training_pool_03_pca_all_local_data.png",
            FIG_EDA_DIR / "training_pool_05_feature_target_correlation.png",
            FIG_EDA_DIR / "training_pool_06_final_input_correlation_matrix.png",
            FIG_MODEL_DIR / "training_pool_ffnn_02_cv_predicted_vs_actual.png",
        ]
        for path in required:
            assert_exists(path, str(path))
            emit(f"OK {path.name}")

        labels = pd.read_csv(TABLE_DIR / "training_pool_labels.csv")
        features = pd.read_csv(TABLE_DIR / "training_pool_window_features.csv", usecols=["run_id", "group"])
        selected_cv = pd.read_csv(TABLE_DIR / "training_pool_ffnn_selected_cv_predictions.csv")
        summary = json.loads((MODEL_DIR / "training_pool_ffnn_selection_summary.json").read_text(encoding="utf-8"))

        emit("")
        emit(f"Training-pool runs : {labels['run_id'].nunique()}")
        emit(f"Groups             : {labels['group'].nunique()}")
        emit(f"Window rows        : {len(features)}")
        emit(f"Selected model     : {summary['selected_model']}")
        emit(f"CV MAE             : {summary['selected_cv_metrics']['mae_bar']:.3f} bar")
        emit(f"CV RMSE            : {summary['selected_cv_metrics']['rmse_bar']:.3f} bar")
        emit(f"CV predictions     : {selected_cv['run_id'].nunique()} runs")

        if labels["run_id"].nunique() != 72:
            raise RuntimeError("Expected 72 local training-pool runs.")
        if selected_cv["run_id"].nunique() != 72:
            raise RuntimeError("Selected CV predictions should cover all 72 runs once.")
        if not bool(summary.get("selected_uses_rider_weight")):
            raise RuntimeError("Selected model should include rider_weight_kg.")

    run_logged_action("06_check_outputs", action)


if __name__ == "__main__":
    main()
