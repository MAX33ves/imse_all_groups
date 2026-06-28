from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from modeling import (
    build_ensemble_candidates,
    build_feature_configs,
    build_feature_spaces,
    build_reference_candidates,
    candidate_uses_rider_weight,
    build_stage1_candidates,
    candidate_by_name,
    fit_final_model,
    metrics_from_run_predictions,
    save_final_model,
    top_stage1_names_for_ensembles,
)
from project_config import MODEL_DIR, REPORT_DIR, TABLE_DIR, ensure_dirs
from step_logging import run_logged_action


def write_final_model_report(selection_summary: dict, fit_metrics: dict, report_path: Path) -> None:
    selected_row = selection_summary["selected_model_row"]
    lines = [
        "# Final FFNN Training Model / 最终 FFNN 训练模型",
        "",
        "## Purpose / 这个模型是给什么用的",
        "",
        "The instructor still has hidden test data, so the final model is retrained on all local G01-G06, P1-P4 data. This final model no longer holds out a local final test set; local generalization is estimated by the previous leave-one-group-out CV step.",
        "老师手里还有隐藏测试数据，所以最终模型使用本地全部 G01-G06、P1-P4 数据重新训练。这个模型不再留出本地 final test；本地泛化能力由上一步 leave-one-group-out CV 估计。",
        "",
        "## Final Training Data / 最终训练数据",
        "",
        "| Item / 项目 | Value / 数值 |",
        "|---|---:|",
        f"| Local training runs / 本地训练 run | {selection_summary['n_training_pool_runs']} |",
        f"| Window samples / 窗口样本 | {selection_summary['n_windows']} |",
        "| Target / 目标 | Continuous tire pressure `pressure_bar` / 连续胎压 `pressure_bar` |",
        "",
        "## Selected Structure / 选中结构",
        "",
        f"- Model: `{selection_summary['selected_model']}` / 模型：`{selection_summary['selected_model']}`",
        f"- Uses rider weight: {selection_summary['selected_uses_rider_weight']} / 是否使用 rider weight：{selection_summary['selected_uses_rider_weight']}",
        f"- Selection rule: {selection_summary['selection_rule']} / 选择规则：在全部本地 P1-P4 run 上使用 leave-one-group-out CV；最终选中的 FFNN 必须包含 `rider_weight_kg`；在合格模型中优先选择距离最佳 FFNN CV score 5% 以内、更简单的模型。",
        f"- Pre-PCA inputs: {int(selected_row['n_features_in'])} / PCA 前输入变量数：{int(selected_row['n_features_in'])}",
        f"- PCA components: {int(selected_row['pca_components'])} / PCA 维度：{int(selected_row['pca_components'])}",
        f"- Hidden layers: {selected_row['hidden_layers']} / 隐藏层：{selected_row['hidden_layers']}",
        "- Output neuron: 1, predicting continuous `pressure_bar` / 输出神经元：1，输出连续胎压 `pressure_bar`",
        f"- Activation: {selected_row['activation']} / 激活函数：{selected_row['activation']}",
        f"- L2 regularization alpha: {selected_row['alpha']} / L2 正则 alpha：{selected_row['alpha']}",
        "- Training implementation: `MLPRegressor(solver='lbfgs', max_iter=3000, max_fun=40000)`. Therefore, training is recorded by optimizer iterations/function-call limits rather than a fixed epoch count. / 训练实现：`MLPRegressor(solver='lbfgs', max_iter=3000, max_fun=40000)`。因此记录优化器迭代/函数调用上限，而不是固定 epoch 数。",
        "",
        "## Local Validation Performance From CV / 本地验证效果（来自上一步 CV）",
        "",
        "| Metric / 指标 | Value / 数值 |",
        "|---|---:|",
        f"| CV MAE | {selection_summary['selected_cv_metrics']['mae_bar']:.3f} bar |",
        f"| CV RMSE | {selection_summary['selected_cv_metrics']['rmse_bar']:.3f} bar |",
        f"| CV nearest-level accuracy | {selection_summary['selected_cv_metrics']['nearest_level_accuracy']:.3f} |",
        f"| CV macro-F1 | {selection_summary['selected_cv_metrics']['nearest_level_macro_f1']:.3f} |",
        "",
        "## Training-Pool Fit After Full Retraining / 全量训练后的训练集拟合效果",
        "",
        "| Metric / 指标 | Value / 数值 |",
        "|---|---:|",
        f"| Training-pool fit MAE | {fit_metrics['run_mae_bar']:.3f} bar |",
        f"| Training-pool fit RMSE | {fit_metrics['run_rmse_bar']:.3f} bar |",
        "",
        "Training-pool fit is a fit-to-training-data result, not a generalization estimate. Reports should cite the CV metrics when discussing local generalization.",
        "Training-pool fit 是训练集拟合效果，不是泛化效果；报告里讨论本地泛化时更应该引用 CV 指标。",
        "",
        "## Hidden Test Set Note / 隐藏测试集说明",
        "",
        "- The instructor hidden test set is not in the current workspace, so hidden-test MAE/RMSE cannot be reported yet. / 老师隐藏测试集不在当前 workspace 中，所以现在不能报告 hidden-test MAE/RMSE。",
        "- The saved `.pkl` is the final local model for later hidden-test processing with the same raw-signal processing, feature extraction, scaling, PCA, and FFNN prediction steps. / 当前 `.pkl` 保存的是最终本地模型，用于后续对隐藏数据做同样的原始信号处理、特征提取、标准化/PCA/FFNN 预测。",
        "- If a report or presentation mentions generalization, cite leave-one-group-out CV; true external generalization requires the instructor hidden-test result. / 如果报告或答辩提到 generalization，应引用 leave-one-group-out CV；真正外部泛化需要等老师隐藏测试结果。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    def action(emit):
        ensure_dirs(TABLE_DIR, MODEL_DIR, REPORT_DIR)

        labels = pd.read_csv(TABLE_DIR / "training_pool_labels.csv")
        features = pd.read_csv(TABLE_DIR / "training_pool_window_features.csv")
        selection_summary = json.loads((MODEL_DIR / "training_pool_ffnn_selection_summary.json").read_text(encoding="utf-8"))
        selected_model = selection_summary["selected_model"]

        # Rebuild the same candidate list used during model selection.
        _, _, spaces = build_feature_spaces(features)
        configs = build_feature_configs(spaces, n_train_rows=len(features))
        stage1_candidates = build_stage1_candidates(configs)
        reference_candidates = build_reference_candidates()
        comparison = pd.read_csv(TABLE_DIR / "training_pool_ffnn_model_comparison.csv")
        top_ffnn_names = top_stage1_names_for_ensembles(comparison)
        top_candidates = [candidate for candidate in stage1_candidates if candidate.name in set(top_ffnn_names)]
        ensemble_candidates = build_ensemble_candidates(top_candidates)
        all_candidates = reference_candidates + stage1_candidates + ensemble_candidates
        selected_candidate = candidate_by_name(all_candidates, selected_model)
        if not candidate_uses_rider_weight(selected_candidate):
            raise RuntimeError("The selected final model must include rider_weight_kg. Re-run step 04 first.")

        # Fit the selected architecture on all local data for hidden testing.
        pipelines, training_fit = fit_final_model(selected_candidate, spaces, features)
        training_fit.to_csv(TABLE_DIR / "training_pool_ffnn_final_model_training_fit_predictions.csv", index=False)
        save_final_model(MODEL_DIR / "training_pool_ffnn_final_model.pkl", selected_candidate, pipelines)

        fit_metrics = metrics_from_run_predictions(training_fit, labels)
        final_summary = {
            "selected_model": selected_model,
            "purpose": "Final local model trained on all observed P1-P4 data for teacher hidden testing.",
            "selected_uses_rider_weight": True,
            "input_feature_note": "Selected FFNN feature columns include rider_weight_kg.",
            "n_training_pool_runs": int(labels["run_id"].nunique()),
            "n_training_pool_windows": int(len(features)),
            "training_pool_fit_metrics": fit_metrics,
            "cv_metrics_should_be_reported_for_generalization": selection_summary["selected_cv_metrics"],
            "model_file": str(MODEL_DIR / "training_pool_ffnn_final_model.pkl"),
        }
        (MODEL_DIR / "training_pool_ffnn_final_model_summary.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_final_model_report(selection_summary, fit_metrics, REPORT_DIR / "training_pool_ffnn_final_model_report_bilingual.md")

        emit(f"Final model saved   : {MODEL_DIR / 'training_pool_ffnn_final_model.pkl'}")
        emit(f"Training fit MAE    : {fit_metrics['run_mae_bar']:.3f} bar")
        emit(f"CV MAE to report    : {selection_summary['selected_cv_metrics']['mae_bar']:.3f} bar")

    run_logged_action("05_train_final_model", action)


if __name__ == "__main__":
    main()
