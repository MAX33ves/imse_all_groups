from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from modeling import (
    add_selection_columns,
    build_confusion_rows,
    build_ensemble_candidates,
    build_feature_configs,
    build_feature_spaces,
    build_reference_candidates,
    build_stage1_candidates,
    evaluate_group_cv,
    metrics_from_run_predictions,
    select_model,
    top_stage1_names_for_ensembles,
)
from plotting import plot_abs_error_by_bike, plot_confusion, plot_cv_error_heatmap, plot_cv_predictions, plot_cv_selection, use_chart_theme
from project_config import FIG_MODEL_DIR, MODEL_DIR, REPORT_DIR, TABLE_DIR, VALIDATION_METHOD, ensure_dirs
from step_logging import run_logged_action


def write_model_selection_report(summary: dict, selected_cv_preds: pd.DataFrame, comparison: pd.DataFrame, report_path: Path) -> None:
    top = comparison.sort_values(["cv_selection_score", "cv_run_mae_bar"]).head(10)
    selected_row = summary["selected_model_row"]
    best_without_weight = comparison[
        comparison["candidate_type"].eq("ffnn") & ~comparison["feature_space"].fillna("").str.contains("weight", case=False)
    ].sort_values(["cv_selection_score", "cv_run_mae_bar"]).head(1)
    best_with_weight = comparison[
        comparison["candidate_type"].eq("ffnn") & comparison["feature_space"].fillna("").str.contains("weight", case=False)
    ].sort_values(["cv_selection_score", "cv_run_mae_bar"]).head(1)
    lines = [
        "# FFNN Group-Level Cross-Validation And Model Selection / FFNN 组级交叉验证与模型选择",
        "",
        "## Why There Is No Local P3/P4 Final Test / 为什么现在没有本地 P3/P4 final test",
        "",
        "The instructor still has hidden test data, so all locally observed P1-P4 runs can be used for training. To still estimate local generalization, this project uses group-level cross-validation: each fold holds out one complete group as validation and trains on the other five groups.",
        "现在已确认老师手里还有隐藏测试数据，所以本地观测到的 P1-P4 全部都可以用于训练。为了仍然估计泛化能力，我们使用组级交叉验证：每次留出一个完整小组作为 validation，其余 5 个组训练。",
        "",
        "## Validation Split / 验证划分",
        "",
        "- Fold 1: train = G02-G06, validation = G01",
        "- Fold 2: train = G01,G03-G06, validation = G02",
        "- The pattern continues until every group has been validated once. / 以此类推，直到每个组都被验证一次。",
        "- Each validation fold contains 12 runs from that group: FAT/ISY/MTB with P1-P4. / 每个验证 fold 包含该组的 12 个 run，也就是 FAT/ISY/MTB 的 P1-P4。",
        "",
        "## Model Inputs / 模型输入",
        "",
        "- Main inputs: Sagemotion signal window features + bike type one-hot + rider weight. / 主要输入：Sagemotion 信号窗口特征 + bike type one-hot + rider weight。",
        "- Control comparison: feature spaces without rider weight are still compared, but the final model must include rider weight. / 对照输入：不含 rider weight 的特征空间仍参与比较，但最终模型必须使用 rider weight。",
        "- Forbidden inputs: `group`, `p_number`, `run_id`, and the true pressure label. / 禁止输入：`group`、`p_number`、`run_id`、真实胎压标签。",
        "",
        "## Regression Task And Auxiliary Classification Metrics / 回归任务和辅助分类指标",
        "",
        "- The main task is regression: the model outputs one continuous value, `pressure_bar`. / 主任务是回归：模型输出 1 个连续值 `pressure_bar`。",
        "- For course-required classification-style metrics, each run-level continuous prediction is mapped to the nearest valid pressure level for that bike. / 课程任务也要求 confusion matrix / F-score 这类分类指标，所以本项目把每个 run 的连续预测映射到该 bike 的最近可行胎压等级。",
        "- MAE/RMSE are the main generalization metrics; confusion matrix and macro-F1 are auxiliary interpretation metrics. / MAE/RMSE 是主要泛化指标；混淆矩阵和 macro-F1 是辅助解释指标。",
        "",
        "## Selected Model / 选中模型",
        "",
        f"- Model: `{summary['selected_model']}` / 模型：`{summary['selected_model']}`",
        f"- CV MAE: {summary['selected_cv_metrics']['mae_bar']:.3f} bar",
        f"- CV RMSE: {summary['selected_cv_metrics']['rmse_bar']:.3f} bar",
        f"- CV bias: {summary['selected_cv_metrics']['bias_bar']:.3f} bar",
        f"- nearest-level accuracy：{summary['selected_cv_metrics']['nearest_level_accuracy']:.3f}",
        f"- macro-F1：{summary['selected_cv_metrics']['nearest_level_macro_f1']:.3f}",
        "",
        "## Network Structure / 网络结构记录",
        "",
        f"- Pre-PCA inputs: {int(selected_row['n_features_in'])}. / PCA 前输入变量数：{int(selected_row['n_features_in'])}。",
        f"- PCA components after scaling: {int(selected_row['pca_components'])}. / 标准化后 PCA 维度：{int(selected_row['pca_components'])}。",
        f"- FFNN hidden layers: {selected_row['hidden_layers']}. / FFNN 隐藏层：{selected_row['hidden_layers']}。",
        "- Output neuron: 1, predicting continuous `pressure_bar`. / 输出神经元：1，输出连续胎压 `pressure_bar`。",
        f"- Activation: {selected_row['activation']}. / 激活函数：{selected_row['activation']}。",
        f"- L2 regularization alpha: {selected_row['alpha']}. / L2 正则 alpha：{selected_row['alpha']}。",
        "- sklearn implementation uses `MLPRegressor(solver='lbfgs', max_iter=3000, max_fun=40000)`, so the report records optimizer limits rather than a fixed epoch count. / sklearn 实现使用 `MLPRegressor(solver='lbfgs', max_iter=3000, max_fun=40000)`，因此这里记录优化器迭代上限，而不是固定 epoch 数。",
        "",
        "## Model-Selection Caveat / 模型选择 caveat",
        "",
        "- The same leave-one-group-out CV is used for candidate comparison and selected-model metric reporting, so these CV metrics are local post-selection estimates, not instructor hidden-test results. / 同一个 leave-one-group-out CV 同时用于候选模型比较和选中模型指标报告，所以这里的 CV 指标是本地模型选择后的泛化估计，不是老师隐藏测试集结果。",
        "- A strictly unbiased nested CV would need more independent groups. With only 6 groups, the project reports candidate comparisons and held-out group errors transparently. / 严格无偏的 nested CV 需要更多独立 group；当前只有 6 个 group，所以采用透明报告候选模型和 held-out group 误差的方式控制风险。",
        "- The best FFNN without rider weight is slightly better, but the selected rider-weight model is within the 5% near-best range and satisfies the current rule. / 最优 FFNN 不含体重，含体重模型在 near-best 5% 范围内并满足当前规则。",
        "",
        "## Rider-Weight Sensitivity / 体重输入敏感性",
        "",
        "| role / 角色 | model / 模型 | CV score | CV MAE | CV RMSE | nearest-level accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for role, frame in [("best without rider weight", best_without_weight), ("best with rider weight", best_with_weight)]:
        if not frame.empty:
            row = frame.iloc[0]
            lines.append(f"| {role} | `{row.model_name}` | {row.cv_selection_score:.3f} | {row.cv_run_mae_bar:.3f} | {row.cv_run_rmse_bar:.3f} | {row.cv_nearest_level_accuracy:.3f} |")
    lines.extend(
        [
            "",
        "## Top 10 Candidate Models / 候选模型前 10",
        "",
        "| model / 模型 | type | CV score | CV MAE | CV RMSE | accuracy |",
        "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in top.itertuples(index=False):
        lines.append(f"| `{row.model_name}` | {row.candidate_type} | {row.cv_selection_score:.3f} | {row.cv_run_mae_bar:.3f} | {row.cv_run_rmse_bar:.3f} | {row.cv_nearest_level_accuracy:.3f} |")
    lines.extend(
        [
            "",
            "## Selected-Model Validation Predictions / 选中模型的验证预测",
            "",
            "| run | fold | actual | predicted | abs error |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in selected_cv_preds.sort_values(["fold", "bike", "p_number"]).itertuples(index=False):
        lines.append(f"| {row.run_id} | {row.fold} | {row.actual_pressure_bar:.3f} | {row.pred_pressure_bar:.3f} | {row.abs_error_bar:.3f} |")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    def action(emit):
        ensure_dirs(TABLE_DIR, MODEL_DIR, REPORT_DIR, FIG_MODEL_DIR)
        use_chart_theme()

        labels = pd.read_csv(TABLE_DIR / "training_pool_labels.csv")
        features = pd.read_csv(TABLE_DIR / "training_pool_window_features.csv")

        # 1) Build feature spaces and candidate FFNNs.
        _, feature_names, spaces = build_feature_spaces(features)
        configs = build_feature_configs(spaces, n_train_rows=len(features))
        stage1_candidates = build_stage1_candidates(configs)
        reference_candidates = build_reference_candidates()
        emit(f"Stage 1 FFNN candidates: {len(stage1_candidates)}")
        emit(f"Reference baselines      : {len(reference_candidates)}")

        # 2) First model-selection pass with leave-one-group-out CV.
        stage1_cv, stage1_preds = evaluate_group_cv(features, labels, spaces, reference_candidates + stage1_candidates)
        stage1_cv = add_selection_columns(stage1_cv)

        # 3) Ensemble only the top few FFNNs, still judged by the same CV method.
        top_ffnn_names = top_stage1_names_for_ensembles(stage1_cv)
        top_candidates = [candidate for candidate in stage1_candidates if candidate.name in top_ffnn_names]
        ensemble_candidates = build_ensemble_candidates(top_candidates)
        emit(f"Ensemble FFNN candidates: {len(ensemble_candidates)}")
        ensemble_cv, ensemble_preds = evaluate_group_cv(features, labels, spaces, ensemble_candidates)

        # 4) Select one FFNN using near-best CV plus simpler-network tie-break.
        # The final selected model must include rider_weight_kg as an input.
        comparison = pd.concat(
            [
                stage1_cv.drop(columns=[col for col in ["cv_selection_score", "model_complexity", "near_best_ffnn_cutoff", "is_near_best_ffnn"] if col in stage1_cv.columns]),
                ensemble_cv,
            ],
            ignore_index=True,
        )
        comparison = add_selection_columns(comparison).sort_values(["cv_selection_score", "cv_run_mae_bar", "model_name"]).reset_index(drop=True)
        cv_predictions = pd.concat([stage1_preds, ensemble_preds], ignore_index=True)
        selected_model = select_model(comparison, require_rider_weight=True)
        selected_row = comparison[comparison["model_name"].eq(selected_model)].iloc[0]
        selected_uses_rider_weight = bool("weight" in str(selected_row["feature_space"]).lower())
        if not selected_uses_rider_weight:
            raise RuntimeError("Selected model does not include rider_weight_kg.")
        selected_cv_preds = cv_predictions[cv_predictions["model_name"].eq(selected_model)].copy()
        selected_metrics = metrics_from_run_predictions(selected_cv_preds, labels)
        confusion_rows = build_confusion_rows(selected_cv_preds, labels, selected_model)

        # 5) Persist selection artifacts.
        comparison.to_csv(TABLE_DIR / "training_pool_ffnn_model_comparison.csv", index=False)
        cv_predictions.to_csv(TABLE_DIR / "training_pool_ffnn_group_cv_predictions.csv", index=False)
        selected_cv_preds.to_csv(TABLE_DIR / "training_pool_ffnn_selected_cv_predictions.csv", index=False)
        confusion_rows.to_csv(TABLE_DIR / "training_pool_ffnn_cv_nearest_level_confusion.csv", index=False)

        plot_cv_selection(comparison, selected_model)
        plot_cv_predictions(selected_cv_preds, selected_model)
        plot_cv_error_heatmap(selected_cv_preds)
        plot_abs_error_by_bike(selected_cv_preds)
        plot_confusion(confusion_rows, selected_model)

        summary = {
            "data_scope": "All observed G01-G06 P1-P4 runs",
            "target": "pressure_bar regression",
            "validation_method": VALIDATION_METHOD,
            "selected_model": selected_model,
            "selected_uses_rider_weight": selected_uses_rider_weight,
            "n_labeled_runs": int(labels["run_id"].nunique()),
            "n_training_pool_runs": int(labels["run_id"].nunique()),
            "n_cv_folds": int(labels["group"].nunique()),
            "n_windows": int(len(features)),
            "n_base_input_features": int(len(feature_names)),
            "selected_model_row": selected_row.to_dict(),
            "selected_cv_metrics": {
                "score": float(selected_row["cv_selection_score"]),
                "mae_bar": float(selected_metrics["run_mae_bar"]),
                "rmse_bar": float(selected_metrics["run_rmse_bar"]),
                "bias_bar": float(selected_metrics["run_bias_bar"]),
                "max_abs_error_bar": float(selected_metrics["run_max_abs_error_bar"]),
                "nearest_level_accuracy": float(selected_metrics["nearest_level_accuracy"]),
                "nearest_level_macro_f1": float(selected_metrics["nearest_level_macro_f1"]),
            },
            "selection_rule": "Use leave-one-group-out CV on all local P1-P4 runs; the final selected FFNN must include rider_weight_kg, and among eligible models prefer a simpler model within 5% of the best FFNN CV score.",
        }
        (MODEL_DIR / "training_pool_ffnn_selection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_model_selection_report(summary, selected_cv_preds, comparison, REPORT_DIR / "training_pool_ffnn_cv_model_report_bilingual.md")

        emit(f"Selected model     : {selected_model}")
        emit(f"CV MAE             : {selected_metrics['run_mae_bar']:.3f} bar")
        emit(f"CV RMSE            : {selected_metrics['run_rmse_bar']:.3f} bar")

    run_logged_action("04_group_cv_model_selection", action)


if __name__ == "__main__":
    main()
