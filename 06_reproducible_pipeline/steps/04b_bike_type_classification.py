from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from bike_type_modeling import (
    add_bike_selection_columns,
    bike_candidate_by_name,
    bike_metrics,
    build_bike_confusion_rows,
    build_bike_ensemble_candidates,
    build_bike_feature_configs,
    build_bike_feature_spaces,
    build_bike_stage1_candidates,
    evaluate_bike_group_cv,
    fit_final_bike_model,
    save_final_bike_model,
    select_bike_model,
    top_bike_stage1_names_for_ensembles,
)
from project_config import FIG_MODEL_DIR, MODEL_DIR, REPORT_DIR, SUSPENSION_BY_BIKE, SUSPENSION_TYPES, TABLE_DIR, VALIDATION_METHOD, ensure_dirs
from step_logging import run_logged_action


def write_suspension_report(summary: dict, selected_cv_preds: pd.DataFrame, comparison: pd.DataFrame, report_path: Path) -> None:
    selected_row = summary["selected_model_row"]
    by_class = selected_cv_preds.groupby("actual_suspension_type", as_index=False).agg(
        n_runs=("run_id", "count"),
        accuracy=("is_correct", "mean"),
        mean_confidence=("pred_confidence", "mean"),
    )
    top = comparison.sort_values(["cv_selection_score", "model_complexity", "model_name"]).head(10)
    lines = [
        "# Suspension-Type Classifier Report / 悬挂类型分类模型报告",
        "",
        "## Purpose / 这个模型是给什么用的",
        "",
        "This model predicts the suspension or damping category from Sagemotion signal features.",
        "这个模型用 Sagemotion 信号特征预测悬挂/阻尼类型。",
        "",
        "The supervised labels are derived from the course table that maps each bike type to a suspension type.",
        "监督标签来自课程表中“单车类型 -> 悬挂类型”的对应关系。",
        "",
        "| Bike context / 单车背景 | Suspension label / 悬挂标签 |",
        "|---|---|",
        *[f"| {bike} | {suspension} |" for bike, suspension in SUSPENSION_BY_BIKE.items()],
        "",
        "## Leakage-Safe Inputs / 防止信息泄漏的输入规则",
        "",
        "The classifier deliberately does not use the following fields as inputs:",
        "分类器刻意不使用以下字段作为输入：",
        "",
        "- `bike`: this would directly reveal the mapped suspension label. / 它会直接泄漏映射后的悬挂标签。",
        "- `suspension_type`: this is the label being predicted. / 这是要预测的标签。",
        "- `pressure_bar`: this would leak the tire-pressure label and is not known at prediction time. / 这会泄漏胎压标签，而且真实预测时也不知道。",
        "- `p_number`, `group`, `run_id`, and file name: these are metadata and split/audit fields. / 这些是元数据、划分字段或审计字段。",
        "- `rider_weight_kg`: this can encode rider/group context rather than bike structure. / 这可能编码骑手或 group 上下文，而不是单车结构。",
        "",
        "Allowed inputs are Sagemotion acceleration and gyroscope signal features only.",
        "允许使用的输入只有 Sagemotion 加速度和陀螺仪信号特征。",
        "",
        "## Validation Design / 验证设计",
        "",
        f"- Validation method: `{VALIDATION_METHOD}` / 验证方法：`{VALIDATION_METHOD}`",
        "- Each fold holds out one complete group and trains on the other five groups. / 每一折留出一个完整 group，用其余五个 group 训练。",
        "- Window-level probabilities are averaged to make one run-level suspension prediction. / 窗口级概率先取平均，再得到一个 run-level 悬挂类型预测。",
        "",
        "## Selected Model / 选中模型",
        "",
        f"- Model: `{summary['selected_model']}` / 模型：`{summary['selected_model']}`",
        f"- Feature space: `{selected_row['feature_space']}` / 特征空间：`{selected_row['feature_space']}`",
        f"- Pre-PCA inputs: {int(selected_row['n_features_in'])} / PCA 前输入数：{int(selected_row['n_features_in'])}",
        f"- PCA components: {int(selected_row['pca_components'])} / PCA 维度：{int(selected_row['pca_components'])}",
        f"- Hidden layers: {selected_row['hidden_layers']} / 隐藏层：{selected_row['hidden_layers']}",
        f"- Seeds: {int(selected_row['n_seeds'])} / 随机种子成员数：{int(selected_row['n_seeds'])}",
        "",
        "## CV Performance / CV 表现",
        "",
        "| Metric / 指标 | Value / 数值 |",
        "|---|---:|",
        f"| Run-level accuracy | {summary['selected_cv_metrics']['run_accuracy']:.3f} |",
        f"| Macro-F1 | {summary['selected_cv_metrics']['macro_f1']:.3f} |",
        f"| Minimum group accuracy | {summary['selected_cv_metrics']['min_group_accuracy']:.3f} |",
        f"| Mean confidence | {summary['selected_cv_metrics']['mean_confidence']:.3f} |",
        "",
        "## Performance By Suspension Type / 按悬挂类型看的表现",
        "",
        "| Suspension type / 悬挂类型 | n runs | accuracy | mean confidence |",
        "|---|---:|---:|---:|",
    ]
    for row in by_class.itertuples(index=False):
        lines.append(f"| {row.actual_suspension_type} | {row.n_runs} | {row.accuracy:.3f} | {row.mean_confidence:.3f} |")
    lines.extend(
        [
            "",
            "## Top Candidate Models / 候选模型前 10",
            "",
            "| Model / 模型 | Feature space | CV accuracy | CV macro-F1 | score |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in top.itertuples(index=False):
        lines.append(f"| `{row.model_name}` | {row.feature_space} | {row.cv_run_accuracy:.3f} | {row.cv_macro_f1:.3f} | {row.cv_selection_score:.3f} |")
    lines.extend(
        [
            "",
            "## How This Connects To The Pressure Model / 和胎压模型的关系",
            "",
            "This classifier addresses the teacher's suspension/damping requirement. The pressure regressor remains a separate model because tire pressure is a continuous regression target.",
            "这个分类器对应老师提出的悬挂/阻尼类型任务。胎压回归模型仍然是单独模型，因为胎压是连续回归目标。",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    def action(emit):
        ensure_dirs(TABLE_DIR, MODEL_DIR, REPORT_DIR, FIG_MODEL_DIR)
        features = pd.read_csv(TABLE_DIR / "training_pool_window_features.csv")
        spaces = build_bike_feature_spaces(features)
        configs = build_bike_feature_configs(spaces, n_train_rows=len(features))
        stage1_candidates = build_bike_stage1_candidates(configs)
        emit(f"Suspension stage-1 candidates: {len(stage1_candidates)}")

        stage1_cv, stage1_preds = evaluate_bike_group_cv(features, spaces, stage1_candidates, list(SUSPENSION_TYPES))
        stage1_cv = add_bike_selection_columns(stage1_cv)
        top_names = top_bike_stage1_names_for_ensembles(stage1_cv)
        top_candidates = [candidate for candidate in stage1_candidates if candidate.name in set(top_names)]
        ensemble_candidates = build_bike_ensemble_candidates(top_candidates)
        emit(f"Suspension ensemble candidates: {len(ensemble_candidates)}")

        ensemble_cv, ensemble_preds = evaluate_bike_group_cv(features, spaces, ensemble_candidates, list(SUSPENSION_TYPES))
        comparison = pd.concat(
            [
                stage1_cv.drop(columns=[col for col in ["cv_selection_score", "model_complexity", "near_best_cutoff", "is_near_best"] if col in stage1_cv.columns]),
                ensemble_cv,
            ],
            ignore_index=True,
        )
        comparison = add_bike_selection_columns(comparison).sort_values(["cv_selection_score", "model_complexity", "model_name"]).reset_index(drop=True)
        cv_predictions = pd.concat([stage1_preds, ensemble_preds], ignore_index=True)
        selected_model = select_bike_model(comparison)
        selected_row = comparison[comparison["model_name"].eq(selected_model)].iloc[0]
        selected_cv_preds = cv_predictions[cv_predictions["model_name"].eq(selected_model)].copy()
        selected_metrics = bike_metrics(selected_cv_preds, list(SUSPENSION_TYPES))
        confusion_rows = build_bike_confusion_rows(selected_cv_preds, selected_model, list(SUSPENSION_TYPES))

        comparison.to_csv(TABLE_DIR / "training_pool_suspension_model_comparison.csv", index=False)
        cv_predictions.to_csv(TABLE_DIR / "training_pool_suspension_group_cv_predictions.csv", index=False)
        selected_cv_preds.to_csv(TABLE_DIR / "training_pool_suspension_selected_cv_predictions.csv", index=False)
        confusion_rows.to_csv(TABLE_DIR / "training_pool_suspension_cv_confusion.csv", index=False)

        all_candidates = stage1_candidates + ensemble_candidates
        selected_candidate = bike_candidate_by_name(all_candidates, selected_model)
        pipelines, training_fit = fit_final_bike_model(selected_candidate, spaces, features, list(SUSPENSION_TYPES))
        training_fit.to_csv(TABLE_DIR / "training_pool_suspension_final_model_training_fit_predictions.csv", index=False)
        save_final_bike_model(MODEL_DIR / "training_pool_suspension_final_model.pkl", selected_candidate, pipelines, list(SUSPENSION_TYPES))
        fit_metrics = bike_metrics(training_fit, list(SUSPENSION_TYPES))

        summary = {
            "task": "suspension_type_classification",
            "target": "suspension_type",
            "class_labels": list(SUSPENSION_TYPES),
            "source_mapping": dict(SUSPENSION_BY_BIKE),
            "validation_method": VALIDATION_METHOD,
            "forbidden_inputs": ["bike", "suspension_type", "pressure_bar", "p_number", "group", "run_id", "file name", "rider_weight_kg"],
            "allowed_input_scope": "Sagemotion acceleration and gyroscope signal features only.",
            "selected_model": selected_model,
            "selected_model_row": selected_row.to_dict(),
            "selected_cv_metrics": selected_metrics,
            "training_pool_fit_metrics": fit_metrics,
            "n_labeled_runs": int(features["run_id"].nunique()),
            "n_windows": int(len(features)),
            "model_file": str(MODEL_DIR / "training_pool_suspension_final_model.pkl"),
        }
        (MODEL_DIR / "training_pool_suspension_selection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (MODEL_DIR / "training_pool_suspension_final_model_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_suspension_report(summary, selected_cv_preds, comparison, REPORT_DIR / "training_pool_suspension_classifier_report_bilingual.md")

        emit(f"Selected suspension model : {selected_model}")
        emit(f"CV accuracy                : {selected_metrics['run_accuracy']:.3f}")
        emit(f"CV macro-F1                : {selected_metrics['macro_f1']:.3f}")

    run_logged_action("04b_suspension_classification", action)


if __name__ == "__main__":
    main()
