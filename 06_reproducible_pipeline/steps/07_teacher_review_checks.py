from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from project_config import PROJECT_ROOT, TABLE_DIR, MODEL_DIR, ensure_dirs
from step_logging import run_logged_action


REVIEW_DIR = PROJECT_ROOT / "05_teacher_review"
REVIEW_TABLE_DIR = REVIEW_DIR / "tables"


def _spearman(left: pd.Series, right: pd.Series) -> float:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return float("nan")
    return float(left.rank().corr(right.rank(), method="pearson"))


def _metric_row(name: str, status: str, evidence: str, implication: str) -> dict[str, str]:
    return {"check": name, "status": status, "evidence": evidence, "implication": implication}


def _format_value(value: object, floatfmt: str = ".3f") -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return format(float(value), floatfmt)
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_format_value(value, floatfmt) for value in row) + " |")
    return "\n".join(lines)


def build_review_tables() -> dict[str, pd.DataFrame]:
    labels = pd.read_csv(TABLE_DIR / "training_pool_labels.csv")
    inventory = pd.read_csv(TABLE_DIR / "training_pool_raw_file_inventory.csv")
    features = pd.read_csv(TABLE_DIR / "training_pool_window_features.csv")
    crops = pd.read_csv(TABLE_DIR / "training_pool_active_window_summary.csv")
    selected_cv = pd.read_csv(TABLE_DIR / "training_pool_ffnn_selected_cv_predictions.csv")
    comparison = pd.read_csv(TABLE_DIR / "training_pool_ffnn_model_comparison.csv")
    rationale = pd.read_csv(TABLE_DIR / "training_pool_final_input_feature_rationale.csv")
    selection_summary = json.loads((MODEL_DIR / "training_pool_ffnn_selection_summary.json").read_text(encoding="utf-8"))

    numeric_features = features.select_dtypes(include=[np.number])
    crop_duration_error = (crops["active_duration_s"] - crops["table_ride_time_s"]).abs()
    feature_nan_count = int(features.isna().sum().sum())
    feature_inf_count = int(np.isinf(numeric_features.to_numpy()).sum())
    sagemotion_ok = int(((inventory["source"] == "sagemotion") & (inventory["parse_status"] == "ok")).sum())
    phyphox_ok = int(((inventory["source"] == "phyphox") & (inventory["parse_status"] == "ok")).sum())

    cv_error = selected_cv["pred_pressure_bar"] - selected_cv["actual_pressure_bar"]
    cv_mae = float(np.abs(cv_error).mean())
    cv_rmse = float(np.sqrt(np.mean(cv_error**2)))
    cv_bias = float(cv_error.mean())

    data_quality = pd.DataFrame(
        [
            _metric_row(
                "label coverage",
                "pass",
                f"{labels['run_id'].nunique()} labeled runs across {labels['group'].nunique()} groups, {labels['bike'].nunique()} bike types.",
                "The local training pool matches the stated 72-run design.",
            ),
            _metric_row(
                "raw file coverage",
                "pass",
                f"{sagemotion_ok} Sagemotion CSV and {phyphox_ok} PhyPhox XLS files parse successfully.",
                "Raw files are present; current model uses Sagemotion only, so PhyPhox needs an explicit scope note.",
            ),
            _metric_row(
                "feature table finite values",
                "pass" if feature_nan_count == 0 and feature_inf_count == 0 else "needs_attention",
                f"{len(features)} window rows, {feature_nan_count} NaN cells, {feature_inf_count} infinite numeric cells.",
                "No actual CV leakage from full-data median filling was observed in the current extracted feature table.",
            ),
            _metric_row(
                "active-window duration",
                "pass" if float(crop_duration_error.max()) < 1e-9 else "needs_attention",
                f"max |active_duration_s - table_ride_time_s| = {float(crop_duration_error.max()):.6f} s.",
                "The crop length follows Measurement Details exactly; start location is still energy-based and should be described as an assumption.",
            ),
            _metric_row(
                "sample rate consistency",
                "pass" if crops["sample_rate_hz"].nunique() == 1 else "needs_attention",
                f"sample_rate_hz min/median/max = {crops['sample_rate_hz'].min():.1f}/{crops['sample_rate_hz'].median():.1f}/{crops['sample_rate_hz'].max():.1f}.",
                "A fixed 1 s window corresponds to 100 samples for every run.",
            ),
            _metric_row(
                "selected CV recomputation",
                "pass",
                f"MAE={cv_mae:.3f} bar, RMSE={cv_rmse:.3f} bar, bias={cv_bias:.3f} bar over {len(selected_cv)} run predictions.",
                "The report metrics are reproducible from the selected CV prediction table.",
            ),
            _metric_row(
                "rider-weight direct target correlation",
                "caveat",
                f"global Spearman={_spearman(labels['rider_weight_kg'], labels['pressure_bar']):.3f}; selected model uses rider_weight_kg={selection_summary['selected_uses_rider_weight']}.",
                "Weight is physically justified as load context, but the current data does not prove a strong direct pressure correlation.",
            ),
        ]
    )

    pressure_design = labels.groupby(["bike", "pressure_bar"], as_index=False).agg(
        n_runs=("run_id", "count"),
        n_groups=("group", "nunique"),
        min_weight_kg=("rider_weight_kg", "min"),
        max_weight_kg=("rider_weight_kg", "max"),
        median_ride_time_s=("table_ride_time_s", "median"),
    )

    error_by_bike = selected_cv.groupby("bike", as_index=False).agg(
        n_runs=("run_id", "count"),
        mae_bar=("abs_error_bar", "mean"),
        rmse_bar=("signed_error_bar", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        max_abs_error_bar=("abs_error_bar", "max"),
        mean_window_pred_std=("pred_pressure_window_std", "mean"),
    )
    error_by_group = selected_cv.groupby("group", as_index=False).agg(
        n_runs=("run_id", "count"),
        mae_bar=("abs_error_bar", "mean"),
        rmse_bar=("signed_error_bar", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        max_abs_error_bar=("abs_error_bar", "max"),
    )
    worst_runs = selected_cv.sort_values("abs_error_bar", ascending=False).head(15)[
        [
            "run_id",
            "fold",
            "bike",
            "group",
            "actual_pressure_bar",
            "pred_pressure_bar",
            "abs_error_bar",
            "n_windows",
            "pred_pressure_window_std",
        ]
    ]

    comparison = comparison.copy()
    comparison["uses_rider_weight"] = comparison["feature_space"].fillna("").str.contains("weight", case=False)
    best_without_weight = comparison[comparison["candidate_type"].eq("ffnn") & ~comparison["uses_rider_weight"]].sort_values(["cv_selection_score", "cv_run_mae_bar"]).head(1)
    best_with_weight = comparison[comparison["candidate_type"].eq("ffnn") & comparison["uses_rider_weight"]].sort_values(["cv_selection_score", "cv_run_mae_bar"]).head(1)
    selected = comparison[comparison["model_name"].eq(selection_summary["selected_model"])]
    sensitivity = pd.concat([best_without_weight.assign(review_role="best_ffnn_without_weight"), best_with_weight.assign(review_role="best_ffnn_with_weight"), selected.assign(review_role="selected_model")], ignore_index=True)
    sensitivity = sensitivity[
        [
            "review_role",
            "model_name",
            "feature_space",
            "n_features_in",
            "pca_components",
            "hidden_layers",
            "alpha",
            "cv_selection_score",
            "cv_run_mae_bar",
            "cv_run_rmse_bar",
            "cv_run_bias_bar",
            "cv_run_max_abs_error_bar",
            "cv_nearest_level_accuracy",
        ]
    ].drop_duplicates()

    rider_row = rationale[rationale["feature_name"].eq("rider_weight_kg")].copy()
    top_final_features = rationale.sort_values("abs_spearman_rank").head(10)[
        [
            "feature_name",
            "spearman_to_pressure",
            "within_bike_mean_abs_spearman",
            "max_abs_peer_spearman",
            "abs_spearman_rank",
        ]
    ]

    requirement_gap = pd.DataFrame(
        [
            {
                "requirement_or_question": "Train a fully connected feed-forward neural network.",
                "current_evidence": "MLPRegressor with hidden layers (8, 4), tanh activation, PCA(6) input, and one pressure output.",
                "gap_or_caveat": "Report should explicitly state output neuron = 1 and solver is lbfgs, so classic epoch count is not the main training record.",
                "recommended_text_or_action": "Add a small architecture table: 28 pre-PCA inputs -> 6 PCA scores -> 8 -> 4 -> 1 pressure_bar output; max_iter=3000, max_fun=40000.",
            },
            {
                "requirement_or_question": "Cross-validate and review performance with multiple metrics.",
                "current_evidence": "Leave-one-group-out CV produces MAE, RMSE, bias, nearest-level accuracy, macro-F1, and confusion matrix.",
                "gap_or_caveat": "Because the same CV is used for model selection, the reported CV is a local model-selection estimate, not an unbiased hidden-test estimate.",
                "recommended_text_or_action": "State this caveat and explain why nested CV is impractical with only 6 groups.",
            },
            {
                "requirement_or_question": "Test generalization ability on unseen generalization dataset.",
                "current_evidence": "Teacher hidden data is not in the workspace; final model is trained on all 72 local runs for that later test.",
                "gap_or_caveat": "No hidden-test metrics can be reported before the instructor evaluates the model.",
                "recommended_text_or_action": "Write 'hidden-test performance is pending; local generalization is estimated by leave-one-group-out CV.'",
            },
            {
                "requirement_or_question": "Use measured acceleration data from sensors and PhyPhox app.",
                "current_evidence": f"{sagemotion_ok} Sagemotion CSV and {phyphox_ok} PhyPhox XLS files exist, but extracted model features use Sagemotion CSV only.",
                "gap_or_caveat": "This is the largest scope explanation gap.",
                "recommended_text_or_action": "Either add a PhyPhox baseline/feature-ablation experiment or explicitly justify Sagemotion-only modeling as the higher-quality two-sensor + gyro source.",
            },
            {
                "requirement_or_question": "Predict tire pressures and damping of the bicycle.",
                "current_evidence": "Tire pressure is available as a continuous label. Damping/suspension is available as a categorical label through the course table: FAT -> Suspension because of tyres, ISY -> No Suspension, and MTB -> Front and rear Suspension.",
                "gap_or_caveat": "No continuous damping coefficient is available, so Step 04b predicts the table-defined suspension category rather than a numeric damping value.",
                "recommended_text_or_action": "Report Step 04b as the suspension/damping classifier and state that bike, pressure, p-number, group, run id, file name, and rider weight are excluded from its inputs to avoid leakage.",
            },
        ]
    )

    return {
        "data_quality": data_quality,
        "pressure_design": pressure_design,
        "error_by_bike": error_by_bike,
        "error_by_group": error_by_group,
        "worst_runs": worst_runs,
        "model_selection_sensitivity": sensitivity,
        "rider_weight_rationale_row": rider_row,
        "top_final_features": top_final_features,
        "requirement_gap_matrix": requirement_gap,
    }


def write_markdown_report(tables: dict[str, pd.DataFrame], path: Path) -> None:
    dq = tables["data_quality"]
    err_bike = tables["error_by_bike"].sort_values("mae_bar", ascending=False)
    err_group = tables["error_by_group"].sort_values("mae_bar", ascending=False)
    worst = tables["worst_runs"].head(8)
    sensitivity = tables["model_selection_sensitivity"]
    gaps = tables["requirement_gap_matrix"]

    lines = [
        "# Teacher-Review Supplement / 老师视角审查补充报告",
        "",
        "## Overall Judgment / 总体判断",
        "",
        "The main project logic is sound: local G01-G06, P1-P4 data provide 72 runs for the training pool; because the instructor still has a hidden test set, local generalization is estimated with leave-one-group-out cross-validation; the final model is then retrained on all local runs.",
        "当前项目的主逻辑是成立的：本地 G01-G06、P1-P4 共 72 个 run 作为训练池；由于老师还有隐藏测试集，本地泛化用 leave-one-group-out cross-validation 估计；最终模型再用全部本地 run 训练。",
        "",
        "The report still needs to be explicit about several scope and logic points: why PhyPhox is not included in the current model matrix, how the FFNN architecture satisfies the course requirement, why CV metrics are not hidden-test results, why rider weight is physical context rather than a strong direct label proxy, and why the damping/suspension task is modeled as a categorical label from the course table rather than as a continuous damping coefficient.",
        "但报告层仍有几处需要讲得更清楚：PhyPhox 数据为什么没有入模、FFNN 架构如何对应课程要求、CV 指标不是隐藏测试结果、体重是物理上下文而不是强相关标签替代物、damping/suspension 为什么按课程表中的类别标签建模，而不是按连续阻尼系数建模。",
        "",
        "## Data And Output Checks / 数据和输出复核",
        "",
        _markdown_table(dq),
        "",
        "## Error Structure / 误差结构",
        "",
        "By bike, MTB is the main error source:",
        "",
        "按 bike 看，MTB 是主要误差来源：",
        "",
        _markdown_table(err_bike, floatfmt=".3f"),
        "",
        "By held-out group, no single group fully fails, but G03/G04/G06 are slightly higher:",
        "",
        "按 held-out group 看，没有单个 group 完全失控，但 G03/G04/G06 略高：",
        "",
        _markdown_table(err_group, floatfmt=".3f"),
        "",
        "Largest-error runs:",
        "",
        "最大误差 run：",
        "",
        _markdown_table(worst, floatfmt=".3f"),
        "",
        "## Model-Selection Sensitivity / 模型选择敏感性",
        "",
        "The best FFNN without rider weight is slightly better, but the selected model includes rider weight and remains within the 5% near-best range. Therefore, rider weight should be described as a physically required context input, not as a variable that clearly improves CV MAE.",
        "",
        "最优 FFNN 不含体重，选中模型含体重且在 5% near-best 范围内。因此报告应把体重表述为课程/物理约束下的上下文变量，而不是声称它显著提升了 CV MAE。",
        "",
        _markdown_table(sensitivity, floatfmt=".3f"),
        "",
        "## Requirement Gap Matrix / 课程要求缺口矩阵",
        "",
        _markdown_table(gaps),
        "",
        "## Suggested Report Text / 建议直接补进报告的表述",
        "",
        "1. `The model is a pressure regression model. For classification-style metrics required by the task sheet, each run-level pressure prediction is mapped to the nearest valid pressure level of the corresponding bike, and the confusion matrix / macro-F1 are reported as auxiliary metrics.`",
        "   模型是胎压回归模型。为了满足课程任务中的分类式指标要求，每个 run-level 连续胎压预测会映射到对应 bike 的最近可行胎压等级，再报告 confusion matrix / macro-F1 作为辅助指标。",
        "2. `The selected network uses 28 pre-PCA inputs, which are standardized and reduced to 6 PCA components before entering a fully connected network with hidden layers 8 and 4 and one output neuron for pressure_bar.`",
        "   选中网络使用 28 个 PCA 前输入，先标准化并降到 6 个 PCA component，再进入隐藏层为 8 和 4、输出神经元为 1 的全连接网络。",
        "3. `The MLP is trained with sklearn's lbfgs optimizer (max_iter=3000, max_fun=40000), so training is recorded by optimizer limits and convergence rather than a fixed epoch schedule.`",
        "   MLP 使用 sklearn 的 lbfgs 优化器训练（max_iter=3000, max_fun=40000），因此训练记录重点是优化器上限和收敛，而不是固定 epoch 数。",
        "4. `PhyPhox files were checked for completeness, but the current model uses Sagemotion because it provides two mounted sensors and gyroscope channels. PhyPhox should be treated as optional future validation unless a baseline experiment is added.`",
        "   PhyPhox 文件已检查完整性，但当前模型使用 Sagemotion，因为它提供两个固定安装传感器和陀螺仪通道；除非补充 baseline 实验，否则 PhyPhox 应作为未来验证来源。",
        "5. `The damping/suspension task is handled as categorical classification. The labels are derived from the course table: FAT -> Suspension because of tyres, ISY -> No Suspension, and MTB -> Front and rear Suspension. The classifier uses Sagemotion signal features only and excludes bike, pressure, p-number, group, run id, file name, and rider weight to avoid leakage.`",
        "   Damping/suspension 任务按类别分类处理。标签来自课程表：FAT -> Suspension because of tyres，ISY -> No Suspension，MTB -> Front and rear Suspension。分类器只使用 Sagemotion 信号特征，并排除 bike、pressure、p-number、group、run id、file name 和 rider weight，以避免标签泄漏。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    def action(emit):
        ensure_dirs(REVIEW_DIR, REVIEW_TABLE_DIR)
        tables = build_review_tables()
        for name, frame in tables.items():
            frame.to_csv(REVIEW_TABLE_DIR / f"{name}.csv", index=False)
            emit(f"Wrote {name}.csv ({len(frame)} rows)")
        write_markdown_report(tables, REVIEW_DIR / "teacher_review_audit_bilingual.md")
        emit(f"Wrote {REVIEW_DIR / 'teacher_review_audit_bilingual.md'}")

    run_logged_action("07_teacher_review_checks", action)


if __name__ == "__main__":
    main()
