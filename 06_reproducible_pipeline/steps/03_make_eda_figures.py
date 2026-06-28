from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from feature_explanation import write_feature_explanation_artifacts
from plotting import (
    plot_counts,
    plot_feature_relationship,
    plot_final_input_correlation_matrix,
    plot_pca,
    plot_pressure_design,
    plot_target_correlation_ranking,
    use_chart_theme,
)
from project_config import FIG_EDA_DIR, MODEL_DIR, REPORT_DIR, TABLE_DIR, ensure_dirs
from step_logging import run_logged_action


def write_eda_report(labels: pd.DataFrame, features: pd.DataFrame, inventory: pd.DataFrame, explanation_summary: dict, report_path: Path) -> None:
    sagemotion_ok = int(((inventory["source"] == "sagemotion") & (inventory["parse_status"] == "ok")).sum())
    phyphox_ok = int(((inventory["source"] == "phyphox") & (inventory["parse_status"] == "ok")).sum())
    lines = [
        "# Training-Pool Data Processing And EDA Report / Training-Pool 数据处理与 EDA 报告",
        "",
        "## Current Data Rule / 当前数据规则",
        "",
        "- All P1-P4 runs are used as local training data. / P1-P4 全部作为本地训练数据使用。",
        "- The instructor hidden data is the true external test set. / 老师手里的隐藏数据才是真正外部测试集。",
        "- Local performance is estimated with leave-one-group-out cross-validation. / 本地效果使用留一组交叉验证估计。",
        "",
        "## Data Scale / 数据规模",
        "",
        "| Item / 项目 | Value / 数值 |",
        "|---|---:|",
        f"| Labeled runs / 标注 run 数 | {labels['run_id'].nunique()} |",
        f"| Groups / 组数 | {labels['group'].nunique()} |",
        f"| Bike types / 单车类型 | {labels['bike'].nunique()} |",
        f"| Window feature rows / 窗口特征行数 | {len(features)} |",
        f"| Parseable Sagemotion CSV files / 可解析 Sagemotion CSV | {sagemotion_ok} |",
        f"| Parseable PhyPhox XLS files / 可解析 PhyPhox XLS | {phyphox_ok} |",
        "",
        "## Raw-Data Scope / 原始数据使用范围说明",
        "",
        "The current model extracts features only from Sagemotion CSV files.",
        "当前模型特征只从 Sagemotion CSV 提取。",
        "",
        "PhyPhox XLS files are checked in the file inventory, but they are not included in the current FFNN feature matrix.",
        "PhyPhox XLS 已在文件清单中检查完整性，但没有进入当前 FFNN 特征矩阵。",
        "",
        "Sagemotion provides two fixed mounted sensors with both acceleration and angular-velocity channels, so it is better suited as the main structural vibration input.",
        "Sagemotion 提供两个固定安装传感器，并包含加速度和角速度通道，因此更适合作为主要结构振动输入。",
        "",
        "## Cleaning And Feature Processing / 数据清洗和特征处理",
        "",
        "- CSV files are read with `pandas`, and sensor columns are converted to numeric values. / 使用 `pandas` 读取 CSV，并把传感器列转换为数值。",
        "- `numpy` is used for vector magnitudes, window indices, and FFT calculations. / 使用 `numpy` 处理向量模长、窗口索引和 FFT。",
        "- `scipy.signal.butter/filtfilt` is used for band-pass filtering. / 使用 `scipy.signal.butter/filtfilt` 做带通滤波。",
        "- Missing and infinite values are interpolated first, then remaining missing values are filled with medians. / 对缺失值和无穷值先插值，再用中位数填补。",
        "- The active window is cropped by acceleration energy; its duration comes from Measurement Details ride time. / 通过加速度能量裁剪 active window，窗口长度来自 Measurement Details 的 ride time。",
        "- Within each active window, 1 s windows with 50% overlap are extracted. / 在 active window 内使用 1 秒窗口和 50% overlap。",
        "- Each window produces time-domain statistics and frequency-domain features. / 每个窗口提取时域统计量和频域特征。",
        "",
        "## PCA",
        "",
        "- PCA is used for EDA visualization and candidate model dimensionality reduction; it is not a label. / PCA 只用于 EDA 可视化和模型候选降维，不是标签。",
        "- EDA PCA is fitted on all local P1-P4 data because all of these runs belong to the local training pool. / 当前 EDA PCA 使用全部本地 P1-P4 数据拟合，因为这些数据都属于训练池。",
        "- True local generalization is estimated later through leave-one-group-out CV. / 真正的本地泛化估计在模型步骤中用 leave-one-group-out 完成。",
        "",
        "## Feature Explanation And Correlation Evidence / 特征解释和相关性证据",
        "",
        "- Correlation analysis uses run-level medians, not raw window rows, because windows from the same run are highly related. / 相关性分析使用 run-level 中位数，而不是窗口级行，因为同一个 run 内的窗口高度相关。",
        "- Pearson and Spearman correlations explain linear and monotonic relationships with `pressure_bar`. / 输出 Pearson 和 Spearman 相关性，用来说明特征与 `pressure_bar` 的线性关系和单调关系。",
        "- The final-input correlation matrix identifies redundancy among selected inputs. / 相关性矩阵用于识别最终输入特征之间的冗余结构。",
        "- Correlation is explanatory evidence only; final model choice is still decided by leave-one-group-out CV. / 相关性只作为解释和候选筛选证据；最终模型仍由 leave-one-group-out CV 决定。",
        f"- Candidate inputs: {explanation_summary['n_candidate_inputs']}. / 候选输入数：{explanation_summary['n_candidate_inputs']}。",
        f"- Current final feature space: `{explanation_summary['final_feature_space']}`, {explanation_summary['n_final_inputs']} inputs. / 当前最终输入空间：`{explanation_summary['final_feature_space']}`，共 {explanation_summary['n_final_inputs']} 个输入。",
        f"- Rider weight included: {explanation_summary['uses_rider_weight']}. / 是否包含 rider weight：{explanation_summary['uses_rider_weight']}。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    def action(emit):
        ensure_dirs(FIG_EDA_DIR, MODEL_DIR, REPORT_DIR)
        use_chart_theme()

        labels = pd.read_csv(TABLE_DIR / "training_pool_labels.csv")
        inventory = pd.read_csv(TABLE_DIR / "training_pool_raw_file_inventory.csv")
        features = pd.read_csv(TABLE_DIR / "training_pool_window_features.csv")
        run_summary = pd.read_csv(TABLE_DIR / "training_pool_run_feature_summary.csv")
        explanation = write_feature_explanation_artifacts(features, TABLE_DIR, MODEL_DIR)

        paths = [
            plot_pressure_design(labels),
            plot_counts(labels, features),
            plot_pca(features),
            plot_feature_relationship(run_summary),
            plot_target_correlation_ranking(explanation["correlations"]),
            plot_final_input_correlation_matrix(explanation["final_correlation_matrix"]),
        ]
        summary = {
            "n_labeled_runs": int(labels["run_id"].nunique()),
            "n_windows": int(len(features)),
            "figure_paths": [str(path) for path in paths],
            "pca_policy": "EDA PCA is fitted on all local P1-P4 training-pool data.",
            "feature_explanation": explanation["summary"],
        }
        (MODEL_DIR / "training_pool_eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_eda_report(labels, features, inventory, explanation["summary"], REPORT_DIR / "training_pool_data_processing_report_bilingual.md")

        for path in paths:
            emit(f"Figure OK: {path.name}")

    run_logged_action("03_make_eda_figures", action)


if __name__ == "__main__":
    main()
