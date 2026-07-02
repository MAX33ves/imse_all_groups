# File Map / 文件地图

## Run Entrypoints / 运行入口

| 文件 / File | 作用 / Purpose |
|---|---|
| `06_reproducible_pipeline/steps/run_all_training_pool.py` | 一键运行全部步骤 / runs all steps |
| `06_reproducible_pipeline/steps/00_check_environment.py` | 环境检查 / environment check |
| `06_reproducible_pipeline/steps/01_build_labels_inventory.py` | 标签和原始文件清单 / labels and raw-file inventory |
| `06_reproducible_pipeline/steps/02_extract_window_features.py` | 特征提取 / feature extraction |
| `06_reproducible_pipeline/steps/03_make_eda_figures.py` | EDA / EDA figures and reports |
| `06_reproducible_pipeline/steps/04_group_cv_model_selection.py` | 模型选择 / model selection |
| `06_reproducible_pipeline/steps/04b_bike_type_classification.py` | 单车类型分类模型 / bike-type classifier |
| `06_reproducible_pipeline/steps/05_train_final_model.py` | 最终模型 / final model training |
| `06_reproducible_pipeline/steps/06_check_outputs.py` | 输出检查 / output checks |
| `06_reproducible_pipeline/steps/07_teacher_review_checks.py` | 老师视角审查补充 / teacher-review supplement |

## Core Outputs / 核心输出

| 文件 / File | 说明 / Description |
|---|---|
| `03_outputs/tables/training_pool_labels.csv` | 标签表 / label table |
| `03_outputs/tables/training_pool_window_features.csv` | 窗口特征 / window-level features |
| `03_outputs/tables/training_pool_candidate_input_features.csv` | 候选输入列，包含 `rider_weight_kg` / candidate input columns, including `rider_weight_kg` |
| `03_outputs/tables/training_pool_candidate_feature_target_correlations.csv` | 候选输入与胎压的相关性 / candidate-input correlations with tire pressure |
| `03_outputs/tables/training_pool_final_input_feature_rationale.csv` | 最终输入选择理由 / final-input selection rationale |
| `03_outputs/tables/training_pool_final_input_correlation_matrix.csv` | 最终输入相关性矩阵 / final-input correlation matrix |
| `03_outputs/tables/training_pool_high_redundancy_feature_pairs.csv` | 高冗余特征对 / high-redundancy feature pairs |
| `03_outputs/tables/training_pool_ffnn_model_comparison.csv` | 候选模型比较 / candidate model comparison |
| `03_outputs/tables/training_pool_ffnn_selected_cv_predictions.csv` | 选中模型 CV 预测 / selected-model CV predictions |
| `03_outputs/models/training_pool_ffnn_selection_summary.json` | 模型选择摘要 / model-selection summary |
| `03_outputs/models/training_pool_ffnn_final_model.pkl` | 最终模型 / final model |
| `03_outputs/tables/training_pool_bike_type_model_comparison.csv` | 单车类型候选模型比较 / bike-type candidate model comparison |
| `03_outputs/tables/training_pool_bike_type_selected_cv_predictions.csv` | 单车类型选中模型 CV 预测 / selected bike-type CV predictions |
| `03_outputs/tables/training_pool_bike_type_cv_confusion.csv` | 单车类型 CV 混淆矩阵 / bike-type CV confusion matrix |
| `03_outputs/models/training_pool_bike_type_selection_summary.json` | 单车类型模型选择摘要 / bike-type model-selection summary |
| `03_outputs/models/training_pool_bike_type_final_model.pkl` | 单车类型最终模型 / final bike-type model |
| `04_report/training_pool_ffnn_cv_model_report_bilingual.md` | CV 模型报告 / CV model report |
| `04_report/training_pool_ffnn_final_model_report_bilingual.md` | 最终模型报告 / final-model report |
| `04_report/training_pool_bike_type_classifier_report_bilingual.md` | 单车类型分类报告 / bike-type classifier report |
| `05_teacher_review/teacher_review_audit_bilingual.md` | 老师视角审查补充报告 / teacher-review supplement |
| `05_teacher_review/tables/*.csv` | 数据质量、误差结构、模型敏感性和课程缺口复核表 / review tables for data quality, error structure, model sensitivity, and requirement gaps |

## Logs / 日志

每一步日志在：

Step logs are stored in:

`06_reproducible_pipeline/logs`

日志文件名包含时间戳和 step 名，方便追踪是哪一步生成的结果。

Each log file name contains a timestamp and the step name, making it easy to trace which step produced each result.
