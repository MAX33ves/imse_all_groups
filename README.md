# IMSE Bicycle Tire-Pressure And Suspension Models / IMSE 自行车胎压与悬挂类型模型项目

This repository contains the reproducible analysis pipeline, model outputs, figures, reports, and bilingual Obsidian notes for the IMSE bicycle tire-pressure and suspension/damping prediction project.  
本仓库包含 IMSE 自行车胎压与悬挂/阻尼类型预测项目的可复现分析流程、模型输出、图表、报告和中英双语 Obsidian 笔记。

## Project Summary / 项目概要

The project now has two supervised models: a tire-pressure regressor for continuous `pressure_bar`, and a separate suspension/damping classifier whose labels are derived from the course table mapping bike type to suspension type.  
项目现在包含两个监督学习模型：一个用于连续胎压 `pressure_bar` 的回归模型，以及一个根据课程表“单车类型 -> 悬挂类型”映射得到标签的悬挂/阻尼类型分类模型。

Selected pressure model:  
选中的胎压模型：

`ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8x4_a1_median`

Selected suspension model:  
选中的悬挂类型模型：

`suspension_ffnn_ensemble_ens3_signal_full_pca10_tanh_h6_a1`

Key local CV metrics:  
关键本地 CV 指标：

| Metric / 指标 | Value / 数值 |
|---|---:|
| Leave-one-group-out CV MAE | 0.290 bar |
| Leave-one-group-out CV RMSE | 0.388 bar |
| Nearest-level accuracy | 0.597 |
| Macro-F1 | 0.466 |
| Suspension-type CV accuracy | 0.986 |
| Suspension-type CV macro-F1 | 0.986 |

## Data Scope / 数据范围

All observed local G01-G06 P1-P4 runs are used as the training pool: 72 runs and 873 window-level feature rows.  
本地已观测的 G01-G06、P1-P4 全部作为训练池使用：72 个 run，873 行窗口级特征。

The instructor hidden test set is not included in this workspace. Local generalization is estimated by leave-one-group-out cross-validation.  
老师隐藏测试集不在当前 workspace 中。本地泛化能力使用 leave-one-group-out 交叉验证估计。

To rerun the full pipeline from raw data, place the raw folder next to this repository as:  
如果要从原始数据重新运行完整流程，请把原始数据文件夹放在本仓库同级位置：

```text
IMSE/
  imse_all_groups/
  Measurement_Campaign/
```

The code expects the raw data at `../Measurement_Campaign`.  
代码默认在 `../Measurement_Campaign` 读取原始数据。

## Repository Structure / 仓库结构

| Path / 路径 | Purpose / 用途 |
|---|---|
| `00_requirements/` | Requirement matrix and current split rule / 要求矩阵和当前数据划分规则 |
| `03_outputs/` | Generated tables, figures, model summaries, and final `.pkl` model / 生成的表格、图、模型摘要和最终 `.pkl` 模型 |
| `04_report/` | Bilingual technical reports / 中英双语技术报告 |
| `05_teacher_review/` | Teacher-style audit supplement and review tables / 老师视角审查补充报告和复核表 |
| `06_reproducible_pipeline/` | Reproducible Python pipeline / 可复现 Python 流程 |
| `obsidian/` | Bilingual Obsidian knowledge base / 中英双语 Obsidian 知识库 |

## How To Run / 如何运行

Install dependencies:  
安装依赖：

```powershell
pip install -r requirements.txt
```

Run the full pipeline:  
运行完整流程：

```powershell
cd .\06_reproducible_pipeline
python .\steps\run_all_training_pool.py
```

Check outputs only:  
只检查关键输出：

```powershell
cd .\06_reproducible_pipeline
python .\steps\06_check_outputs.py
```

VSCode users can also run the configured tasks under `Tasks: Run Task`.  
VSCode 用户也可以通过 `Tasks: Run Task` 运行已经配置好的任务。

## Main Reports / 主要报告

- `04_report/training_pool_data_processing_report_bilingual.md`
- `04_report/training_pool_ffnn_cv_model_report_bilingual.md`
- `04_report/training_pool_ffnn_final_model_report_bilingual.md`
- `04_report/training_pool_suspension_classifier_report_bilingual.md`
- `05_teacher_review/teacher_review_audit_bilingual.md`
- `obsidian/00_Index.md`

## Important Caveats / 重要注意事项

- CV metrics are local post-selection estimates, not hidden-test results. / CV 指标是本地模型选择后的泛化估计，不是隐藏测试集结果。
- Training-pool fit metrics are not test metrics. / 训练池拟合指标不是测试指标。
- The suspension classifier deliberately excludes `bike`, `suspension_type`, `pressure_bar`, `p_number`, `group`, `run_id`, file name, and `rider_weight_kg` from its inputs. / 悬挂类型分类模型刻意不使用 `bike`、`suspension_type`、`pressure_bar`、`p_number`、`group`、`run_id`、文件名和 `rider_weight_kg` 作为输入。
- PhyPhox files were checked for completeness, but the current FFNN feature matrix uses Sagemotion CSV features only. / PhyPhox 文件已检查完整性，但当前 FFNN 特征矩阵只使用 Sagemotion CSV 特征。
- Suspension/damping labels are derived from the provided bike-to-suspension table, so the model predicts the table-defined suspension category from signal features. / 悬挂/阻尼标签来自给定的“车型到悬挂类型”表，因此模型从信号特征预测表中定义的悬挂类别。
