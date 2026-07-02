# VSCode Step-By-Step Run Guide / VSCode 逐步运行说明

P1-P4 full training-pool version.  
P1-P4 全量训练版。

The current rule is: all P1-P4 runs are local training data. The instructor still has a hidden test set, so P3/P4 are no longer used as a local final test.  
当前规则是：P1-P4 全部都是本地可用训练数据。老师手里还有隐藏测试集，所以我们不再把 P3/P4 当成本地 final test。

## 1. Folder To Open / 打开哪个文件夹

Recommended VSCode folder:  
推荐在 VSCode 打开：

`C:\Users\user\Desktop\IMSE\imse_all_groups`

Runnable step scripts:  
核心可运行脚本在：

`C:\Users\user\Desktop\IMSE\imse_all_groups\06_reproducible_pipeline\steps`

Core module code:  
核心模块代码在：

`C:\Users\user\Desktop\IMSE\imse_all_groups\06_reproducible_pipeline\src`

## 2. Data Split / 新数据划分

| Item / 项目 | Current rule / 当前规则 |
|---|---|
| Local training pool / 本地训练池 | All 72 P1/P2/P3/P4 runs from G01-G06 / G01-G06 的 P1/P2/P3/P4 全部 72 个 run |
| Local validation / 本地验证 | leave-one-group-out cross-validation / 留一组交叉验证 |
| External test / 外部测试 | Instructor hidden test data / 老师隐藏测试数据 |
| Pressure target / 胎压预测目标 | Continuous tire pressure `pressure_bar` / 连续胎压 `pressure_bar` |
| Pressure-model inputs / 胎压模型输入 | Signal features, bike type one-hot, `rider_weight_kg` / 信号特征、bike type one-hot、`rider_weight_kg` |
| Bike-type target / 单车类型预测目标 | Bike class `FAT`, `ISY`, or `MTB` / 单车类别 `FAT`、`ISY` 或 `MTB` |
| Bike-type model inputs / 单车类型模型输入 | Sagemotion signal features only; no `bike`, `pressure_bar`, `rider_weight_kg`, or split metadata / 只使用 Sagemotion 信号特征；不使用 `bike`、`pressure_bar`、`rider_weight_kg` 或划分元数据 |

Meaning of leave-one-group-out:  
leave-one-group-out 的含义：

- Each fold holds out one complete group for validation. / 每一折留出 1 个完整 group 做 validation。
- The other five groups are used for training. / 其余 5 个 group 用来训练。
- Every group is validated once. / 每个 group 都会被验证一次。
- Metrics are computed after aggregating predictions to run level, not by random window splitting. / 所有指标都按 run-level 聚合后计算，不用随机窗口划分。

## 3. Recommended Run Order / 推荐运行顺序

| Step | Python file / Python 文件 | Purpose / 做什么 |
|---:|---|---|
| 00 | `00_check_environment.py` | Check raw data, dependencies, and module files / 检查原始数据、依赖包和模块文件 |
| 01 | `01_build_labels_inventory.py` | Build labels from Measurement Details and check raw-file inventory / 从 Measurement Details 建标签表，检查原始文件清单 |
| 02 | `02_extract_window_features.py` | Clean signals, crop active windows, extract 1 s window features / 清洗信号、裁剪 active window、切 1 秒窗口、提取特征 |
| 03 | `03_make_eda_figures.py` | Generate EDA figures, PCA figures, and data-processing report / 生成 EDA 图、PCA 图和数据处理报告 |
| 04 | `04_group_cv_model_selection.py` | Select the FFNN with leave-one-group-out CV / 用 leave-one-group-out CV 选择 FFNN |
| 04b | `04b_bike_type_classification.py` | Select and train the bike-type classifier / 选择并训练单车类型分类模型 |
| 05 | `05_train_final_model.py` | Train the final model on all 72 runs and save `.pkl` / 用全部 72 个 run 训练最终模型，保存 `.pkl` |
| 06 | `06_check_outputs.py` | Check that required outputs exist / 检查关键输出是否齐全 |
| 07 | `07_teacher_review_checks.py` | Generate teacher-review supplement and audit tables / 生成老师视角审查补充报告和复核表 |

One-click run in VSCode:  
一键运行：

Press `Ctrl+Shift+P`, choose `Tasks: Run Task`, then run:

`Training Pool Run Full Pipeline`

Manual PowerShell run:  
如果在 PowerShell 里手动运行，请使用当前环境中的 Python：

```powershell
cd "C:\Users\user\Desktop\IMSE\imse_all_groups\06_reproducible_pipeline"
python .\steps\run_all_training_pool.py
```

Step-by-step manual run:  
逐步运行：

```powershell
cd "C:\Users\user\Desktop\IMSE\imse_all_groups\06_reproducible_pipeline"
python .\steps\00_check_environment.py
python .\steps\01_build_labels_inventory.py
python .\steps\02_extract_window_features.py
python .\steps\03_make_eda_figures.py
python .\steps\04_group_cv_model_selection.py
python .\steps\04b_bike_type_classification.py
python .\steps\05_train_final_model.py
python .\steps\06_check_outputs.py
python .\steps\07_teacher_review_checks.py
```

## 4. Logs / 日志

Each step writes a log file to:  
每一步都会生成日志：

`C:\Users\user\Desktop\IMSE\imse_all_groups\06_reproducible_pipeline\logs`

Logs record the step name, Python path, project path, key statistics, and `DONE` or `FAILED`.  
日志会记录当前步骤名、Python 路径、项目路径、关键统计结果，以及 `DONE` 或 `FAILED`。

## 5. Key Outputs / 关键输出

| Output / 输出 | Path / 路径 |
|---|---|
| Labels / 标签表 | `03_outputs/tables/training_pool_labels.csv` |
| Raw-file inventory / 原始文件清单 | `03_outputs/tables/training_pool_raw_file_inventory.csv` |
| Window features / 窗口特征表 | `03_outputs/tables/training_pool_window_features.csv` |
| Candidate input columns / 候选输入列 | `03_outputs/tables/training_pool_candidate_input_features.csv` |
| Candidate input correlations / 候选输入相关性 | `03_outputs/tables/training_pool_candidate_feature_target_correlations.csv` |
| Final input rationale / 最终输入选择理由 | `03_outputs/tables/training_pool_final_input_feature_rationale.csv` |
| Final input correlation matrix / 最终输入相关性矩阵 | `03_outputs/tables/training_pool_final_input_correlation_matrix.csv` |
| Model comparison / 模型比较 | `03_outputs/tables/training_pool_ffnn_model_comparison.csv` |
| Selected-model CV predictions / 选中模型 CV 预测 | `03_outputs/tables/training_pool_ffnn_selected_cv_predictions.csv` |
| Final-model training-fit predictions / 最终模型训练拟合预测 | `03_outputs/tables/training_pool_ffnn_final_model_training_fit_predictions.csv` |
| Final model file / 最终模型文件 | `03_outputs/models/training_pool_ffnn_final_model.pkl` |
| Bike-type model comparison / 单车类型模型比较 | `03_outputs/tables/training_pool_bike_type_model_comparison.csv` |
| Bike-type selected CV predictions / 单车类型选中模型 CV 预测 | `03_outputs/tables/training_pool_bike_type_selected_cv_predictions.csv` |
| Bike-type CV confusion matrix / 单车类型 CV 混淆矩阵 | `03_outputs/tables/training_pool_bike_type_cv_confusion.csv` |
| Bike-type final model file / 单车类型最终模型文件 | `03_outputs/models/training_pool_bike_type_final_model.pkl` |
| Data-processing report / 数据处理报告 | `04_report/training_pool_data_processing_report_bilingual.md` |
| Model-selection report / 模型选择报告 | `04_report/training_pool_ffnn_cv_model_report_bilingual.md` |
| Final-model report / 最终模型报告 | `04_report/training_pool_ffnn_final_model_report_bilingual.md` |
| Bike-type classifier report / 单车类型分类报告 | `04_report/training_pool_bike_type_classifier_report_bilingual.md` |
| Teacher-review supplement / 老师审查补充报告 | `05_teacher_review/teacher_review_audit_bilingual.md` |
| Teacher-review tables / 老师审查复核表 | `05_teacher_review/tables/*.csv` |

## 6. Code Reading Order / 读代码的建议

Start with `steps/` because each script corresponds to one pipeline stage. Then read `src/`:  
先看 `steps/`，因为每个脚本都对应项目的一步。再看 `src/`：

| Module / 模块 | Purpose / 用途 |
|---|---|
| `project_config.py` | Paths, colors, source label table, current training rules / 路径、颜色、标签原始表、当前训练规则 |
| `labels.py` | Generate labels from Measurement Details / 从 Measurement Details 生成标签表 |
| `data_io.py` | Filename parsing, CSV reading, file inventory / 文件名解析、CSV 读取、文件清单 |
| `signal_features.py` | Signal cleaning, filtering, active window, FFT/time-domain features / 信号清洗、滤波、active window、FFT/时域特征 |
| `feature_pipeline.py` | Window slicing, feature tables, model input matrix / 窗口切片、特征表、模型输入矩阵 |
| `modeling.py` | FFNN candidates, group CV, final training / FFNN 候选、组级 CV、最终模型训练 |
| `bike_type_modeling.py` | Bike-type classifier candidates, leakage-safe feature spaces, group CV, and final training / 单车类型分类候选、防泄漏特征空间、组级 CV 和最终训练 |
| `plotting.py` | EDA and model figures / EDA 和模型图表 |
