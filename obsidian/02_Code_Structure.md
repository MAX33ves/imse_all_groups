# Code Structure / 代码结构

## Where To Start / 你应该先看哪个文件夹

主要看这个文件夹：

Start with this folder:

`06_reproducible_pipeline/steps`

这里每个脚本都可以单独运行，每个脚本对应项目的一步。

Each script can be run independently, and each script corresponds to one project step.

底层函数在：

The lower-level implementation lives in:

`06_reproducible_pipeline/src`

## `steps` Folder / `steps` 文件夹

| 脚本 / Script | 作用 / Purpose |
|---|---|
| `00_check_environment.py` | 检查原始数据、依赖包、模块文件 / checks raw data, Python packages, and source modules |
| `01_build_labels_inventory.py` | 生成标签表和原始文件清单 / builds the label table and raw-file inventory |
| `02_extract_window_features.py` | 清洗信号、裁剪 active window、切窗口、提取特征 / cleans signals, crops active windows, slices windows, and extracts features |
| `03_make_eda_figures.py` | 生成 EDA 图、PCA 图、数据处理报告 / creates EDA figures, PCA figures, and the data-processing report |
| `04_group_cv_model_selection.py` | 组级交叉验证并选择 FFNN / performs group-level CV and selects the FFNN |
| `04b_bike_type_classification.py` | 组级交叉验证并选择悬挂类型分类模型 / performs group-level CV and selects the suspension classifier |
| `05_train_final_model.py` | 用全部 72 个 run 训练最终模型 / trains the final model on all 72 runs |
| `06_check_outputs.py` | 检查关键输出 / checks required outputs |
| `07_teacher_review_checks.py` | 生成老师视角审查补充 / creates the teacher-review supplement |
| `run_all_training_pool.py` | 一键运行全部步骤 / runs the full pipeline |

## `src` Folder / `src` 文件夹

| 模块 / Module | 主要内容 / Main contents |
|---|---|
| `project_config.py` | 路径、标签原始表、训练规则、颜色配置 / paths, source label table, training rules, color settings |
| `labels.py` | 从 Measurement Details 建立真实胎压标签 / builds true pressure labels from Measurement Details |
| `data_io.py` | 文件名解析、CSV 读取、文件清单 / file-name parsing, CSV reading, raw-file inventory |
| `signal_features.py` | 缺失值处理、滤波、active window、FFT/时域特征 / missing-value handling, filtering, active-window crop, FFT and time-domain features |
| `feature_pipeline.py` | 窗口切片、特征表、模型输入矩阵 / window slicing, feature tables, model input matrix |
| `modeling.py` | FFNN、PCA、CV、最终训练 / FFNN, PCA, CV, and final training |
| `bike_type_modeling.py` | 悬挂类型分类候选、防泄漏输入空间、组级 CV、最终训练 / suspension classifier candidates, leakage-safe input spaces, group CV, and final training |
| `plotting.py` | 所有 EDA 和模型图 / all EDA and model figures |
| `step_logging.py` | 每一步日志记录 / per-step logging |

## Why It Is Split This Way / 为什么这样拆

- 队友可以先看 `steps` 理解流程。
- Teammates can read `steps` first to understand the workflow.
- 想看细节时再进入 `src`。
- They can then enter `src` for implementation details.
- 每一步都有日志，报错时容易定位。
- Each step has a log, making errors easier to locate.
- 报告里可以直接引用每一步的输出表和图。
- The report can directly cite each step's output tables and figures.
