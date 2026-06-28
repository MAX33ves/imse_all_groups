# IMSE Bicycle Tire-Pressure Requirement Matrix / IMSE 自行车胎压项目要求矩阵

P1-P4 full training-pool version.  
P1-P4 全量训练版。

## Current Execution Scope / 当前执行口径

| Item / 项目 | Current implementation / 当前执行口径 |
|---|---|
| Data scope / 数据范围 | All groups in `Measurement_Campaign/G01-G06` / 使用 `Measurement_Campaign/G01-G06` 全部组数据 |
| Local training data / 本地训练数据 | All P1-P4 runs, 72 runs in total / 所有 P1-P4，共 72 个 run |
| Local validation / 本地验证 | leave-one-group-out cross-validation / 留一组交叉验证 |
| External test / 外部测试 | Instructor hidden test data / 老师隐藏测试数据 |
| Label source / 标签来源 | True pressure, rider weight, and ride time from `Measurement Details.pdf` / `Measurement Details.pdf` 中真实胎压、骑手质量、骑行时间 |
| Prediction target / 预测目标 | Continuous tire pressure `pressure_bar` / 连续胎压 `pressure_bar` |

## Course Requirements And Implementation / 课程要求与当前实现

| Requirement / 要求 | Current implementation / 当前实现 |
|---|---|
| Data preprocessing / 数据预处理 | Active-window cropping, 1 s windows, missing-value handling / active window 裁剪、1 秒窗口切片、缺失值处理 |
| Signal processing / 信号处理 | Acceleration/angular-velocity magnitude, band-pass filtering, FFT frequency features, time-domain statistics / 加速度/角速度模长、带通滤波、FFT 频域特征、时域统计特征 |
| PCA | EDA PCA figure and PCA reduction inside model candidates / EDA PCA 图；模型候选中使用 PCA 降维 |
| FFNN | Fully connected feed-forward regression network via `sklearn.neural_network.MLPRegressor` / 使用 `sklearn.neural_network.MLPRegressor` 实现全连接前馈回归网络 |
| Cross-validation / 交叉验证 | Leave-one-group-out CV to avoid window leakage / 按 group 留一交叉验证，避免窗口泄漏 |
| Multiple metrics / 多指标评价 | MAE, RMSE, bias, nearest-level accuracy, macro-F1, confusion matrix / MAE、RMSE、bias、nearest-level accuracy、macro-F1、混淆矩阵 |
| Final model / 最终模型 | Retrained on all 72 local runs and saved as `.pkl` for instructor hidden testing / 用全部 72 个本地 run 训练，保存为 `.pkl`，用于老师隐藏测试 |

## Errors To Avoid / 必须避免的错误

| Error / 错误 | Correct handling / 正确处理 |
|---|---|
| Treating P3/P4 as a local final test / 把 P3/P4 当作本地 final test | Use all P1-P4 runs in the training pool / P1-P4 全部进入训练池 |
| Randomly splitting windows for train/validation / 随机窗口划分训练/验证 | Use leave-one-group-out CV / 使用 leave-one-group-out |
| Treating P number as the pressure label / 把 P 编号当胎压标签 | Use true `pressure_bar` from Measurement Details / 使用 Measurement Details 中真实 `pressure_bar` |
| Feeding `group` or `p_number` into the model / 把 `group` 或 `p_number` 放进输入 | Use these fields only for splitting and audit / 这些字段只用于划分和审计，不进入模型 |
| Reporting training-fit error as generalization / 用训练集拟合误差冒充泛化效果 | Report CV metrics for local generalization / 报告泛化效果时引用 CV 指标 |
