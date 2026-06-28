# Bilingual Quick Brief：P1-P4 全量训练版

## 1. 当前数据规则 / Current Data Rule

老师手里还有隐藏测试数据，所以本地已有的 G01-G06、P1-P4 全部作为训练池使用。我们不再把 P3/P4 当成本地 final test。

Since the instructor has an additional hidden test set, all observed G01-G06 P1-P4 runs are used as the local training pool. P3/P4 are no longer treated as a local final test set.

## 2. 项目目标 / Project Goal

这是一个回归任务，目标是根据骑行过程中的传感器信号、单车类型和 rider weight 预测连续胎压 `pressure_bar`。

This is a regression task. The goal is to predict continuous tire pressure `pressure_bar` from riding sensor features, bike type, and rider weight.

## 3. 数据处理 / Data Processing

Step 00-01 检查环境、生成 72 个 run 的标签表，并确认 72 个 Sagemotion CSV 和 72 个 PhyPhox XLS 文件都存在。

Steps 00-01 check the environment, build the label table for 72 runs, and verify that all 72 Sagemotion CSV files and 72 PhyPhox XLS files exist.

Step 02 读取 Sagemotion 数据，清洗传感器列，裁剪 active window，然后用 1 秒窗口和 50% overlap 提取时域、频域和传感器组合特征。

Step 02 reads Sagemotion data, cleans sensor columns, crops the active window, and extracts time-domain, frequency-domain, and sensor-pair features using 1-second windows with 50% overlap.

## 4. EDA 和 PCA / EDA And PCA

Step 03 生成数据分布图、数量检查图、PCA 特征空间图和特征-胎压关系图。这里的二维 PCA 是 EDA 可视化，用来观察全部本地训练池在低维空间中的分布；Step 04 的最终模型另外使用 `PCA(n_components=6)` 作为建模 pipeline 的降维步骤。模型泛化仍由 leave-one-group-out CV 估计。

Step 03 generates pressure-design plots, count checks, a PCA feature-space plot, and feature-vs-pressure plots. The two-dimensional PCA in Step 03 is for EDA visualization, while the final model in Step 04 separately uses `PCA(n_components=6)` inside the modeling pipeline. Model generalization is still estimated by leave-one-group-out CV.

## 5. 模型验证 / Model Validation

Step 04 使用 leave-one-group-out cross-validation。每一折留出一个完整 group 做 validation，其余 5 个 groups 训练，避免随机窗口划分造成同一个组的数据泄漏到训练和验证两边。

Step 04 uses leave-one-group-out cross-validation. Each fold holds out one full group for validation and trains on the other five groups, avoiding leakage that could happen if random windows from the same group appeared in both training and validation.

特征解释使用 run-level 相关性矩阵和冗余检查作为辅助证据，但最终模型输入仍由物理意义、降维/正则控制和组级 CV 一起决定。

Feature explanation uses run-level correlations and redundancy checks as supporting evidence, but final model inputs are still justified by physical meaning, dimensionality/regularization control, and group-level CV together.

## 6. 选中模型 / Selected Model

选中模型是 `ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8x4_a1_median`。

The selected model is `ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8x4_a1_median`.

结构为 `StandardScaler -> PCA(n_components=6) -> MLPRegressor(hidden_layer_sizes=(8, 4), activation='tanh')`，输入空间为 compact 信号特征、bike one-hot 和 `rider_weight_kg`，并使用 3 个随机种子模型做 ensemble。窗口预测聚合到 run-level 时使用 median。

The structure is `StandardScaler -> PCA(n_components=6) -> MLPRegressor(hidden_layer_sizes=(8, 4), activation='tanh')`, using compact signal features, bike one-hot features, and `rider_weight_kg`, with a 3-seed ensemble. Window-level predictions are aggregated to run-level predictions using the median.

## 7. 当前效果 / Current Performance

留一组交叉验证的 run-level CV MAE 为 0.290 bar，CV RMSE 为 0.388 bar。最终模型用全部 72 个 run 重新训练，训练池拟合 MAE 为 0.226 bar，但这个训练拟合误差不能当作测试误差。

The leave-one-group-out run-level CV MAE is 0.290 bar, and the CV RMSE is 0.388 bar. The final model is retrained on all 72 runs, with a training-pool fit MAE of 0.226 bar, but this training-fit error should not be interpreted as test error.

目前 FAT 和 ISY 的误差较低，MTB 的误差明显更大，是后续改进的主要方向。

FAT and ISY currently have lower errors, while MTB has much larger errors and is the main direction for future improvement.

## 8. 推荐复跑方式 / Recommended Re-run Method

在 VSCode 打开 `C:\Users\user\Desktop\IMSE\imse_all_groups`，运行任务 `Training Pool Run Full Pipeline`。也可以按照 README 逐步运行 Step 00 到 Step 07。Step 07 会生成老师视角审查补充报告。

Open `C:\Users\user\Desktop\IMSE\imse_all_groups` in VSCode and run the task `Training Pool Run Full Pipeline`. You can also follow the README and run Steps 00 to 07 one by one. Step 07 generates the teacher-review supplement.
