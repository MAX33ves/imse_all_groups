# FFNN Group-Level Cross-Validation And Model Selection / FFNN 组级交叉验证与模型选择

## Why There Is No Local P3/P4 Final Test / 为什么现在没有本地 P3/P4 final test

The instructor still has hidden test data, so all locally observed P1-P4 runs can be used for training. To still estimate local generalization, this project uses group-level cross-validation: each fold holds out one complete group as validation and trains on the other five groups.  
现在已确认老师手里还有隐藏测试数据，所以本地观测到的 P1-P4 全部都可以用于训练。为了仍然估计本地泛化能力，本项目使用组级交叉验证：每次留出一个完整小组作为 validation，其余 5 个组训练。

## Validation Split / 验证划分

- Fold 1: train = G02-G06, validation = G01.
- Fold 2: train = G01,G03-G06, validation = G02.
- The pattern continues until every group has been validated once. / 以此类推，直到每个组都被验证一次。
- Each validation fold contains 12 runs from that group: FAT/ISY/MTB with P1-P4. / 每个验证 fold 包含该组的 12 个 run，也就是 FAT/ISY/MTB 的 P1-P4。

## Model Inputs / 模型输入

- Main inputs: Sagemotion signal window features + bike type one-hot + rider weight. / 主要输入：Sagemotion 信号窗口特征 + bike type one-hot + rider weight。
- Control comparison: feature spaces without rider weight are still compared, but the final model must include rider weight. / 对照输入：不含 rider weight 的特征空间仍参与比较，但最终模型必须使用 rider weight。
- Forbidden inputs: `group`, `p_number`, `run_id`, and the true pressure label. / 禁止输入：`group`、`p_number`、`run_id`、真实胎压标签。

## Regression Task And Auxiliary Classification Metrics / 回归任务和辅助分类指标

The main task is regression: the model outputs one continuous value, `pressure_bar`.  
主任务是回归：模型输出 1 个连续值 `pressure_bar`。

Because the course task also asks for classification-style metrics such as confusion matrix and F-score, each run-level continuous prediction is mapped to the nearest valid pressure level for that bike. The nearest-level accuracy, macro-F1, and confusion matrix are reported as auxiliary interpretation metrics.  
课程任务也要求 confusion matrix / F-score 这类分类指标，所以本项目把每个 run 的连续预测映射到该 bike 的最近可行胎压等级，再计算 nearest-level accuracy、macro-F1 和混淆矩阵。

Therefore, MAE/RMSE are the main generalization metrics. The confusion matrix and macro-F1 do not mean the model has become a pure classifier.  
因此 MAE/RMSE 是主要泛化指标；混淆矩阵和 macro-F1 是辅助解释指标，不代表模型变成了纯分类器。

## Selected Model / 选中模型

| Item / 项目 | Value / 数值 |
|---|---|
| Model / 模型 | `ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8x4_a1_median` |
| CV MAE | 0.290 bar |
| CV RMSE | 0.388 bar |
| CV bias | 0.031 bar |
| Nearest-level accuracy | 0.597 |
| Macro-F1 | 0.466 |

## Network Structure / 网络结构记录

| Component / 组件 | Value / 数值 |
|---|---|
| Pre-PCA inputs / PCA 前输入变量数 | 28 |
| PCA components / 标准化后 PCA 维度 | 6 |
| FFNN hidden layers / FFNN 隐藏层 | (8, 4) |
| Output neuron / 输出神经元 | 1, predicting continuous `pressure_bar` / 1，输出连续胎压 `pressure_bar` |
| Activation / 激活函数 | tanh |
| L2 regularization alpha / L2 正则 alpha | 1.0 |
| Solver / 求解器 | `MLPRegressor(solver='lbfgs', max_iter=3000, max_fun=40000)` |

The `lbfgs` solver is recorded by optimizer iteration/function-call limits, not by a fixed epoch schedule.  
`lbfgs` 求解器记录的是优化器迭代/函数调用上限，而不是固定 epoch 数。

## Model-Selection Caveat / 模型选择注意事项

- The same leave-one-group-out CV is used both to compare candidates and to report the selected model metrics; therefore, these CV metrics are local post-selection generalization estimates, not instructor hidden-test results. / 同一个 leave-one-group-out CV 同时用于候选模型比较和选中模型指标报告，所以这里的 CV 指标是本地模型选择后的泛化估计，不是老师隐藏测试集结果。
- A strictly unbiased nested CV would need more independent groups. With only 6 groups, the project instead reports candidate comparisons and held-out group errors transparently. / 严格无偏的 nested CV 需要更多独立 group；当前只有 6 个 group，所以采用透明报告候选模型和 held-out group 误差的方式控制风险。
- The best FFNN without rider weight is slightly better on CV score, but the selected rider-weight model is within the 5% near-best range and satisfies the project rule. Rider weight should be described as physical load context, not as a proven strong direct pressure predictor. / 最优 FFNN 不含体重，含体重模型在 near-best 5% 范围内并满足当前规则。报告中应把 rider weight 表述为物理载荷上下文和课程约束，而不是声称它显著提升了 MAE。

## Rider-Weight Sensitivity / 体重输入敏感性

| Role / 角色 | Model / 模型 | CV score | CV MAE | CV RMSE | Nearest-level accuracy |
|---|---|---:|---:|---:|---:|
| Best without rider weight / 不含体重的最佳模型 | `ffnn_training_pool_ensemble_ens3_compact_pca6_tanh_h8x4_a1_median` | 0.380 | 0.286 | 0.383 | 0.569 |
| Best with rider weight / 含体重的最佳模型 | `ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8x4_a1_median` | 0.389 | 0.290 | 0.388 | 0.597 |

## Top 10 Candidate Models / 候选模型前 10

| Model / 模型 | Type | CV score | CV MAE | CV RMSE | Accuracy |
|---|---|---:|---:|---:|---:|
| `ffnn_training_pool_ensemble_ens3_compact_pca6_tanh_h8x4_a1_median` | ffnn | 0.380 | 0.286 | 0.383 | 0.569 |
| `ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8x4_a1_median` | ffnn | 0.389 | 0.290 | 0.388 | 0.597 |
| `ffnn_training_pool_stage1_s42_compact_weight_pca6_tanh_h8x4_a1_median` | ffnn | 0.406 | 0.303 | 0.410 | 0.569 |
| `ffnn_training_pool_ensemble_ens3_compact_pca6_tanh_h8_a0p1_median` | ffnn | 0.418 | 0.310 | 0.427 | 0.583 |
| `ffnn_training_pool_stage1_s42_compact_pca6_tanh_h8x4_a1_median` | ffnn | 0.419 | 0.300 | 0.412 | 0.542 |
| `ffnn_training_pool_stage1_s42_compact_pca6_tanh_h4_a1_median` | ffnn | 0.419 | 0.315 | 0.416 | 0.542 |
| `ffnn_training_pool_ensemble_ens3_compact_pca6_tanh_h4_a1_median` | ffnn | 0.424 | 0.319 | 0.434 | 0.542 |
| `ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8_a0p1_median` | ffnn | 0.433 | 0.326 | 0.440 | 0.542 |
| `ffnn_training_pool_stage1_s42_compact_pca6_tanh_h8_a0p1_median` | ffnn | 0.437 | 0.328 | 0.451 | 0.611 |
| `ffnn_training_pool_stage1_s42_compact_pca6_tanh_h8_a1_median` | ffnn | 0.442 | 0.327 | 0.441 | 0.569 |

## Selected-Model Validation Predictions / 选中模型的验证预测

The full selected-model CV prediction table is stored in `03_outputs/tables/training_pool_ffnn_selected_cv_predictions.csv`.  
完整的选中模型 CV 预测表保存在 `03_outputs/tables/training_pool_ffnn_selected_cv_predictions.csv`。

The largest errors are discussed in the teacher-review supplement and summarized by bike/group in the output figures.  
最大误差 run 已在老师视角审查补充报告中讨论，并在输出图中按 bike/group 总结。
