# Final FFNN Training Model / 最终 FFNN 训练模型

## Purpose / 这个模型是给什么用的

The instructor still has hidden test data, so the final model is retrained on all local G01-G06, P1-P4 data. This final model no longer holds out a local final test set; local generalization is estimated by the previous leave-one-group-out CV step.  
老师手里还有隐藏测试数据，所以最终模型使用本地全部 G01-G06、P1-P4 数据重新训练。这个模型不再留出本地 final test；本地泛化能力由上一步 leave-one-group-out CV 估计。

## Final Training Data / 最终训练数据

| Item / 项目 | Value / 数值 |
|---|---:|
| Local training runs / 本地训练 run | 72 |
| Window samples / 窗口样本 | 873 |
| Target / 目标 | Continuous tire pressure `pressure_bar` / 连续胎压 `pressure_bar` |

## Selected Structure / 选中结构

- Model: `ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8x4_a1_median` / 模型：`ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8x4_a1_median`
- Uses rider weight: True / 是否使用 rider weight：True
- Selection rule: use leave-one-group-out CV on all local P1-P4 runs; the final selected FFNN must include `rider_weight_kg`; among eligible models, prefer a simpler model within 5% of the best FFNN CV score. / 选择规则：在全部本地 P1-P4 run 上使用 leave-one-group-out CV；最终选中的 FFNN 必须包含 `rider_weight_kg`；在合格模型中优先选择距离最佳 FFNN CV score 5% 以内、更简单的模型。
- Pre-PCA inputs: 28 / PCA 前输入变量数：28
- PCA components: 6 / PCA 维度：6
- Hidden layers: (8, 4) / 隐藏层：(8, 4)
- Output neuron: 1, predicting continuous `pressure_bar` / 输出神经元：1，输出连续胎压 `pressure_bar`
- Activation: tanh / 激活函数：tanh
- L2 regularization alpha: 1.0 / L2 正则 alpha：1.0
- Training implementation: `MLPRegressor(solver='lbfgs', max_iter=3000, max_fun=40000)`. Therefore, training is recorded by optimizer iterations/function-call limits rather than a fixed epoch count. / 训练实现：`MLPRegressor(solver='lbfgs', max_iter=3000, max_fun=40000)`。因此记录优化器迭代/函数调用上限，而不是固定 epoch 数。

## Local Validation Performance From CV / 本地验证效果（来自上一步 CV）

| Metric / 指标 | Value / 数值 |
|---|---:|
| CV MAE | 0.290 bar |
| CV RMSE | 0.388 bar |
| CV nearest-level accuracy | 0.597 |
| CV macro-F1 | 0.466 |

## Training-Pool Fit After Full Retraining / 全量训练后的训练集拟合效果

| Metric / 指标 | Value / 数值 |
|---|---:|
| Training-pool fit MAE | 0.226 bar |
| Training-pool fit RMSE | 0.283 bar |

Training-pool fit is a fit-to-training-data result, not a generalization estimate. Reports should cite the CV metrics when discussing local generalization.  
Training-pool fit 是训练集拟合效果，不是泛化效果；报告里讨论本地泛化时更应该引用 CV 指标。

## Hidden Test Set Note / 隐藏测试集说明

- The instructor hidden test set is not in the current workspace, so hidden-test MAE/RMSE cannot be reported yet. / 老师隐藏测试集不在当前 workspace 中，所以现在不能报告 hidden-test MAE/RMSE。
- The saved `.pkl` is the final local model for later hidden-test processing with the same raw-signal processing, feature extraction, scaling, PCA, and FFNN prediction steps. / 当前 `.pkl` 保存的是最终本地模型，用于后续对隐藏数据做同样的原始信号处理、特征提取、标准化/PCA/FFNN 预测。
- If a report or presentation mentions generalization, cite leave-one-group-out CV; true external generalization requires the instructor hidden-test result. / 如果报告或答辩提到 generalization，应引用 leave-one-group-out CV；真正外部泛化需要等老师隐藏测试结果。
