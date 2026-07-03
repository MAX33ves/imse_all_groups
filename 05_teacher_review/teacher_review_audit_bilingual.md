# Teacher-Review Supplement / 老师视角审查补充报告

## Overall Judgment / 总体判断

The main project logic is sound: local G01-G06, P1-P4 data provide 72 runs for the training pool; because the instructor still has a hidden test set, local generalization is estimated with leave-one-group-out cross-validation; the final model is then retrained on all local runs.
当前项目的主逻辑是成立的：本地 G01-G06、P1-P4 共 72 个 run 作为训练池；由于老师还有隐藏测试集，本地泛化用 leave-one-group-out cross-validation 估计；最终模型再用全部本地 run 训练。

The report still needs to be explicit about several scope and logic points: why PhyPhox is not included in the current model matrix, how the FFNN architecture satisfies the course requirement, why CV metrics are not hidden-test results, why rider weight is physical context rather than a strong direct label proxy, and why the damping/suspension task is modeled as a categorical label from the course table rather than as a continuous damping coefficient.
但报告层仍有几处需要讲得更清楚：PhyPhox 数据为什么没有入模、FFNN 架构如何对应课程要求、CV 指标不是隐藏测试结果、体重是物理上下文而不是强相关标签替代物、damping/suspension 为什么按课程表中的类别标签建模，而不是按连续阻尼系数建模。

## Data And Output Checks / 数据和输出复核

| check | status | evidence | implication |
| --- | --- | --- | --- |
| label coverage | pass | 72 labeled runs across 6 groups, 3 bike types. | The local training pool matches the stated 72-run design. |
| raw file coverage | pass | 72 Sagemotion CSV and 72 PhyPhox XLS files parse successfully. | Raw files are present; current model uses Sagemotion only, so PhyPhox needs an explicit scope note. |
| feature table finite values | pass | 873 window rows, 0 NaN cells, 0 infinite numeric cells. | No actual CV leakage from full-data median filling was observed in the current extracted feature table. |
| active-window duration | pass | max \|active_duration_s - table_ride_time_s\| = 0.000000 s. | The crop length follows Measurement Details exactly; start location is still energy-based and should be described as an assumption. |
| sample rate consistency | pass | sample_rate_hz min/median/max = 100.0/100.0/100.0. | A fixed 1 s window corresponds to 100 samples for every run. |
| selected CV recomputation | pass | MAE=0.290 bar, RMSE=0.388 bar, bias=0.031 bar over 72 run predictions. | The report metrics are reproducible from the selected CV prediction table. |
| rider-weight direct target correlation | caveat | global Spearman=0.011; selected model uses rider_weight_kg=True. | Weight is physically justified as load context, but the current data does not prove a strong direct pressure correlation. |

## Error Structure / 误差结构

By bike, MTB is the main error source:

按 bike 看，MTB 是主要误差来源：

| bike | n_runs | mae_bar | rmse_bar | max_abs_error_bar | mean_window_pred_std |
| --- | --- | --- | --- | --- | --- |
| MTB | 24 | 0.532 | 0.591 | 1.121 | 0.436 |
| ISY | 24 | 0.203 | 0.281 | 0.894 | 0.377 |
| FAT | 24 | 0.136 | 0.153 | 0.255 | 0.082 |

By held-out group, no single group fully fails, but G03/G04/G06 are slightly higher:

按 held-out group 看，没有单个 group 完全失控，但 G03/G04/G06 略高：

| group | n_runs | mae_bar | rmse_bar | max_abs_error_bar |
| --- | --- | --- | --- | --- |
| G03 | 12 | 0.321 | 0.431 | 0.894 |
| G06 | 12 | 0.315 | 0.431 | 1.085 |
| G04 | 12 | 0.312 | 0.429 | 1.121 |
| G05 | 12 | 0.293 | 0.372 | 0.703 |
| G02 | 12 | 0.253 | 0.321 | 0.812 |
| G01 | 12 | 0.247 | 0.325 | 0.787 |

Largest-error runs:

最大误差 run：

| run_id | fold | bike | group | actual_pressure_bar | pred_pressure_bar | abs_error_bar | n_windows | pred_pressure_window_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G04_MTB_P4 | G04 | MTB | G04 | 1.000 | 2.121 | 1.121 | 12 | 0.462 |
| G06_MTB_P4 | G06 | MTB | G06 | 1.000 | 2.085 | 1.085 | 10 | 0.366 |
| G03_ISY_P2 | G03 | ISY | G03 | 3.000 | 2.106 | 0.894 | 15 | 0.690 |
| G03_MTB_P4 | G03 | MTB | G03 | 1.000 | 1.845 | 0.845 | 11 | 0.550 |
| G02_MTB_P1 | G02 | MTB | G02 | 2.000 | 2.812 | 0.812 | 10 | 0.416 |
| G01_MTB_P2 | G01 | MTB | G01 | 3.000 | 2.213 | 0.787 | 13 | 0.450 |
| G05_MTB_P4 | G05 | MTB | G05 | 1.000 | 1.703 | 0.703 | 11 | 0.478 |
| G05_MTB_P1 | G05 | MTB | G05 | 2.000 | 2.635 | 0.635 | 12 | 0.228 |

## Model-Selection Sensitivity / 模型选择敏感性

The best FFNN without rider weight is slightly better, but the selected model includes rider weight and remains within the 5% near-best range. Therefore, rider weight should be described as a physically required context input, not as a variable that clearly improves CV MAE.

最优 FFNN 不含体重，选中模型含体重且在 5% near-best 范围内。因此报告应把体重表述为课程/物理约束下的上下文变量，而不是声称它显著提升了 CV MAE。

| review_role | model_name | feature_space | n_features_in | pca_components | hidden_layers | alpha | cv_selection_score | cv_run_mae_bar | cv_run_rmse_bar | cv_run_bias_bar | cv_run_max_abs_error_bar | cv_nearest_level_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_ffnn_without_weight | ffnn_training_pool_ensemble_ens3_compact_pca6_tanh_h8x4_a1_median | compact | 27 | 6.000 | (8, 4) | 1.000 | 0.380 | 0.286 | 0.383 | 0.030 | 1.049 | 0.569 |
| best_ffnn_with_weight | ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8x4_a1_median | compact_weight | 28 | 6.000 | (8, 4) | 1.000 | 0.389 | 0.290 | 0.388 | 0.031 | 1.121 | 0.597 |
| selected_model | ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8x4_a1_median | compact_weight | 28 | 6.000 | (8, 4) | 1.000 | 0.389 | 0.290 | 0.388 | 0.031 | 1.121 | 0.597 |

## Requirement Gap Matrix / 课程要求缺口矩阵

| requirement_or_question | current_evidence | gap_or_caveat | recommended_text_or_action |
| --- | --- | --- | --- |
| Train a fully connected feed-forward neural network. | MLPRegressor with hidden layers (8, 4), tanh activation, PCA(6) input, and one pressure output. | Report should explicitly state output neuron = 1 and solver is lbfgs, so classic epoch count is not the main training record. | Add a small architecture table: 28 pre-PCA inputs -> 6 PCA scores -> 8 -> 4 -> 1 pressure_bar output; max_iter=3000, max_fun=40000. |
| Cross-validate and review performance with multiple metrics. | Leave-one-group-out CV produces MAE, RMSE, bias, nearest-level accuracy, macro-F1, and confusion matrix. | Because the same CV is used for model selection, the reported CV is a local model-selection estimate, not an unbiased hidden-test estimate. | State this caveat and explain why nested CV is impractical with only 6 groups. |
| Test generalization ability on unseen generalization dataset. | Teacher hidden data is not in the workspace; final model is trained on all 72 local runs for that later test. | No hidden-test metrics can be reported before the instructor evaluates the model. | Write 'hidden-test performance is pending; local generalization is estimated by leave-one-group-out CV.' |
| Use measured acceleration data from sensors and PhyPhox app. | 72 Sagemotion CSV and 72 PhyPhox XLS files exist, but extracted model features use Sagemotion CSV only. | This is the largest scope explanation gap. | Either add a PhyPhox baseline/feature-ablation experiment or explicitly justify Sagemotion-only modeling as the higher-quality two-sensor + gyro source. |
| Predict tire pressures and damping of the bicycle. | Tire pressure is available as a continuous label. Damping/suspension is available as a categorical label through the course table: FAT -> Suspension because of tyres, ISY -> No Suspension, and MTB -> Front and rear Suspension. | No continuous damping coefficient is available, so Step 04b predicts the table-defined suspension category rather than a numeric damping value. | Report Step 04b as the suspension/damping classifier and state that bike, pressure, p-number, group, run id, file name, and rider weight are excluded from its inputs to avoid leakage. |

## Suggested Report Text / 建议直接补进报告的表述

1. `The model is a pressure regression model. For classification-style metrics required by the task sheet, each run-level pressure prediction is mapped to the nearest valid pressure level of the corresponding bike, and the confusion matrix / macro-F1 are reported as auxiliary metrics.`
   模型是胎压回归模型。为了满足课程任务中的分类式指标要求，每个 run-level 连续胎压预测会映射到对应 bike 的最近可行胎压等级，再报告 confusion matrix / macro-F1 作为辅助指标。
2. `The selected network uses 28 pre-PCA inputs, which are standardized and reduced to 6 PCA components before entering a fully connected network with hidden layers 8 and 4 and one output neuron for pressure_bar.`
   选中网络使用 28 个 PCA 前输入，先标准化并降到 6 个 PCA component，再进入隐藏层为 8 和 4、输出神经元为 1 的全连接网络。
3. `The MLP is trained with sklearn's lbfgs optimizer (max_iter=3000, max_fun=40000), so training is recorded by optimizer limits and convergence rather than a fixed epoch schedule.`
   MLP 使用 sklearn 的 lbfgs 优化器训练（max_iter=3000, max_fun=40000），因此训练记录重点是优化器上限和收敛，而不是固定 epoch 数。
4. `PhyPhox files were checked for completeness, but the current model uses Sagemotion because it provides two mounted sensors and gyroscope channels. PhyPhox should be treated as optional future validation unless a baseline experiment is added.`
   PhyPhox 文件已检查完整性，但当前模型使用 Sagemotion，因为它提供两个固定安装传感器和陀螺仪通道；除非补充 baseline 实验，否则 PhyPhox 应作为未来验证来源。
5. `The damping/suspension task is handled as categorical classification. The labels are derived from the course table: FAT -> Suspension because of tyres, ISY -> No Suspension, and MTB -> Front and rear Suspension. The classifier uses Sagemotion signal features only and excludes bike, pressure, p-number, group, run id, file name, and rider weight to avoid leakage.`
   Damping/suspension 任务按类别分类处理。标签来自课程表：FAT -> Suspension because of tyres，ISY -> No Suspension，MTB -> Front and rear Suspension。分类器只使用 Sagemotion 信号特征，并排除 bike、pressure、p-number、group、run id、file name 和 rider weight，以避免标签泄漏。