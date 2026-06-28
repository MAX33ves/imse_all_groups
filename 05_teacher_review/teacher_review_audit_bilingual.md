# Teacher-Review Supplement / 老师视角审查补充报告

## Overall Judgment / 总体判断

The main project logic is sound: local G01-G06, P1-P4 data provide 72 runs for the training pool; because the instructor still has a hidden test set, local generalization is estimated with leave-one-group-out cross-validation; the final model is then retrained on all local runs.  
当前项目的主逻辑是成立的：本地 G01-G06、P1-P4 共 72 个 run 作为训练池；由于老师还有隐藏测试集，本地泛化用 leave-one-group-out cross-validation 估计；最终模型再用全部本地 run 训练。

The report still needs to be explicit about several scope and logic points: why PhyPhox is not included in the current model matrix, how the FFNN architecture satisfies the course requirement, why CV metrics are not hidden-test results, why rider weight is physical context rather than a strong direct label proxy, and why damping is represented through vibration features rather than supervised as a separate label.  
但报告层仍需要把几件事讲清楚：PhyPhox 数据为什么没有入模、FFNN 架构如何对应课程要求、CV 指标为什么不是隐藏测试结果、体重为什么是物理上下文而不是强相关标签替代物、damping 为什么没有独立监督标签。

## Data And Output Checks / 数据和输出复核

| Check | Status | Evidence | Implication |
|---|---|---|---|
| label coverage | pass | 72 labeled runs across 6 groups and 3 bike types. | The local training pool matches the stated 72-run design. |
| raw file coverage | pass | 72 Sagemotion CSV and 72 PhyPhox XLS files parse successfully. | Raw files are present; current model uses Sagemotion only, so PhyPhox needs an explicit scope note. |
| feature table finite values | pass | 873 window rows, 0 NaN cells, 0 infinite numeric cells. | The extracted feature table is numerically valid for modeling. |
| active-window duration | pass | max absolute duration difference from Measurement Details is 0.000000 s. | The crop length follows Measurement Details exactly; crop start is still energy-based and should be described as an assumption. |
| sample rate consistency | pass | sample_rate_hz min/median/max = 100.0/100.0/100.0. | A fixed 1 s window corresponds to 100 samples for every run. |
| selected CV recomputation | pass | MAE=0.290 bar, RMSE=0.388 bar, bias=0.031 bar over 72 run predictions. | The report metrics are reproducible from the selected CV prediction table. |
| rider-weight direct target correlation | caveat | global Spearman=0.011; selected model uses `rider_weight_kg=True`. | Weight is physically justified as load context, but the current data does not prove a strong direct pressure correlation. |

## Error Structure / 误差结构

MTB is the main error source by bike.  
按 bike 看，MTB 是主要误差来源。

| Bike | n_runs | MAE bar | RMSE bar | Max abs error bar | Mean window prediction std |
|---|---:|---:|---:|---:|---:|
| MTB | 24 | 0.532 | 0.591 | 1.121 | 0.436 |
| ISY | 24 | 0.203 | 0.281 | 0.894 | 0.377 |
| FAT | 24 | 0.136 | 0.153 | 0.255 | 0.082 |

No held-out group completely fails, but G03/G04/G06 are slightly higher.  
按 held-out group 看，没有单个 group 完全失控，但 G03/G04/G06 略高。

| Group | n_runs | MAE bar | RMSE bar | Max abs error bar |
|---|---:|---:|---:|---:|
| G03 | 12 | 0.321 | 0.431 | 0.894 |
| G06 | 12 | 0.315 | 0.431 | 1.085 |
| G04 | 12 | 0.312 | 0.429 | 1.121 |
| G05 | 12 | 0.293 | 0.372 | 0.703 |
| G02 | 12 | 0.253 | 0.321 | 0.812 |
| G01 | 12 | 0.247 | 0.325 | 0.787 |

The largest-error runs are mostly MTB cases, especially low-pressure MTB P4.  
最大误差 run 大多来自 MTB，尤其是低胎压 MTB P4。

| Run | Fold | Bike | Actual bar | Predicted bar | Abs error bar |
|---|---|---|---:|---:|---:|
| G04_MTB_P4 | G04 | MTB | 1.000 | 2.121 | 1.121 |
| G06_MTB_P4 | G06 | MTB | 1.000 | 2.085 | 1.085 |
| G03_ISY_P2 | G03 | ISY | 3.000 | 2.106 | 0.894 |
| G03_MTB_P4 | G03 | MTB | 1.000 | 1.845 | 0.845 |
| G02_MTB_P1 | G02 | MTB | 2.000 | 2.812 | 0.812 |
| G01_MTB_P2 | G01 | MTB | 3.000 | 2.213 | 0.787 |
| G05_MTB_P4 | G05 | MTB | 1.000 | 1.703 | 0.703 |
| G05_MTB_P1 | G05 | MTB | 2.000 | 2.635 | 0.635 |

## Model-Selection Sensitivity / 模型选择敏感性

The best FFNN without rider weight is slightly better, but the selected model includes rider weight and remains within the 5% near-best range. Therefore, rider weight should be described as a physically required context input, not as a variable that clearly improves CV MAE.  
最优 FFNN 不含体重，选中模型含体重且在 5% near-best 范围内。因此报告应把体重表述为课程/物理约束下的上下文变量，而不是声称它显著提升了 CV MAE。

| Review role | Model | Feature space | Inputs | PCA | Hidden layers | Alpha | CV score | CV MAE | CV RMSE | Accuracy |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| best FFNN without weight | `ffnn_training_pool_ensemble_ens3_compact_pca6_tanh_h8x4_a1_median` | compact | 27 | 6 | (8, 4) | 1.0 | 0.380 | 0.286 | 0.383 | 0.569 |
| best FFNN with weight | `ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8x4_a1_median` | compact_weight | 28 | 6 | (8, 4) | 1.0 | 0.389 | 0.290 | 0.388 | 0.597 |
| selected model | `ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8x4_a1_median` | compact_weight | 28 | 6 | (8, 4) | 1.0 | 0.389 | 0.290 | 0.388 | 0.597 |

## Requirement Gap Matrix / 课程要求缺口矩阵

| Requirement or question | Current evidence | Gap or caveat | Recommended text or action |
|---|---|---|---|
| Train a fully connected feed-forward neural network. | MLPRegressor with hidden layers (8, 4), tanh activation, PCA(6) input, and one pressure output. | The report should explicitly state output neuron = 1 and solver = lbfgs. | Add the architecture chain: 28 pre-PCA inputs -> 6 PCA scores -> 8 -> 4 -> 1 pressure output. |
| Cross-validate and review performance with multiple metrics. | Leave-one-group-out CV produces MAE, RMSE, bias, nearest-level accuracy, macro-F1, and confusion matrix. | The same CV is used for model selection, so it is a local post-selection estimate. | State the caveat and explain why nested CV is impractical with only 6 groups. |
| Test generalization on unseen data. | Teacher hidden data is not in the workspace; final model is trained on all 72 local runs for that later test. | Hidden-test metrics cannot be reported before the instructor evaluates the model. | Write that hidden-test performance is pending and local generalization is estimated by leave-one-group-out CV. |
| Use measured acceleration data from sensors and PhyPhox app. | 72 Sagemotion CSV and 72 PhyPhox XLS files exist, but current extracted model features use Sagemotion CSV only. | This is the largest scope explanation gap. | Either add a PhyPhox baseline/feature-ablation experiment or explicitly justify Sagemotion-only modeling as the higher-quality two-sensor + gyro source. |
| Predict tire pressures and damping. | The available label table contains tire pressure, rider weight, and ride time; no continuous damping label is available. | Damping cannot be supervised as an output without labels. | Explain that damping-related behavior is represented through amplitude, energy, frequency-band, and spectral features while the supervised target is tire pressure. |

## Suggested Report Text / 建议直接补进报告的表述

1. The model is a pressure regression model. For classification-style metrics required by the task sheet, each run-level pressure prediction is mapped to the nearest valid pressure level of the corresponding bike, and the confusion matrix / macro-F1 are reported as auxiliary metrics.  
   模型是胎压回归模型。为了满足课程任务中的分类式指标要求，每个 run-level 连续胎压预测会映射到对应 bike 的最近可行胎压等级，再报告 confusion matrix / macro-F1 作为辅助指标。
2. The selected network uses 28 pre-PCA inputs, which are standardized and reduced to 6 PCA components before entering a fully connected network with hidden layers 8 and 4 and one output neuron for `pressure_bar`.  
   选中网络使用 28 个 PCA 前输入，先标准化并降到 6 个 PCA component，再进入隐藏层为 8 和 4、输出神经元为 1 的全连接网络。
3. The MLP is trained with sklearn's lbfgs optimizer (`max_iter=3000`, `max_fun=40000`), so training is recorded by optimizer limits and convergence rather than a fixed epoch schedule.  
   MLP 使用 sklearn 的 lbfgs 优化器训练（`max_iter=3000`, `max_fun=40000`），因此训练记录重点是优化器上限和收敛，而不是固定 epoch 数。
4. PhyPhox files were checked for completeness, but the current model uses Sagemotion because it provides two mounted sensors and gyroscope channels.  
   PhyPhox 文件已检查完整性，但当前模型使用 Sagemotion，因为它提供两个固定安装传感器和陀螺仪通道。
5. Damping is not a supervised target because no damping label is available in Measurement Details; damping-related behavior is represented through amplitude, energy, frequency-band, and spectral features.  
   Damping 不是监督学习目标，因为 Measurement Details 中没有 damping 标签；与阻尼相关的行为通过振幅、能量、频带和频谱特征表示。
