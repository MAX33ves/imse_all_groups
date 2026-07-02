# Bike-Type Classifier Report / 单车类型分类模型报告

## Purpose / 这个模型是给什么用的

This model predicts the bike type, `FAT`, `ISY`, or `MTB`, from Sagemotion signal features.
这个模型用 Sagemotion 信号特征预测单车类型：`FAT`、`ISY` 或 `MTB`。

It is separate from the tire-pressure regressor because bike type is a classification target, while pressure is a continuous regression target.
它和胎压回归模型分开，因为单车类型是分类目标，而胎压是连续回归目标。

## Leakage-Safe Inputs / 防止信息泄漏的输入规则

The classifier deliberately does not use the following fields as inputs:
分类器刻意不使用以下字段作为输入：

- `bike`: this is the label being predicted. / 这是要预测的标签。
- `pressure_bar`: this would leak the tire-pressure label and is not known at prediction time. / 这会泄漏胎压标签，而且真实预测时也不知道。
- `p_number`, `group`, `run_id`, and file name: these are metadata and split/audit fields. / 这些是元数据、划分字段或审计字段。
- `rider_weight_kg`: this can encode rider/group context rather than bike structure. / 这可能编码骑手或 group 上下文，而不是单车结构。

Allowed inputs are Sagemotion acceleration and gyroscope signal features only.
允许使用的输入只有 Sagemotion 加速度和陀螺仪信号特征。

## Validation Design / 验证设计

- Validation method: `leave_one_group_out_cv` / 验证方法：`leave_one_group_out_cv`
- Each fold holds out one complete group and trains on the other five groups. / 每一折留出一个完整 group，用其余五个 group 训练。
- Window-level probabilities are averaged to make one run-level bike prediction. / 窗口级概率先取平均，再得到一个 run-level 单车类型预测。

## Selected Model / 选中模型

- Model: `bike_type_ffnn_ensemble_ens3_signal_full_pca10_tanh_h6_a1` / 模型：`bike_type_ffnn_ensemble_ens3_signal_full_pca10_tanh_h6_a1`
- Feature space: `signal_full` / 特征空间：`signal_full`
- Pre-PCA inputs: 128 / PCA 前输入数：128
- PCA components: 10 / PCA 维度：10
- Hidden layers: (6,) / 隐藏层：(6,)
- Seeds: 3 / 随机种子成员数：3

## CV Performance / CV 表现

| Metric / 指标 | Value / 数值 |
|---|---:|
| Run-level accuracy | 0.986 |
| Macro-F1 | 0.986 |
| Minimum group accuracy | 0.917 |
| Mean confidence | 0.848 |

## Performance By Bike / 按单车类型看的表现

| Bike / 单车 | n runs | accuracy | mean confidence |
|---|---:|---:|---:|
| FAT | 24 | 0.958 | 0.841 |
| ISY | 24 | 1.000 | 0.853 |
| MTB | 24 | 1.000 | 0.851 |

## Top Candidate Models / 候选模型前 10

| Model / 模型 | Feature space | CV accuracy | CV macro-F1 | score |
|---|---|---:|---:|---:|
| `bike_type_ffnn_ensemble_ens3_signal_full_pca10_tanh_h6_a1` | signal_full | 0.986 | 0.986 | 0.020 |
| `bike_type_ffnn_stage1_s42_signal_full_pca10_tanh_h6_a1` | signal_full | 0.986 | 0.986 | 0.020 |
| `bike_type_ffnn_ensemble_ens3_signal_full_pca10_tanh_h8x4_a1` | signal_full | 0.986 | 0.986 | 0.020 |
| `bike_type_ffnn_stage1_s42_signal_full_pca10_tanh_h8x4_a1` | signal_full | 0.986 | 0.986 | 0.020 |
| `bike_type_ffnn_ensemble_ens3_signal_compact_pca10_tanh_h12x6_a1` | signal_compact | 0.986 | 0.986 | 0.020 |
| `bike_type_ffnn_ensemble_ens3_signal_full_pca10_tanh_h12x6_a1` | signal_full | 0.986 | 0.986 | 0.020 |
| `bike_type_ffnn_stage1_s42_signal_full_pca10_tanh_h12x6_a1` | signal_full | 0.986 | 0.986 | 0.020 |
| `bike_type_ffnn_stage1_s42_signal_compact_pca10_tanh_h12x6_a1` | signal_compact | 0.972 | 0.972 | 0.040 |
| `bike_type_ffnn_stage1_s42_signal_compact_pca10_tanh_h6_a1` | signal_compact | 0.958 | 0.959 | 0.050 |
| `bike_type_ffnn_stage1_s42_signal_compact_pca6_tanh_h6_a1` | signal_compact | 0.931 | 0.931 | 0.085 |

## How This Connects To The Pressure Model / 和胎压模型的关系

The existing pressure regressor can still be reported as the known-bike-context model. If the final system must infer bike type automatically, this classifier should run first and its predicted bike type can be passed into a downstream pressure model or compared with a pressure model that does not use bike type.
现有胎压回归模型仍可作为 known-bike-context 模型报告。如果最终系统必须自动推断单车类型，则应先运行这个分类器，再把预测出的 bike type 传给下游胎压模型，或与不使用 bike type 的胎压模型进行对照。