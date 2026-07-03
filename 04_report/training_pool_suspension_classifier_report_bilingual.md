# Suspension-Type Classifier Report / 悬挂类型分类模型报告

## Purpose / 这个模型是给什么用的

This model predicts the suspension or damping category from Sagemotion signal features.
这个模型用 Sagemotion 信号特征预测悬挂/阻尼类型。

The supervised labels are derived from the course table that maps each bike type to a suspension type.
监督标签来自课程表中“单车类型 -> 悬挂类型”的对应关系。

| Bike context / 单车背景 | Suspension label / 悬挂标签 |
|---|---|
| FAT | Suspension because of tyres |
| ISY | No Suspension |
| MTB | Front and rear Suspension |

## Leakage-Safe Inputs / 防止信息泄漏的输入规则

The classifier deliberately does not use the following fields as inputs:
分类器刻意不使用以下字段作为输入：

- `bike`: this would directly reveal the mapped suspension label. / 它会直接泄漏映射后的悬挂标签。
- `suspension_type`: this is the label being predicted. / 这是要预测的标签。
- `pressure_bar`: this would leak the tire-pressure label and is not known at prediction time. / 这会泄漏胎压标签，而且真实预测时也不知道。
- `p_number`, `group`, `run_id`, and file name: these are metadata and split/audit fields. / 这些是元数据、划分字段或审计字段。
- `rider_weight_kg`: this can encode rider/group context rather than bike structure. / 这可能编码骑手或 group 上下文，而不是单车结构。

Allowed inputs are Sagemotion acceleration and gyroscope signal features only.
允许使用的输入只有 Sagemotion 加速度和陀螺仪信号特征。

## Validation Design / 验证设计

- Validation method: `leave_one_group_out_cv` / 验证方法：`leave_one_group_out_cv`
- Each fold holds out one complete group and trains on the other five groups. / 每一折留出一个完整 group，用其余五个 group 训练。
- Window-level probabilities are averaged to make one run-level suspension prediction. / 窗口级概率先取平均，再得到一个 run-level 悬挂类型预测。

## Selected Model / 选中模型

- Model: `suspension_ffnn_ensemble_ens3_signal_full_pca10_tanh_h6_a1` / 模型：`suspension_ffnn_ensemble_ens3_signal_full_pca10_tanh_h6_a1`
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
| Mean confidence | 0.841 |

## Performance By Suspension Type / 按悬挂类型看的表现

| Suspension type / 悬挂类型 | n runs | accuracy | mean confidence |
|---|---:|---:|---:|
| Front and rear Suspension | 24 | 1.000 | 0.847 |
| No Suspension | 24 | 1.000 | 0.847 |
| Suspension because of tyres | 24 | 0.958 | 0.829 |

## Top Candidate Models / 候选模型前 10

| Model / 模型 | Feature space | CV accuracy | CV macro-F1 | score |
|---|---|---:|---:|---:|
| `suspension_ffnn_ensemble_ens3_signal_full_pca10_tanh_h6_a1` | signal_full | 0.986 | 0.986 | 0.020 |
| `suspension_ffnn_stage1_s42_signal_full_pca10_tanh_h6_a1` | signal_full | 0.986 | 0.986 | 0.020 |
| `suspension_ffnn_ensemble_ens3_signal_full_pca10_tanh_h8x4_a1` | signal_full | 0.986 | 0.986 | 0.020 |
| `suspension_ffnn_stage1_s42_signal_full_pca10_tanh_h8x4_a1` | signal_full | 0.986 | 0.986 | 0.020 |
| `suspension_ffnn_ensemble_ens3_signal_full_pca10_tanh_h12x6_a1` | signal_full | 0.986 | 0.986 | 0.020 |
| `suspension_ffnn_stage1_s42_signal_full_pca10_tanh_h12x6_a1` | signal_full | 0.986 | 0.986 | 0.020 |
| `suspension_ffnn_ensemble_ens3_signal_compact_pca10_tanh_h8x4_a1` | signal_compact | 0.972 | 0.972 | 0.035 |
| `suspension_ffnn_stage1_s42_signal_compact_pca10_tanh_h8x4_a1` | signal_compact | 0.972 | 0.972 | 0.035 |
| `suspension_ffnn_stage1_s42_signal_compact_pca10_tanh_h12x6_a1` | signal_compact | 0.972 | 0.972 | 0.035 |
| `suspension_ffnn_stage1_s42_signal_compact_pca10_tanh_h6_a1` | signal_compact | 0.958 | 0.959 | 0.050 |

## How This Connects To The Pressure Model / 和胎压模型的关系

This classifier addresses the teacher's suspension/damping requirement. The pressure regressor remains a separate model because tire pressure is a continuous regression target.
这个分类器对应老师提出的悬挂/阻尼类型任务。胎压回归模型仍然是单独模型，因为胎压是连续回归目标。