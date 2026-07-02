# Bike-Type Classifier / 单车类型分类模型

## What This Model Predicts / 这个模型预测什么

这个模型预测每个 run 属于哪一种单车：`FAT`、`ISY` 或 `MTB`。

This model predicts the bike class for each run: `FAT`, `ISY`, or `MTB`.

它和胎压模型不是同一个任务。胎压模型预测连续数值 `pressure_bar`，所以是 regression；单车类型模型预测三个类别之一，所以是 classification。

It is not the same task as tire-pressure prediction. The pressure model predicts the continuous value `pressure_bar`, so it is a regression task; the bike-type model predicts one of three classes, so it is a classification task.

## Why It Is Separate From The Pressure Model / 为什么和胎压模型分开

当前最清晰的结构是两个模型：

The clearest current structure uses two models:

| Model / 模型 | Target / 目标 | Output / 输出 |
|---|---|---|
| Pressure FFNN regressor / 胎压 FFNN 回归模型 | `pressure_bar` | continuous pressure in bar / 连续胎压数值 |
| Bike-type FFNN classifier / 单车类型 FFNN 分类模型 | `bike` | `FAT`, `ISY`, or `MTB` / 三类单车之一 |

分开的原因是目标类型不同、评价指标不同、可用输入也不同。压力模型可以在“已知单车类型”的设定下使用 bike one-hot；但分类模型绝对不能把 `bike` 当输入，因为 `bike` 本身就是要预测的答案。

The reason for separating them is that the target type, evaluation metrics, and allowed inputs differ. The pressure model can use bike one-hot under a known-bike-context assumption; the classifier must not use `bike` as an input because `bike` is the label being predicted.

如果最终系统不知道单车类型，可以先运行单车类型分类模型，再把预测出来的 bike type 交给下游胎压模型。更严格的扩展版本是再训练一个“不使用 bike type 的胎压模型”，用来和级联方案做对照。

If the final system does not know the bike type, the classifier can run first and pass the predicted bike type to a downstream pressure model. A stricter extension would also train a pressure model that does not use bike type, then compare it with the cascaded design.

## Leakage-Safe Inputs / 防泄漏输入规则

分类模型只允许使用 Sagemotion 信号特征。

The classifier only uses Sagemotion signal features.

这些字段被明确排除：

These fields are explicitly excluded:

| Excluded field / 排除字段 | Reason / 原因 |
|---|---|
| `bike` | 这是分类标签本身 / this is the classification label itself |
| `pressure_bar` | 真实预测时不应已知，而且会把胎压标签信息泄漏进分类任务 / it should not be known at prediction time and would leak pressure-label information |
| `p_number` | 胎压水平编号，不是传感器信号 / pressure-level identifier, not a sensor signal |
| `group` | CV 划分字段，不能作为模型输入 / CV split field, not a model input |
| `run_id` | 唯一编号，会让模型记忆样本 / unique identifier that can let the model memorize samples |
| file name | 元数据，不是物理信号 / metadata, not a physical signal |
| `rider_weight_kg` | 可能编码骑手或 group 背景，而不是单车结构本身 / may encode rider or group context rather than bike structure |

这个规则让模型必须从加速度和陀螺仪信号模式中学习单车结构差异。

This rule forces the model to learn bike-structure differences from acceleration and gyroscope signal patterns.

## Selected Model / 选中的模型

选中的模型是：

The selected model is:

`bike_type_ffnn_ensemble_ens3_signal_full_pca10_tanh_h6_a1`

它的结构是：

Its structure is:

```text
Sagemotion signal features
-> StandardScaler
-> PCA(10)
-> MLPClassifier(hidden_layer_sizes=(6,), activation="tanh", alpha=1.0)
-> 3-seed ensemble
-> average class probabilities at run level
```

PCA 的作用是把 128 个信号输入压缩成 10 个主成分，保留主要变化方向，同时减少小样本下的过拟合风险。

PCA compresses 128 signal inputs into 10 principal components, preserving the main variation directions while reducing overfitting risk in a small dataset.

3-seed ensemble 的作用是减少单次神经网络初始化带来的随机性。三个成员使用相同结构、不同 random seed，最后平均它们的类别概率。

The 3-seed ensemble reduces randomness from a single neural-network initialization. The three members use the same structure with different random seeds, and their class probabilities are averaged.

## Training And CV Logic / 训练与 CV 逻辑

训练逻辑仍然是 leave-one-group-out cross-validation。

The training logic still uses leave-one-group-out cross-validation.

一共有 6 个 group，所以 CV 会训练 6 次。每一次留出 1 个完整 group 做 validation，用另外 5 个 group 训练模型。

There are 6 groups, so CV trains 6 times. Each fold holds out one complete group for validation and trains on the other five groups.

每个候选模型都会经历这 6 次训练和验证。CV 结束后，我们根据 run-level accuracy、macro-F1、group 稳定性和模型复杂度选择最终模型。

Every candidate model goes through these 6 train-and-validate folds. After CV, the final candidate is selected using run-level accuracy, macro-F1, group stability, and model complexity.

选好结构以后，还会在全部 72 个本地 training-pool runs 上重新训练一次最终模型，并保存成 `.pkl` 文件。这个最终训练不是新的测试，它只是为了把所有本地可用数据都用于最终交付模型。

After selecting the structure, the final model is retrained once on all 72 local training-pool runs and saved as a `.pkl` file. This final fit is not a new test; it is used so the delivered model can learn from all locally available data.

## Current Evidence / 当前证据

| Metric / 指标 | Value / 数值 |
|---|---:|
| Run-level CV accuracy | 0.986 |
| Run-level CV macro-F1 | 0.986 |
| Minimum group accuracy | 0.917 |
| Mean prediction confidence | 0.848 |
| Labeled runs | 72 |
| Window rows | 873 |

这些结果来自本地 leave-one-group-out CV，说明模型在当前 G01-G06 数据上跨 group 泛化表现很好。

These results come from local leave-one-group-out CV and show strong cross-group generalization within the current G01-G06 data.

但它仍然不是老师隐藏测试集结果。真正的外部泛化需要等老师隐藏测试数据验证。

However, these are still not instructor hidden-test results. True external generalization must be confirmed on the hidden test set.

## Key Files / 关键文件

| File / 文件 | Purpose / 作用 |
|---|---|
| `../06_reproducible_pipeline/steps/04b_bike_type_classification.py` | Runs bike-type model selection and final training / 运行单车类型模型选择与最终训练 |
| `../06_reproducible_pipeline/src/bike_type_modeling.py` | Reusable classifier logic / 可复用分类模型逻辑 |
| `../03_outputs/tables/training_pool_bike_type_model_comparison.csv` | Candidate comparison / 候选模型比较 |
| `../03_outputs/tables/training_pool_bike_type_selected_cv_predictions.csv` | Selected-model CV predictions / 选中模型 CV 预测 |
| `../03_outputs/tables/training_pool_bike_type_cv_confusion.csv` | CV confusion matrix / CV 混淆矩阵 |
| `../03_outputs/models/training_pool_bike_type_final_model.pkl` | Final saved classifier / 最终保存的分类模型 |
| `../04_report/training_pool_bike_type_classifier_report_bilingual.md` | Bilingual report / 双语报告 |

