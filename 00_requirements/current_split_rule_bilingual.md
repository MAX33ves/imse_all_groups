# Current Data Split Rule / 当前数据划分规则

P1-P4 full training-pool version.  
P1-P4 全量训练版。

## Current Rule / 最新规则

| Item / 项目 | Current rule / 当前规则 |
|---|---|
| Available data scope / 可用数据范围 | All groups in `Measurement_Campaign/G01-G06` / `Measurement_Campaign/G01-G06` 全部组 |
| Local training pool / 本地训练池 | All P1/P2/P3/P4 runs from all groups, 72 runs in total / 所有组的 P1/P2/P3/P4，共 72 个 run |
| Local validation / 本地验证方式 | leave-one-group-out cross-validation / 留一组交叉验证 |
| External test set / 外部测试集 | Instructor hidden test data / 老师手里的隐藏测试数据 |
| Prediction task / 预测任务 | Regression: predict continuous tire pressure `pressure_bar` / 回归任务，预测连续胎压 `pressure_bar` |

## Why P3/P4 Are No Longer A Local Final Test / 为什么不再用 P3/P4 做本地 final test

The instructor still has a hidden test set, so all locally observed data should be used as efficiently as possible for training. P3/P4 are no longer treated as a local final test set; they are part of the training pool.  
现在确认老师手里还有隐藏测试数据，所以本地观测数据应该尽可能用于训练。P3/P4 不再是本地 final test，而是训练池的一部分。

To still estimate generalization locally, we use group-level cross-validation:  
为了仍然估计模型泛化能力，我们使用组级交叉验证：

1. Hold out one complete group as validation. / 每次留出一个完整 group 做 validation。
2. Train on the other five groups. / 其余五个 group 做 training。
3. Rotate until all six groups have been used as validation once. / 6 个 group 轮流做 validation。
4. Each validation fold contains that group's FAT/ISY/MTB P1-P4 runs. / 每个 validation fold 都包含该组 FAT/ISY/MTB 的 P1-P4。

This is more reliable than random window splitting because windows from the same run are highly correlated; random window splitting would leak near-duplicate ride information across training and validation.  
这样比随机窗口划分更可靠，因为同一个 run 切出来的窗口高度相关；随机窗口划分会让相近的骑行信息同时出现在训练和验证两边，导致结果偏乐观。

## Fields That Must Not Be Model Inputs / 不可作为模型输入的字段

- `pressure_bar`
- `group`
- `p_number`
- `run_id`
- file name / 文件名

These fields are used for labels, grouping, audit, or traceability only.  
这些字段只用于标签、分组、审计或追踪，不能作为模型输入。

## Allowed Input Or Candidate-Input Fields / 可以作为输入或候选输入的字段

- Sagemotion signal window features / Sagemotion 信号窗口特征
- bike type one-hot features / bike type one-hot 特征
- rider weight / 骑手体重

The current final model must include `rider_weight_kg`. Model selection still compares candidates with and without rider weight using group-level CV, but the final eligible FFNN must include rider weight.  
当前最终模型必须使用 `rider_weight_kg`。模型选择仍使用组级交叉验证评估含体重和不含体重的候选，但最终合格的 FFNN 必须包含体重。
