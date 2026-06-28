# Current Rules / 当前规则

## Data Usage / 数据怎么用

| 项目 / Item | 规则 / Rule |
|---|---|
| 本地数据 / Local data | G01-G06 全部组 / all G01-G06 groups |
| 可用于训练的数据 / Data used for training | P1、P2、P3、P4 全部 / all P1, P2, P3, and P4 runs |
| 本地测试集 / Local test set | 不再单独设置 / no separate local final test set |
| 本地验证方法 / Local validation method | leave-one-group-out CV |
| 真正外部测试 / Real external test | 老师隐藏数据 / instructor hidden dataset |

## Why This Split Is Used / 为什么这样划分

之前我们以为 P3/P4 是最终测试集，所以只用 P1/P2 训练。现在已确认老师手里还有额外测试数据，所以本地的 P1-P4 都应该用于训练最终模型。

We previously treated P3/P4 as the final test set and trained only on P1/P2. Now that we know the instructor has an additional hidden test set, all local P1-P4 runs should be used to train the final local model.

但是如果完全不验证，我们就不知道模型效果。因此采用 leave-one-group-out：

However, we still need a local estimate of model performance, so we use leave-one-group-out cross-validation:

- 一次留出一个完整 group。
- Hold out one complete group at a time.
- 其余 5 个 group 训练。
- Train on the remaining five groups.
- 留出的 group 做 validation。
- Use the held-out group as validation.
- 6 个 group 轮流做一次 validation。
- Rotate through all six groups.

这种方式比随机窗口划分更合理，因为同一个 run 切出来的窗口高度相关。随机窗口划分会让训练集和验证集看到同一个 run 的相邻窗口，结果会虚高。

This is more reliable than a random window split because windows from the same run are highly correlated. A random split would put neighboring windows from the same run into both training and validation, making the result overly optimistic.

## Fields That Must Not Be Model Inputs / 模型不能使用的字段

这些字段只用于记录和划分，不能作为模型输入：

These fields are only for bookkeeping, splitting, or auditing. They must not be used as model inputs:

- `pressure_bar`
- `group`
- `p_number`
- `run_id`
- 文件名 / file name

## Main Model Inputs / 模型主要输入

- Sagemotion 传感器窗口特征。
- Sagemotion sensor window features.
- bike type one-hot。
- Bike-type one-hot features.
- rider weight。最终模型必须使用 `rider_weight_kg`。
- Rider weight. The final model must include `rider_weight_kg`.
