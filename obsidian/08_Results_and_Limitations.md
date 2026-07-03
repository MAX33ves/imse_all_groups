# Results And Limitations / 结果与局限

## Most Important Results / 当前最重要的结果

| 项目 / Item | 值 / Value |
|---|---:|
| 本地训练 run / local training runs | 72 |
| 窗口特征行 / window feature rows | 873 |
| CV folds | 6 |
| CV MAE | 0.290 bar |
| CV RMSE | 0.388 bar |
| CV nearest-level accuracy | 0.597 |
| CV macro-F1 | 0.466 |
| Suspension-type CV accuracy | 0.986 |
| Suspension-type CV macro-F1 | 0.986 |

## How To Interpret This Performance / 怎么解释这个水平

这个结果比之前只用 P1/P2 训练更合理，因为现在训练池覆盖了更多胎压水平，尤其是 MTB 的 1.0 bar 也进入训练池。

This result is more reasonable than the earlier P1/P2-only setup because the training pool now covers more pressure levels, especially the 1.0 bar MTB case.

但是 MTB 仍然是最难预测的单车类型。

However, MTB remains the most difficult bike type to predict.

- FAT MAE：0.136 bar。
- ISY MAE：0.203 bar。
- MTB MAE：0.532 bar。

这说明模型对 MTB 的振动-胎压关系学习得不够稳定，可能和 MTB 结构、传感器响应、骑行差异或特征不足有关。

This suggests that the model learns the vibration-pressure relationship less reliably for MTB. Possible causes include MTB structure, sensor response, rider behavior, or insufficient features for MTB-specific dynamics.

悬挂类型分类任务表现明显更强，选中模型在 run-level leave-one-group-out CV 上达到 0.986 accuracy 和 0.986 macro-F1。悬挂标签来自课程表中 FAT/ISY/MTB 与悬挂类型的一一映射；模型只使用 Sagemotion 信号特征，不使用 `bike`、`suspension_type`、`pressure_bar`、`rider_weight_kg` 或 group/run/file metadata。

The suspension-type classification task is much stronger: the selected model reaches 0.986 run-level leave-one-group-out CV accuracy and 0.986 macro-F1. The suspension labels are derived from the one-to-one FAT/ISY/MTB to suspension-type mapping in the course table. It uses only Sagemotion signal features and excludes `bike`, `suspension_type`, `pressure_bar`, `rider_weight_kg`, and group/run/file metadata.

## Suggested Report Text / 报告里应该怎么写

> Since the instructor has a hidden test dataset, all observed P1-P4 runs were used as the local training pool. Local generalization was estimated by leave-one-group-out cross-validation. The selected FFNN uses rider weight as an input and achieved a run-level CV MAE of 0.290 bar and RMSE of 0.388 bar. The largest remaining error source is MTB, suggesting that the current feature representation captures FAT and ISY pressure changes better than MTB pressure changes.



> 由于老师还有隐藏测试集，所有已观测的 P1-P4 run 都被作为本地训练池。为了估计泛化能力，我们使用 leave-one-group-out 交叉验证。选中的 FFNN 使用 rider weight 作为输入，在 run-level 上达到 0.290 bar 的 CV MAE 和 0.388 bar 的 CV RMSE。剩余误差主要来自 MTB，说明当前特征对 FAT 和 ISY 的胎压变化描述更稳定，而对 MTB 还不够充分。

> For the additional suspension/damping task, a separate FFNN classifier was trained because suspension type is a categorical target, not a continuous regression target. The labels were derived from the course table mapping bike type to suspension type. The selected classifier uses Sagemotion signal features only and achieved 0.986 run-level CV accuracy and 0.986 macro-F1 under the same leave-one-group-out validation design.



> 对新增的悬挂/阻尼类型任务，我们训练了一个单独的 FFNN 分类模型，因为悬挂类型是类别目标，而不是连续回归目标。标签来自课程表中“单车类型 -> 悬挂类型”的映射。选中的分类模型只使用 Sagemotion 信号特征，在同样的 leave-one-group-out 验证设计下达到 0.986 run-level CV accuracy 和 0.986 macro-F1。

## Limitations / 局限

- 本地验证只有 6 个 group fold，样本仍然偏小。
- Local validation has only 6 group folds, so the sample size is still small.
- 窗口样本不是独立样本，所以必须看 run-level 指标。
- Window samples are not independent, so run-level metrics must be used.
- 隐藏测试集分布未知，CV 只能估计本地跨组泛化。
- The hidden-test distribution is unknown; CV only estimates local cross-group generalization.
- 悬挂类型分类模型表现很高，但仍需要隐藏测试集验证它是否能泛化到老师的数据。
- The suspension classifier performs strongly locally, but the hidden test set is still needed to confirm generalization to the instructor data.
- 当前模型主要使用手工信号特征，可能还没有完全捕捉 MTB 的结构响应。
- The current model mainly uses handcrafted signal features and may not fully capture MTB structural response.
- 相关性分析只用于解释特征和检查冗余，最终模型选择仍以 leave-one-group-out CV 为准。详细证据链见 [[11_Feature_Selection_Rationale]]。
- Correlation analysis is only used to explain features and check redundancy. Final model selection is still based on leave-one-group-out CV. See [[11_Feature_Selection_Rationale]] for the full evidence chain.
