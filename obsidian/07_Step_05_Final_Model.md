# Step 05: Final Model Training / 最终模型训练

脚本 / Script:

`06_reproducible_pipeline/steps/05_train_final_model.py`

## Why This Step Exists / 这一步为什么存在

Step 04 用 leave-one-group-out CV 评估模型和选择结构。选好结构后，最终要提交或用于老师隐藏测试的模型应该用全部本地数据训练。

Step 04 evaluates candidate models and selects the architecture using leave-one-group-out CV. After the structure is selected, the final model for submission or hidden testing should be trained on all local data.

所以 Step 05 做的是 / Therefore, Step 05:

- 读取 Step 04 选中的模型结构。 / Reads the model structure selected in Step 04.
- 用全部 72 个 run 的窗口特征重新训练。 / Retrains on window features from all 72 runs.
- 最终模型输入包含 `rider_weight_kg`。 / Keeps `rider_weight_kg` in the final model inputs.
- 保存最终模型文件。 / Saves the final model file.

## How Final Training Relates To CV / 最终训练和 CV 的关系

Step 04 的 6-fold CV 用来选择和评估模型结构。CV 结束后，Step 05 不再留出 validation group，而是用全部 72 个本地 run 重新训练最终模型，供老师隐藏测试集使用。

Step 04 uses 6-fold leave-one-group-out CV to select and evaluate the model structure. After CV is finished, Step 05 no longer holds out a validation group. It retrains the final model on all 72 local runs for the instructor's hidden test set.

因为最终模型是 `ensemble_ens3`，所以最终全量训练仍然包含 3 个成员：

Because the final model is `ensemble_ens3`, final full-data training still contains three members:

```text
full local training pool × seed 11
full local training pool × seed 42
full local training pool × seed 91
```

这 3 个成员结构相同，但随机初始化不同。最终预测时，三个成员的预测先取平均，再把同一个 run 的多个窗口预测用 median 聚合成一个 run-level 胎压。

These three members have the same architecture but different random initializations. During prediction, the three member predictions are first averaged, and then the multiple window-level predictions from the same run are aggregated with the median into one run-level pressure prediction.

最终保存模型中的三个成员实际优化迭代次数为：

The three final saved ensemble members converged after these optimizer iterations:

| seed / member | actual `n_iter_` |
|---|---:|
| seed 11 | 433 |
| seed 42 | 539 |
| seed 91 | 560 |

这些是 `lbfgs` 优化器迭代次数，不是传统 epoch 数。训练设置仍然是 `max_iter=3000` 和 `max_fun=40000`。

These are `lbfgs` optimizer iterations, not traditional epoch counts. The training limits are still `max_iter=3000` and `max_fun=40000`.

## Outputs / 输出

| 文件 / File                                                     | 说明 / Description                                                                |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `training_pool_ffnn_final_model.pkl`                          | 最终模型，包含 3 个 ensemble pipeline / final model containing three ensemble pipelines |
| `training_pool_ffnn_final_model_summary.json`                 | 最终模型摘要 / final-model summary                                                    |
| `training_pool_ffnn_final_model_training_fit_predictions.csv` | 训练池拟合预测 / training-pool fit predictions                                         |
| `training_pool_ffnn_final_model_report_bilingual.md`             | 中英双语说明 / bilingual final-model explanatory report                             |

## Important Note / 注意

Training-pool fit 指标不是泛化能力。报告里应该主要引用 Step 04 的 CV 指标：

Training-pool fit metrics are not generalization metrics. The report should mainly cite the Step 04 CV metrics:

- CV MAE：0.290 bar。 / CV MAE: 0.290 bar.
- CV RMSE：0.388 bar。 / CV RMSE: 0.388 bar.

Step 05 的 training fit MAE 是 0.226 bar，这只是模型拟合训练数据的误差，不能当作测试误差。

The Step 05 training-fit MAE is 0.226 bar. This is only the error on the training pool and must not be treated as test error.
