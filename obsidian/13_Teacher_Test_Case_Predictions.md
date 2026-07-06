# Teacher Test Case Predictions / 老师测试集预测结果

## Task Interpretation / 任务理解

老师的三项要求不是三组 case 分别对应三项任务，而是每个 test case 都要完成前两项预测：

The three requested outputs are not assigned one-by-one to the three cases. The first two predictions are required for every test case:

| Test case / 测试 case | Task 1: tire pressure / 任务 1：胎压 | Task 2: suspension type / 任务 2：悬挂/阻尼类型 | Task 3: Bike 3 comment / 任务 3：Bike 3 说明 |
|---|---|---|---|
| Case 1 | required / 需要 | required / 需要 | not applicable / 不适用 |
| Case 2 | required / 需要 | required / 需要 | not applicable / 不适用 |
| Case 3 | required / 需要 | required / 需要 | required / 需要 |

Task 3 is only about Case 3 because the instructor states that Case 3 is a completely new bicycle.

任务 3 只针对 Case 3，因为老师说明 Case 3 是一辆全新的单车。

## Instructor Metadata / 老师提供的元数据

The instructor supplied rider weight and ride-time metadata on 2026-07-06.

老师在 2026-07-06 提供了测试集的 rider weight 和 ride time。

| Case / Case | Rider weight / 骑手体重 | Ride time / 测试时间 |
|---|---:|---:|
| Case 1 | 75 kg | 9.36 s |
| Case 2 | 100 kg | 8.61 s |
| Case 3 | 89 kg | 4.97 s |

## Final Predictions / 最终预测

| Case / Case | Predicted pressure / 预测胎压 | Nearest known level / 最近已知胎压档位 | Predicted suspension / 预测悬挂类型 | Confidence / 置信度 |
|---|---:|---:|---|---:|
| Case 1 | 2.658 bar | 3.0 bar | Front and rear Suspension | 0.808 |
| Case 2 | 1.506 bar | 2.0 bar | No Suspension | 0.553 |
| Case 3 | 2.292 bar | 2.0 bar | Front and rear Suspension | 0.579 |

## Important Caveat / 重要注意点

The tire-pressure FFNN requires bike one-hot inputs. The hidden test metadata gives rider weight and ride time, but it does not give bike labels. Therefore the final pressure result uses the bike context implied by the suspension classifier.

胎压 FFNN 需要 bike one-hot 输入。老师测试集元数据提供了 rider weight 和 ride time，但没有提供 bike label。因此最终胎压预测使用悬挂分类器推断出的 bike context。

The full pressure-sensitivity table keeps all three bike-context scenarios for auditability:

完整胎压敏感性表保留了三种 bike context 情景，便于审查：

| Case / Case | FAT context | ISY context | MTB context |
|---|---:|---:|---:|
| Case 1 | 1.908 bar | 2.894 bar | 2.658 bar |
| Case 2 | 0.673 bar | 1.506 bar | 2.415 bar |
| Case 3 | 0.842 bar | 0.993 bar | 2.292 bar |

## Case 3 Evidence / Case 3 证据

Case 3 is predicted as `Front and rear Suspension`, with the closest bike context being MTB. In the fitted suspension-model PCA space, the five nearest training runs are all MTB runs, which supports the conclusion that Bike 3 is most similar to the MTB / front-and-rear-suspension class.

Case 3 被预测为 `Front and rear Suspension`，最接近的 bike context 是 MTB。在悬挂模型拟合后的 PCA 空间中，距离 Case 3 最近的 5 个训练 run 全部都是 MTB run，因此支持 Bike 3 最接近 MTB / 前后悬挂类别。

At the same time, the distances for Case 3 are larger than the nearest-run distances for Case 1 and Case 2. Therefore the correct wording is not “Bike 3 is exactly an MTB”, but “Bike 3 is a new bicycle whose signal response is closest to the MTB / front-and-rear-suspension group.”

同时，Case 3 的距离明显大于 Case 1 和 Case 2 的最近样本距离。因此更严谨的表述不是“Bike 3 就是 MTB”，而是“Bike 3 是新单车，但它的信号响应最接近 MTB / 前后悬挂组”。

## Key Files / 关键文件

| File / 文件 | Purpose / 用途 |
|---|---|
| `../06_reproducible_pipeline/steps/08_predict_teacher_test_cases.py` | Reproducible test-case prediction script / 可复现的测试集预测脚本 |
| `../03_outputs/tables/teacher_test_case_final_predictions.csv` | Final predictions / 最终预测表 |
| `../03_outputs/tables/teacher_test_case_pressure_scenarios.csv` | Pressure predictions under each bike context / 不同 bike context 下的胎压预测 |
| `../03_outputs/tables/teacher_test_case_suspension_predictions.csv` | Suspension predictions and probabilities / 悬挂预测与概率 |
| `../03_outputs/tables/teacher_test_case_nearest_training_runs.csv` | Nearest training-run evidence / 最近训练 run 证据 |
| `../04_report/teacher_test_case_predictions_bilingual.md` | Bilingual report / 双语报告 |
