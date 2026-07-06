# Teacher Test Case Predictions / 老师测试集预测结果

## Inputs / 输入信息

The instructor supplied rider weight and ride-time metadata on 2026-07-06. The saved final models were not retrained; they were only applied to the teacher test cases.
老师在 2026-07-06 提供了测试集的 rider weight 和 ride time。这里没有重新训练模型，只是把已经保存的最终模型应用到老师测试数据上。

| case_id | rider_weight_kg | table_ride_time_s | sample_rate_hz | raw_n_rows | active_start_idx | active_end_idx | active_duration_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Case 1 | 75.000 | 9.360 | 100.000 | 1702 | 711 | 1647 | 9.360 |
| Case 2 | 100.000 | 8.610 | 100.000 | 1143 | 240 | 1101 | 8.610 |
| Case 3 | 89.000 | 4.970 | 100.000 | 883 | 68 | 565 | 4.970 |

## Final Predictions / 最终预测

| case_id | rider_weight_kg | table_ride_time_s | pred_pressure_bar | nearest_known_pressure_level_bar | pred_suspension_type | pred_bike_context_for_pressure_model | suspension_confidence | n_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Case 1 | 75.000 | 9.360 | 2.658 | 3.000 | Front and rear Suspension | MTB | 0.808 | 18 |
| Case 2 | 100.000 | 8.610 | 1.506 | 2.000 | No Suspension | ISY | 0.553 | 17 |
| Case 3 | 89.000 | 4.970 | 2.292 | 2.000 | Front and rear Suspension | MTB | 0.579 | 9 |

## Pressure Sensitivity By Bike Context / 不同车型上下文下的胎压敏感性

The pressure FFNN requires bike one-hot inputs. Because the hidden test sheet does not provide bike labels, the final pressure column uses the bike context implied by the suspension classifier. The table below keeps all three context scenarios for auditability.
胎压 FFNN 需要 bike one-hot 输入。由于隐藏测试表没有给出 bike label，最终胎压使用悬挂分类器推断出的 bike context。下表保留三种 context 的胎压情景，方便审查。

| case_id | bike_context_for_pressure_model | pred_pressure_bar | pred_pressure_window_std | n_windows |
| --- | --- | --- | --- | --- |
| Case 1 | FAT | 1.908 | 0.511 | 18 |
| Case 1 | ISY | 2.894 | 0.247 | 18 |
| Case 1 | MTB | 2.658 | 0.463 | 18 |
| Case 2 | FAT | 0.673 | 0.122 | 17 |
| Case 2 | ISY | 1.506 | 0.569 | 17 |
| Case 2 | MTB | 2.415 | 0.490 | 17 |
| Case 3 | FAT | 0.842 | 0.211 | 9 |
| Case 3 | ISY | 0.993 | 0.465 | 9 |
| Case 3 | MTB | 2.292 | 0.532 | 9 |

## Case 3 Comment / Case 3 说明

Case 3 is treated as a new bicycle. The model can compare its signal pattern with the learned FAT/ISY/MTB suspension categories, but the pressure result remains an approximation because the pressure regressor was trained with known bike one-hot context.
Case 3 是新单车。模型可以把它的信号模式与已学习的 FAT/ISY/MTB 悬挂类别比较，但胎压结果仍是近似值，因为胎压回归模型训练时使用了已知 bike one-hot 上下文。

Nearest training runs in the fitted suspension-model PCA space:

| case_id | rank | nearest_training_run | nearest_group | nearest_bike | nearest_p_number | nearest_suspension_type | pca_distance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Case 3 | 1 | G01_MTB_P4 | G01 | MTB | P4 | Front and rear Suspension | 15.082 |
| Case 3 | 2 | G05_MTB_P4 | G05 | MTB | P4 | Front and rear Suspension | 15.543 |
| Case 3 | 3 | G06_MTB_P4 | G06 | MTB | P4 | Front and rear Suspension | 16.054 |
| Case 3 | 4 | G03_MTB_P3 | G03 | MTB | P3 | Front and rear Suspension | 16.186 |
| Case 3 | 5 | G01_MTB_P3 | G01 | MTB | P3 | Front and rear Suspension | 16.339 |

Distances to training bike/suspension centroids:

| case_id | bike_centroid | suspension_centroid | centroid_distance |
| --- | --- | --- | --- |
| Case 3 | MTB | Front and rear Suspension | 16.819 |
| Case 3 | FAT | Suspension because of tyres | 19.068 |
| Case 3 | ISY | No Suspension | 22.435 |