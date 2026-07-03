# IMSE Project Index / IMSE 项目索引：P1-P4 Full Training Pool / P1-P4 全量训练版

## Current Project Rule / 当前项目规则

老师手里还有隐藏测试数据，所以我们本地已有的 G01-G06、P1-P4 全部都作为训练池使用。我们不再把 P3/P4 当成本地 final test。

Because the instructor still has a hidden test dataset, all locally observed G01-G06 P1-P4 runs are used as the local training pool. P3/P4 are no longer treated as a separate local final test set.

## Key Conclusions / 核心结论

| 中文 | English |
|---|---|
| 本地训练池：72 个 run。 | Local training pool: 72 runs. |
| 窗口特征：873 行。 | Window-level feature rows: 873. |
| 本地验证：leave-one-group-out cross-validation。 | Local validation: leave-one-group-out cross-validation. |
| 选中模型：`ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8x4_a1_median`。 | Selected model: `ffnn_training_pool_ensemble_ens3_compact_weight_pca6_tanh_h8x4_a1_median`. |
| 最终模型输入包含 rider weight。 | The final model includes rider weight as an input. |
| CV MAE：0.290 bar。 | CV MAE: 0.290 bar. |
| CV RMSE：0.388 bar。 | CV RMSE: 0.388 bar. |
| 悬挂类型分类模型：`suspension_ffnn_ensemble_ens3_signal_full_pca10_tanh_h6_a1`。 | Suspension classifier: `suspension_ffnn_ensemble_ens3_signal_full_pca10_tanh_h6_a1`. |
| 悬挂类型 CV accuracy：0.986。 | Suspension CV accuracy: 0.986. |
| 最终模型已用全部 72 个 run 重新训练并保存为 `.pkl`。 | The final model has been retrained on all 72 runs and saved as a `.pkl` file. |
| 老师视角审查补充报告：`../05_teacher_review/teacher_review_audit_bilingual.md`。 | Teacher-review supplement: `../05_teacher_review/teacher_review_audit_bilingual.md`. |

## Suggested Reading Order / 建议阅读顺序

1. [[01_Current_Rules]]
2. [[02_Code_Structure]]
3. [[03_Step_00_01_Data_Inventory]]
4. [[04_Step_02_Feature_Extraction]]
5. [[05_Step_03_EDA_and_PCA]]
6. [[06_Step_04_Model_Selection]]
7. [[07_Step_05_Final_Model]]
8. [[08_Results_and_Limitations]]
9. [[09_File_Map]]
10. [[10_Bilingual_Quick_Brief]]
11. [[11_Feature_Selection_Rationale]]
12. [[12_Suspension_Classifier]]
13. `../05_teacher_review/teacher_review_audit_bilingual.md`

## Key Figures / 关键图

![CV predicted vs actual](assets/training_pool_ffnn_02_cv_predicted_vs_actual.png)

![CV error heatmap](assets/training_pool_ffnn_03_cv_error_heatmap.png)

![Feature target correlation](assets/training_pool_05_feature_target_correlation.png)
