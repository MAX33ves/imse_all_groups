# Feature Selection And Model Input Rationale / 特征选择与模型输入解释

## One-Sentence Conclusion / 一句话结论

当前最终模型不是“随便把所有列都丢进 FFNN”。它的输入选择遵循一条证据链：

The final model does not simply throw every available column into the FFNN. Its input selection follows an evidence chain:

`物理意义 -> run-level 相关性检查 -> 特征冗余检查 -> compact_weight 输入空间 -> StandardScaler + PCA -> leave-one-group-out CV 验证`

`Physical meaning -> run-level correlation checks -> redundancy checks -> compact_weight feature space -> StandardScaler + PCA -> leave-one-group-out CV validation`

最终模型使用 / Final model input:

`24 个 compact 信号特征 + 3 个 bike one-hot + rider_weight_kg = 28 个 PCA 前输入`

`24 compact signal features + 3 bike one-hot features + rider_weight_kg = 28 pre-PCA inputs`

## Why Window-Level Correlation Is Not Used / 为什么不用窗口级相关性

虽然窗口特征表有 873 行，但这些窗口来自 72 个 run。同一个 run 内相邻窗口高度相关，不能当成 873 个完全独立样本。

Although the window-feature table has 873 rows, these windows come from 72 runs. Neighboring windows within the same run are highly correlated, so they cannot be treated as 873 independent samples.

所以解释性相关性分析使用 run-level 粒度 / Therefore, explanatory correlation analysis uses run-level data:

1. 先对每个 run 的窗口输入取中位数。
   First take the median of window-level inputs within each run.
2. 得到 72 行 run-level 输入。
   This produces 72 run-level input rows.
3. 再计算每个输入和 `pressure_bar` 的 Pearson / Spearman 相关性。
   Then compute Pearson / Spearman correlations between each input and `pressure_bar`.

这样和最终报告使用 run-level CV 指标的逻辑一致。

This is consistent with the final report, where model performance is also evaluated at run level.

## What The Correlation Matrix Can Explain / 相关性矩阵能说明什么

相关性矩阵主要回答两个问题 / The correlation matrix mainly answers two questions:

1. 哪些输入和胎压 `pressure_bar` 有明显单调或线性关系。
   Which inputs have clear monotonic or linear relationships with `pressure_bar`.
2. 哪些输入彼此高度重复，说明需要降维、正则化或更紧凑的输入空间。
   Which inputs are highly redundant with each other, indicating the need for dimensionality reduction, regularization, or a more compact input space.

但相关性矩阵不能单独决定最终模型输入，因为 / But the correlation matrix cannot determine the final model inputs by itself because:

- Pearson / Spearman 主要描述线性或单调关系。 / Pearson / Spearman mainly describe linear or monotonic relationships.
- FFNN 可以学习非线性组合。 / An FFNN can learn nonlinear combinations.
- bike type 和实验设计会影响全局相关性。 / Bike type and experimental design influence global correlations.
- 如果用全部数据按相关性自动筛选特征，再报告 CV，容易产生偏乐观解释。 / Automatically selecting features from all data by correlation and then reporting CV can make the explanation too optimistic.

因此这里把相关性作为解释证据，不把它当成唯一选择规则。最终模型仍由 leave-one-group-out CV 验证。

Therefore, correlation is used as explanatory evidence, not as the only selection rule. The final model is still validated by leave-one-group-out CV.

## Target-Correlation Evidence / 目标相关性证据

下图展示最终输入中，与 `pressure_bar` 的 Spearman 相关性较明显的特征：

The figure below shows final inputs with relatively strong Spearman correlation to `pressure_bar`:

![Feature target correlation](assets/training_pool_05_feature_target_correlation.png)

关键观察 / Key observations:

| 特征 / Feature | Spearman vs pressure | 解释 / Explanation |
|---|---:|---|
| `bike_FAT` | -0.839 | FAT 的胎压范围整体低于 ISY/MTB，所以 bike type 是必要上下文。 / FAT pressures are generally lower than ISY/MTB pressures, so bike type is necessary context. |
| `acc_spectral_centroid_mean` | 0.607 | 胎压变化会改变振动频率重心，频域信息有解释价值。 / Tire pressure can shift the vibration spectral centroid, so frequency information is useful. |
| `acc_band_15_30_power_mean` | 0.501 | 高频段能量比例随胎压变化，反映轮胎刚度和冲击响应差异。 / High-frequency energy ratio changes with pressure and reflects tire stiffness and impact response. |
| `acc_band_3_8_power_mean` | -0.486 | 中低频段能量比例与胎压呈反向单调关系。 / Mid-low-frequency energy ratio has a negative monotonic relationship with pressure. |
| `acc_energy_per_s_mean` | 0.471 | 单位时间能量反映振动强度，胎压会改变阻尼和冲击强度。 / Energy per second reflects vibration intensity; pressure changes damping and impact intensity. |
| `acc_spectral_entropy_mean` | 0.468 | 频谱分散程度随胎压变化，说明压力影响振动能量分布。 / Spectral dispersion changes with pressure, indicating pressure affects energy distribution. |
| `acc_p95_abs_mean` | 0.448 | 高分位振幅代表较强冲击，和胎压响应有关。 / High-percentile amplitude captures stronger impacts related to pressure response. |
| `acc_ptp_mean` | 0.428 | peak-to-peak 范围反映振动幅值范围。 / Peak-to-peak range reflects vibration amplitude span. |

`rider_weight_kg` 和 `pressure_bar` 本身的直接相关性很弱，Spearman 约为 0.011。这并不是坏事，反而说明体重不是胎压标签的简单替代物。它作为输入的意义是提供载荷上下文：同样胎压下，不同骑手体重会改变轮胎压缩、接触、冲击和振动响应。

`rider_weight_kg` has very weak direct correlation with `pressure_bar` itself, with Spearman correlation around 0.011. This is not necessarily bad; it suggests rider weight is not simply a proxy for the pressure label. Its purpose is to provide load context: at the same pressure, different rider weights can change tire compression, contact behavior, impact response, and vibration response.

## Why These Signal Feature Families Are Kept / 为什么保留这些信号特征家族

最终模型保留 12 类 compact 信号特征家族，每类同时保留 `acc` 和 `gyro` 的双传感器平均值。

The final model keeps 12 compact signal feature families. For each family, it keeps the two-sensor mean for both `acc` and `gyro`.

| 特征家族 / Feature family | 为什么有物理意义 / Physical meaning |
|---|---|
| `rms` | 均方根振动强度，胎压会影响冲击和阻尼。 / RMS vibration intensity; tire pressure affects impact and damping. |
| `std` | 波动程度，反映信号不稳定性和振动幅度。 / Signal variability and vibration amplitude. |
| `energy_per_s` | 单位时间振动能量，和路面冲击传递有关。 / Vibration energy per second, related to road-impact transmission. |
| `p95_abs` | 较大冲击幅值，避免只看均值。 / Large impact amplitudes, avoiding reliance on the mean only. |
| `ptp` | 峰峰值范围，反映振动上下界跨度。 / Peak-to-peak range of vibration. |
| `dom_freq` | 主导频率，胎压变化可能改变结构响应频率。 / Dominant frequency; pressure may shift structural response frequency. |
| `spectral_centroid` | 频谱重心，描述振动能量偏低频还是高频。 / Spectral centroid; indicates whether energy is lower- or higher-frequency. |
| `spectral_entropy` | 频谱分散程度，描述振动能量是否集中。 / Spectral dispersion; shows whether vibration energy is concentrated or spread out. |
| `band_0p5_3_power` | 低频能量比例。 / Low-frequency energy ratio. |
| `band_3_8_power` | 中低频能量比例。 / Mid-low-frequency energy ratio. |
| `band_8_15_power` | 中高频能量比例。 / Mid-high-frequency energy ratio. |
| `band_15_30_power` | 高频能量比例。 / High-frequency energy ratio. |

这些特征覆盖了两类信息 / These features cover two information types:

- 幅值/能量：回答“震得多强”。 / Amplitude/energy: "how strong is the vibration?"
- 频率分布：回答“震动能量集中在哪些频率”。 / Frequency distribution: "where is the vibration energy concentrated?"

胎压变化既可能改变振动强度，也可能改变频率和阻尼表现，所以两类信息都要保留。

Tire pressure can change both vibration intensity and frequency/damping behavior, so both types of information are retained.

## Why The Full 132 Inputs Are Not Used / 为什么不用完整 132 个输入

完整输入池包含 / The full input pool contains:

- 128 个信号特征。 / 128 signal features.
- 3 个 bike one-hot。 / 3 bike one-hot features.
- 1 个 rider weight。 / 1 rider-weight feature.

但完整特征池中有很多更容易受噪声影响的列，例如 `_max`、`_min`、`_absdiff`、`skew`、`kurtosis`。这些列有时和胎压相关性也不低，例如 `acc_kurtosis_max`、`acc_skew_max` 等在全局相关性排行中靠前。但它们更容易受到单个窗口、单个传感器安装状态或异常冲击影响。当前样本只有 72 个 run，直接使用完整 132 维更容易过拟合。

The full feature pool contains many noise-sensitive columns, such as `_max`, `_min`, `_absdiff`, `skew`, and `kurtosis`. Some of these may correlate with pressure, for example `acc_kurtosis_max` or `acc_skew_max`, but they are more sensitive to individual windows, sensor mounting, or abnormal impacts. With only 72 runs, directly using all 132 dimensions would increase overfitting risk.

所以最终模型采用更稳健的 compact 思路 / Therefore, the final model uses a more robust compact approach:

1. 只保留物理意义直接的 12 类信号家族。 / Keep only 12 physically interpretable signal families.
2. 双传感器聚合只保留 `_mean`，减少单个传感器局部异常的影响。 / Keep only the `_mean` two-sensor aggregation to reduce local single-sensor anomalies.
3. 保留 bike one-hot，提供车型上下文。 / Keep bike one-hot features for bike context.
4. 加入 `rider_weight_kg`，提供载荷上下文。 / Add `rider_weight_kg` for load context.

## Why PCA Is Needed / 为什么需要 PCA

最终输入之间仍有明显冗余。相关性矩阵如下：

The final inputs are still substantially redundant. The correlation matrix is shown below:

![Final input correlation matrix](assets/training_pool_06_final_input_correlation_matrix.png)

高冗余示例 / High-redundancy examples:

| 特征 1 / Feature 1 | 特征 2 / Feature 2 | Spearman |
|---|---|---:|
| `acc_rms_mean` | `acc_std_mean` | 0.998 |
| `gyro_rms_mean` | `gyro_std_mean` | 0.992 |
| `acc_energy_per_s_mean` | `acc_rms_mean` | 0.988 |
| `acc_energy_per_s_mean` | `acc_std_mean` | 0.987 |
| `gyro_energy_per_s_mean` | `gyro_std_mean` | 0.984 |
| `acc_spectral_centroid_mean` | `acc_spectral_entropy_mean` | 0.926 |

这说明很多特征描述的是相近物理现象，例如振动强度和能量。完全删除其中某一个会损失解释直觉，但全部直接输入 FFNN 又会增加共线性和过拟合风险。

This shows that many features describe similar physical phenomena, such as vibration intensity and energy. Removing one completely may reduce interpretability, but feeding all correlated inputs directly into the FFNN increases collinearity and overfitting risk.

因此最终 pipeline 使用 / Therefore, the final pipeline uses:

`StandardScaler -> PCA(n_components=6) -> MLPRegressor`

PCA 的作用是把 28 个相关输入压缩成 6 个更紧凑的综合方向，让模型看到主要信息而不是直接学习一堆高度重复的列。

PCA compresses the 28 correlated inputs into 6 compact combined directions, letting the model learn the main information instead of directly fitting many highly redundant columns.

## Why Bike One-Hot Is Used / 为什么要 bike one-hot

FAT、ISY、MTB 不只是标签，它们代表不同结构和不同胎压范围。比如 FAT 的胎压整体更低，MTB/ISY 的胎压范围更高。

FAT, ISY, and MTB are not just labels; they represent different structures and pressure ranges. For example, FAT pressures are generally lower, while MTB/ISY pressures are higher.

如果不加入 bike one-hot，模型会被迫用一套完全相同的振动-胎压关系解释三种车。加入 bike one-hot 后，模型可以学习 / Without bike one-hot inputs, the model would have to explain all three bikes using one identical vibration-pressure relationship. With bike one-hot inputs, the model can learn:

- 不同车型的基础胎压范围不同。 / Different bike types have different base pressure ranges.
- 相同振动特征在不同车型上的含义不同。 / The same vibration feature may mean different things for different bikes.
- FAT、ISY、MTB 的结构响应不同。 / FAT, ISY, and MTB have different structural responses.

所以 bike one-hot 是合理输入，不是泄漏字段。

Therefore, bike one-hot is a valid context input, not a leakage field.

## Why Rider Weight Is Added / 为什么加入 rider weight

`rider_weight_kg` 的直接目标相关性很弱，但它仍然有合理意义：

`rider_weight_kg` has weak direct target correlation, but it is still meaningful:

- 骑手体重改变轮胎载荷。 / Rider weight changes tire load.
- 载荷改变轮胎压缩和接地形态。 / Load changes tire compression and contact behavior.
- 同样胎压下，较重骑手可能产生不同振动幅值和频率响应。 / At the same pressure, a heavier rider may produce different vibration amplitudes and frequency response.
- 它不是目标标签，也不是由胎压计算出来的字段，因此不是标签泄漏。 / It is not the target label and is not calculated from tire pressure, so it is not label leakage.

当前规则要求最终模型使用 `rider_weight_kg`。代码中也加入了检查：如果最终模型不含体重，Step 05 和 Step 06 会报错。

The current project rule requires the final model to include `rider_weight_kg`. The code also checks this: Step 05 and Step 06 will fail if the final model does not include rider weight.

## Suggested Report Text / 最终选择链条报告表述

> Feature selection was guided by physical interpretability, run-level correlation checks, redundancy analysis, and group-level cross-validation. The compact feature set keeps vibration amplitude, energy, and frequency-distribution descriptors that have clear mechanical links to tire pressure. Bike type is included because tire-pressure ranges and structural responses differ across FAT, ISY, and MTB. Rider weight is included as a load-context input. Since many compact features are strongly correlated, the final FFNN uses StandardScaler and PCA before the neural network. Final model selection is still based on leave-one-group-out cross-validation rather than correlation alone.

> 特征选择基于物理可解释性、run-level 相关性检查、冗余分析和组级交叉验证。compact 特征集保留了与胎压有机械联系的振动幅值、能量和频率分布特征。bike type 被纳入输入，因为 FAT、ISY、MTB 的胎压范围和结构响应不同。rider weight 被纳入输入，因为它提供骑手载荷上下文。由于多个 compact 特征之间高度相关，最终 FFNN 在输入神经网络前使用 StandardScaler 和 PCA 降维。最终模型选择仍然由 leave-one-group-out 交叉验证决定，而不是只由相关性决定。

## Related Output Files / 相关输出文件

| 文件 / File | 说明 / Description |
|---|---|
| `tables/training_pool_candidate_feature_target_correlations.csv` | 132 个候选输入与 `pressure_bar` 的 Pearson / Spearman 相关性 / Pearson and Spearman correlations between 132 candidate inputs and `pressure_bar` |
| `tables/training_pool_final_input_feature_rationale.csv` | 28 个最终输入的相关性、冗余和选择理由 / correlations, redundancy, and selection rationale for the 28 final inputs |
| `tables/training_pool_final_input_correlation_matrix.csv` | 最终输入之间的 Spearman 相关性矩阵 / Spearman correlation matrix among final inputs |
| `tables/training_pool_high_redundancy_feature_pairs.csv` | 高冗余输入对 / highly redundant input pairs |
| `assets/training_pool_05_feature_target_correlation.png` | 最终输入与胎压相关性排行图 / final-input target-correlation ranking |
| `assets/training_pool_06_final_input_correlation_matrix.png` | 最终输入相关性矩阵图 / final-input correlation-matrix figure |
