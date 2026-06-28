# Step 02: Signal Cleaning And Feature Extraction / 信号清洗与特征提取

脚本 / Script:

`06_reproducible_pipeline/steps/02_extract_window_features.py`

核心模块 / Core modules:

- `signal_features.py`
- `feature_pipeline.py`

## Packages And Methods / 使用的包和方法

| 包 / Package | 用途 / Purpose |
|---|---|
| `pandas` | 读取 CSV、表格处理、缺失值处理 / read CSV files, handle tables, and process missing values |
| `numpy` | 数组计算、向量模长、FFT / array computation, vector magnitudes, and FFT |
| `scipy.signal` | Butterworth 带通滤波、`filtfilt` 零相位滤波 / Butterworth band-pass filtering and zero-phase `filtfilt` filtering |
| `scipy.stats` | skewness、kurtosis / skewness and kurtosis |

## Cleaning Steps / 清洗步骤

1. 把传感器列转换为数值。
   Convert sensor columns to numeric values.
2. 把 `inf` 和 `-inf` 替换为 `NaN`。
   Replace `inf` and `-inf` with `NaN`.
3. 对信号做线性插值。
   Apply linear interpolation to the signal.
4. 剩余缺失值用中位数填补。
   Fill remaining missing values with the median.
5. 用三轴向量模长替代原始 XYZ 轴，包括加速度模长和角速度模长。
   Replace raw XYZ axes with 3-axis vector magnitudes, including acceleration magnitude and angular-velocity magnitude.
6. 这样可以降低传感器安装方向对特征的影响。
   This reduces the influence of sensor mounting orientation.

## Active-Window Cropping / active window 怎么裁剪

1. 从 Measurement Details 读取每个 run 的 ride time。
   Read each run's ride time from Measurement Details.
2. 用两个加速度传感器的能量计算运动强度。
   Use the energy of the two acceleration sensors to estimate motion intensity.
3. 在整段信号里寻找总能量最大的连续片段。
   Search the full signal for the continuous segment with the highest total energy.
4. 这个片段长度等于 ride time。
   The segment length is equal to the ride time.
5. 这个片段就是 active window。
   That segment is used as the active window.

## Windowing / 窗口怎么切

- 窗口长度：1 秒。 / Window length: 1 second.
- overlap：50%。 / Overlap: 50%.
- 每个窗口提取一行特征。 / One feature row is extracted from each window.

## Extracted Features / 提取了哪些特征

时域特征 / Time-domain features:

- mean
- std
- RMS
- max absolute value
- 95 percentile absolute value
- peak-to-peak
- energy per second
- skewness
- kurtosis

频域特征 / Frequency-domain features:

- dominant frequency
- spectral centroid
- spectral entropy
- band power:
  - 0.5-3 Hz
  - 3-8 Hz
  - 8-15 Hz
  - 15-30 Hz

## What The 132 Candidate Input Features Are / 132 个候选输入特征是什么

这里的“候选输入特征”指 `training_pool_candidate_input_features.csv` 里保存的完整模型输入池。它不是最终模型一定全部使用的特征，而是后续模型选择时可以使用的基础特征集合。

"Candidate input features" refers to the full model-input pool saved in `training_pool_candidate_input_features.csv`. These are not all necessarily used by the final model; they are the base feature set available during model selection.

132 个候选输入特征由三部分组成：

The 132 candidate input features consist of three parts:

1. 传感器信号特征：128 个。 / Sensor signal features: 128.
2. bike type one-hot：3 个。 / Bike-type one-hot features: 3.
3. rider weight：1 个。 / Rider weight: 1.

计算方式 / Calculation:

`2 类信号 * 16 个基础特征 * 4 种双传感器聚合方式 + 3 个 bike one-hot + 1 个 rider weight = 132`

其中 2 类信号是 / The two signal types are:

- `acc`：两个 Sagemotion 传感器的加速度模长。 / acceleration magnitudes from the two Sagemotion sensors.
- `gyro`：两个 Sagemotion 传感器的角速度模长。 / angular-velocity magnitudes from the two Sagemotion sensors.

每类信号先在 Sensor 1 和 Sensor 2 上分别提取 16 个基础特征。 / For each signal type, 16 base features are first extracted separately on Sensor 1 and Sensor 2.

| 类型 / Type | 基础特征 / Base features |
|---|---|
| 时域 / Time domain | `mean`, `std`, `rms`, `max_abs`, `p95_abs`, `ptp`, `energy_per_s`, `skew`, `kurtosis` |
| 频域 / Frequency domain | `dom_freq`, `spectral_centroid`, `spectral_entropy`, `band_0p5_3_power`, `band_3_8_power`, `band_8_15_power`, `band_15_30_power` |

然后把两个传感器的同名特征聚合成 4 个值。 / Then the same feature from the two sensors is aggregated into four values.

| 聚合后缀 / Aggregation suffix | 含义 / Meaning |
|---|---|
| `_mean` | Sensor 1 和 Sensor 2 的平均值 / average of Sensor 1 and Sensor 2 |
| `_max` | 两个传感器中的较大值 / larger value between the two sensors |
| `_min` | 两个传感器中的较小值 / smaller value between the two sensors |
| `_absdiff` | 两个传感器之间的绝对差 / absolute difference between the two sensors |

Examples / 例子：

- `acc_rms_mean`：两个加速度传感器 RMS 的平均值。 / average RMS from the two acceleration sensors.
- `gyro_band_3_8_power_absdiff`：两个角速度传感器在 3-8 Hz 频带功率比例的差异。 / absolute difference between the two gyroscope sensors in the 3-8 Hz band-power ratio.
- `acc_energy_per_s_max`：两个加速度传感器中较大的单位时间能量。 / larger energy-per-second value between the two acceleration sensors.

最后加入 3 个 bike type one-hot 和 rider weight。 / Finally, three bike-type one-hot columns and rider weight are added:

- `bike_FAT`
- `bike_ISY`
- `bike_MTB`
- `rider_weight_kg`

完整 132 个候选输入列如下。 / The complete 132 candidate input columns are listed below.

```text
acc_band_0p5_3_power_mean
acc_band_0p5_3_power_max
acc_band_0p5_3_power_min
acc_band_0p5_3_power_absdiff
acc_band_15_30_power_mean
acc_band_15_30_power_max
acc_band_15_30_power_min
acc_band_15_30_power_absdiff
acc_band_3_8_power_mean
acc_band_3_8_power_max
acc_band_3_8_power_min
acc_band_3_8_power_absdiff
acc_band_8_15_power_mean
acc_band_8_15_power_max
acc_band_8_15_power_min
acc_band_8_15_power_absdiff
acc_dom_freq_mean
acc_dom_freq_max
acc_dom_freq_min
acc_dom_freq_absdiff
acc_energy_per_s_mean
acc_energy_per_s_max
acc_energy_per_s_min
acc_energy_per_s_absdiff
acc_kurtosis_mean
acc_kurtosis_max
acc_kurtosis_min
acc_kurtosis_absdiff
acc_max_abs_mean
acc_max_abs_max
acc_max_abs_min
acc_max_abs_absdiff
acc_mean_mean
acc_mean_max
acc_mean_min
acc_mean_absdiff
acc_p95_abs_mean
acc_p95_abs_max
acc_p95_abs_min
acc_p95_abs_absdiff
acc_ptp_mean
acc_ptp_max
acc_ptp_min
acc_ptp_absdiff
acc_rms_mean
acc_rms_max
acc_rms_min
acc_rms_absdiff
acc_skew_mean
acc_skew_max
acc_skew_min
acc_skew_absdiff
acc_spectral_centroid_mean
acc_spectral_centroid_max
acc_spectral_centroid_min
acc_spectral_centroid_absdiff
acc_spectral_entropy_mean
acc_spectral_entropy_max
acc_spectral_entropy_min
acc_spectral_entropy_absdiff
acc_std_mean
acc_std_max
acc_std_min
acc_std_absdiff
gyro_band_0p5_3_power_mean
gyro_band_0p5_3_power_max
gyro_band_0p5_3_power_min
gyro_band_0p5_3_power_absdiff
gyro_band_15_30_power_mean
gyro_band_15_30_power_max
gyro_band_15_30_power_min
gyro_band_15_30_power_absdiff
gyro_band_3_8_power_mean
gyro_band_3_8_power_max
gyro_band_3_8_power_min
gyro_band_3_8_power_absdiff
gyro_band_8_15_power_mean
gyro_band_8_15_power_max
gyro_band_8_15_power_min
gyro_band_8_15_power_absdiff
gyro_dom_freq_mean
gyro_dom_freq_max
gyro_dom_freq_min
gyro_dom_freq_absdiff
gyro_energy_per_s_mean
gyro_energy_per_s_max
gyro_energy_per_s_min
gyro_energy_per_s_absdiff
gyro_kurtosis_mean
gyro_kurtosis_max
gyro_kurtosis_min
gyro_kurtosis_absdiff
gyro_max_abs_mean
gyro_max_abs_max
gyro_max_abs_min
gyro_max_abs_absdiff
gyro_mean_mean
gyro_mean_max
gyro_mean_min
gyro_mean_absdiff
gyro_p95_abs_mean
gyro_p95_abs_max
gyro_p95_abs_min
gyro_p95_abs_absdiff
gyro_ptp_mean
gyro_ptp_max
gyro_ptp_min
gyro_ptp_absdiff
gyro_rms_mean
gyro_rms_max
gyro_rms_min
gyro_rms_absdiff
gyro_skew_mean
gyro_skew_max
gyro_skew_min
gyro_skew_absdiff
gyro_spectral_centroid_mean
gyro_spectral_centroid_max
gyro_spectral_centroid_min
gyro_spectral_centroid_absdiff
gyro_spectral_entropy_mean
gyro_spectral_entropy_max
gyro_spectral_entropy_min
gyro_spectral_entropy_absdiff
gyro_std_mean
gyro_std_max
gyro_std_min
gyro_std_absdiff
bike_FAT
bike_ISY
bike_MTB
rider_weight_kg
```

## Current Result / 当前结果

- 窗口特征行数：873。 / Window feature rows: 873.
- 候选输入特征：132。 / Candidate input features: 132.
- run-level summary：72 行。 / Run-level summary: 72 rows.

输出 / Outputs:

| 文件 / File | 说明 / Description |
|---|---|
| `training_pool_window_features.csv` | 窗口级特征表 / window-level feature table |
| `training_pool_active_window_summary.csv` | active window 裁剪结果 / active-window crop results |
| `training_pool_run_feature_summary.csv` | run 级特征摘要 / run-level feature summary |
| `training_pool_candidate_input_features.csv` | 模型候选输入列 / candidate model input columns |
