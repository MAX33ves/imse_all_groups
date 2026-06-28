# Training-Pool Data Processing And EDA Report / Training-Pool 数据处理与 EDA 报告

## Current Data Rule / 当前数据规则

- All P1-P4 runs are used as local training data. / P1-P4 全部作为本地训练数据使用。
- The instructor hidden data is the true external test set. / 老师手里的隐藏数据才是真正外部测试集。
- Local performance is estimated with leave-one-group-out cross-validation. / 本地效果使用留一组交叉验证估计。

## Data Scale / 数据规模

| Item / 项目 | Value / 数值 |
|---|---:|
| Labeled runs / 标注 run 数 | 72 |
| Groups / 组数 | 6 |
| Bike types / 单车类型 | 3 |
| Window feature rows / 窗口特征行数 | 873 |
| Parseable Sagemotion CSV files / 可解析 Sagemotion CSV | 72 |
| Parseable PhyPhox XLS files / 可解析 PhyPhox XLS | 72 |

## Raw-Data Scope / 原始数据使用范围说明

The current model extracts features only from Sagemotion CSV files.  
当前模型特征只从 Sagemotion CSV 提取。

PhyPhox XLS files are checked in the file inventory, but they are not included in the current FFNN feature matrix.  
PhyPhox XLS 已在文件清单中检查完整性，但没有进入当前 FFNN 特征矩阵。

The reason is that Sagemotion provides two fixed mounted sensors with both acceleration and angular-velocity channels. Compared with phone-based PhyPhox data, it is more suitable as the main structural vibration input.  
这样做的理由是 Sagemotion 提供两个固定安装传感器，并包含加速度和角速度通道；相比手机 PhyPhox，它更适合作为主要结构振动输入。

If the final report needs to cover PhyPhox explicitly, a separate PhyPhox baseline should be added, or PhyPhox should be described as an optional/future validation source.  
如果报告需要覆盖 PhyPhox，应补一个单独的 PhyPhox baseline，或明确把 PhyPhox 作为备用/未来验证数据源。

## Cleaning And Feature Processing / 数据清洗和特征处理

- CSV files are read with `pandas`, and sensor columns are converted to numeric values. / 使用 `pandas` 读取 CSV，并把传感器列转换为数值。
- `numpy` is used for vector magnitudes, window indices, and FFT calculations. / 使用 `numpy` 处理向量模长、窗口索引和 FFT。
- `scipy.signal.butter/filtfilt` is used for band-pass filtering. / 使用 `scipy.signal.butter/filtfilt` 做带通滤波。
- Missing and infinite values are interpolated first, then remaining missing values are filled with medians. / 对缺失值和无穷值先插值，再用中位数填补。
- The active window is cropped by acceleration energy; its duration comes from Measurement Details ride time. / 通过加速度能量裁剪 active window，窗口长度来自 Measurement Details 的 ride time。
- Within each active window, 1 s windows with 50% overlap are extracted. / 在 active window 内使用 1 秒窗口和 50% overlap。
- Each window produces time-domain statistics and frequency-domain features. / 每个窗口提取时域统计量和频域特征。

## PCA

PCA is used for EDA visualization and candidate model dimensionality reduction; it is not a label.  
PCA 只用于 EDA 可视化和模型候选降维，不是标签。

The EDA PCA is fitted on all local P1-P4 data because all of these runs belong to the local training pool.  
当前 EDA PCA 使用全部本地 P1-P4 数据拟合，因为这些数据都属于训练池。

True local generalization is estimated later through leave-one-group-out CV.  
真正的本地泛化估计在模型步骤中用 leave-one-group-out 完成。

## Feature Explanation And Correlation Evidence / 特征解释和相关性证据

- Correlation analysis uses run-level medians, not raw window rows, because windows from the same run are highly related. / 相关性分析使用 run-level 中位数，而不是窗口级行，因为同一个 run 内的窗口高度相关。
- Pearson and Spearman correlations explain linear and monotonic relationships with `pressure_bar`. / 输出 Pearson 和 Spearman 相关性，用来说明特征与 `pressure_bar` 的线性关系和单调关系。
- The final-input correlation matrix identifies redundancy among selected inputs. / 相关性矩阵用于识别最终输入特征之间的冗余结构。
- Correlation is explanatory evidence only; final model choice is still decided by leave-one-group-out CV. / 相关性只作为解释和候选筛选证据；最终模型仍由 leave-one-group-out CV 决定。
- Candidate inputs: 132. / 候选输入数：132。
- Current final feature space: `compact_weight`, 28 inputs. / 当前最终输入空间：`compact_weight`，共 28 个输入。
- Rider weight included: True. / 是否包含 rider weight：True。
