# Step 00-01: Environment, Labels, And File Inventory / 环境检查、标签和文件清单

## What Step 00 Does / Step 00 做什么

脚本 / Script:

`06_reproducible_pipeline/steps/00_check_environment.py`

主要检查 / Main checks:

- `Measurement_Campaign/G01-G06` 是否存在 / whether `Measurement_Campaign/G01-G06` exists.
- 每个组是否有 `Sagemotion Sensor Data` 和 `PhyPhox Data` / whether each group has `Sagemotion Sensor Data` and `PhyPhox Data`.
- Sagemotion CSV 是否为 72 个 / whether there are 72 Sagemotion CSV files.
- PhyPhox XLS 是否为 72 个 / whether there are 72 PhyPhox XLS files.
- Python 包是否可用：`numpy`、`pandas`、`scipy`、`sklearn`、`matplotlib`、`seaborn` / whether the required Python packages are available: `numpy`, `pandas`, `scipy`, `sklearn`, `matplotlib`, and `seaborn`.

## What Step 01 Does / Step 01 做什么

脚本 / Script:

`06_reproducible_pipeline/steps/01_build_labels_inventory.py`

使用的模块 / Modules used:

- `labels.py`
- `data_io.py`

主要操作 / Main operations:

1. 从 `project_config.py` 中的 `GROUP_DATA` 建立标签表。
   Build the label table from `GROUP_DATA` in `project_config.py`.
2. 标签包括 group、bike、P 编号、真实胎压、骑手质量、ride time。
   Labels include group, bike, P number, true tire pressure, rider weight, and ride time.
3. 扫描原始文件夹，建立文件清单。
   Scan the raw-data folders and build a raw-file inventory.
4. 检查每个文件名是否能解析出 group、bike、P 编号。
   Check whether each file name can be parsed into group, bike, and P number.
5. 对 Sagemotion CSV 做基本检查：行数、列数、采样率、时长、缺失单元格数量。
   Run basic checks on Sagemotion CSV files: row count, column count, sample rate, duration, and missing-cell count.

## Outputs / 输出

| 文件 / File | 说明 / Description |
|---|---|
| `training_pool_labels.csv` | 72 个 run 的标签表 / label table for 72 runs |
| `training_pool_raw_file_inventory.csv` | 144 个原始文件的清单 / inventory of 144 raw files |
| `training_pool_data_inventory_summary.json` | 数据清单摘要 / data-inventory summary |

## Current Result / 当前结果

- 标注 run：72。 / Labeled runs: 72.
- 原始文件：144。 / Raw files: 144.
- Sagemotion CSV：72，全部 OK。 / Sagemotion CSV: 72, all OK.
- PhyPhox XLS：72，全部 OK。 / PhyPhox XLS: 72, all OK.
