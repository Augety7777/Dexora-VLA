# LeRobot Split & Merge Processor

一个用于机器人数据集的拆分与合并的命令行工具集：
- 拆分（split）：从单个数据集中选择前 N 帧或前 N 个 episode，生成一个子集
- 合并（merge）：从多个源数据集中选择并合并为一个统一的数据集

本项目结构：
- `lerobot_dataset_lib.py`：函数库，提供选择、统计合并、视频与数据拷贝、维度与索引对齐等底层能力
- `split_merge_dataset.py`：命令行入口，解析参数并调用 `lerobot_dataset_lib.py` 的函数

## 安装与环境

运行时需要以下环境与包：
- Python 3.8+（推荐 3.10）
- pandas（用于读取/写入 Parquet）
- 至少一个 Parquet 后端：pyarrow 或 fastparquet（二选一或都安装）
- numpy

使用 pip 安装：
```bash
pip install -U pandas numpy pyarrow fastparquet
```

可选：使用 conda 创建隔离环境：
```bash
conda create -n lerobot-dp python=3.10
conda activate lerobot-dp
conda install -c conda-forge pandas numpy pyarrow fastparquet
```
备注：若未安装任何 Parquet 引擎，pandas 的 `read_parquet`/`to_parquet` 将失败。
## 使用

命令行入口：
```bash
python split_merge_dataset.py --help
```

### 子命令

- `split`：拆分单个数据集为一个子集
- `merge`：合并多个数据集为一个数据集

## 通用约定
- 数据集目录格式需包含 `meta/info.json`、`meta/episodes.jsonl`、`meta/episodes_stats.jsonl`、`meta/tasks.jsonl` 等文件，视频与数据帧按 `info.json` 的结构组织。
- 起始偏移优先级：`start_entries`（帧偏移）优先于 `start_episodes`（episode 偏移）；设置了帧偏移时按整 episode 对齐。
- 截断优先级：`max_entries`（帧数）优先于 `max_episodes`。
- 维度处理：对 `observation.state` 与 `action` 进行零填充到 `max_dim` 并更新 `info.json` 的 `features.shape`；未设置时取所有源的最大维度。
- 帧率处理：`fps` 未提供时从源数据集 `meta/info.json` 读取并写入输出数据集。

## 参数说明

### split 参数（拆分）

- `--input`：输入数据集路径（必填）
- `--output`：输出数据集路径（必填）
- `--max_entries`：最大帧数限制（按帧数截断）
- `--max_episodes`：最大 episode 限制（按 episode 截断）
- `--fps`：输出数据集的帧率；未提供时从 `input/meta/info.json` 读取
- `--max_dim`：目标维度；未提供时使用源数据集中的最大维度
- `--start_entries`：起始帧偏移（跳过前 N 帧，优先级高于起始 episode）
- `--start_episodes`：起始 episode 偏移（跳过前 N 个 episode）

示例（按 episode 数量拆分）：
```bash
python split_merge_dataset.py split \
  --input /mnt/nas/synnas/docker2/robocoin-datasets/realman_rmc_aidal_box_up_down \
  --output /home/kemove/robotics-data-processor/lerobot/box_up_down \
  --start_episodes 2 \
  --max_episodes 300 \
  --fps 20 \
  --max_dim 32
```

示例（从第 20,000 帧开始取 10,000 帧，按整 episode 对齐）：
```bash
python split_merge_dataset.py split \
  --input /path/to/ds \
  --output /path/to/out \
  --start_entries 20000 \
  --max_entries 10000 \
  --fps 20 \
  --max_dim 32
```

### merge 参数（合并）

- `--sources`：源数据集路径列表（可选，支持多个路径）
- `--sources_dir`：源数据集父目录（扫描一、二级子目录，自动发现含 `meta/info.json` 的 Lerobot 数据集）
- `--output`：输出数据集路径（必填）
- `--max_episodes`：最大 episode 限制（从整体合并后按数量截断）
- `--fps`：输出数据集的帧率；未提供时从第一个有效源的 `meta/info.json` 读取，缺失则回退为 20
- `--max_dim`：目标维度；未提供时使用所有源数据集中的最大维度统一并零填充
- `--start_entries`：起始帧偏移（跨多源整体跳过前 N 帧，按整 episode 对齐）
- `--start_episodes`：起始 episode 偏移（跨多源整体跳过前 N 个 episode）

示例（显式列出多源）：
```bash
python split_merge_dataset.py merge \
  --sources /home/kemove/robotics-data-processor/lerobot/basket_storage_banana \
            /home/kemove/robotics-data-processor/lerobot/basket_storage_fruit \
            /home/kemove/robotics-data-processor/lerobot/plate_storage \
  --output /home/kemove/robotics-data-processor/lerobot/agilex_merged \
  --max_episodes 550 \
  --fps 20 \
  --max_dim 32
```

示例（自动发现一二级子目录）：
```bash
python split_merge_dataset.py merge \
  --sources_dir /mnt/nas/robodatasets \
  --output /mnt/nas/merged_ds \
  --max_episodes 300 \
  --fps 20 \
  --max_dim 32
```

示例（起始偏移 + 合并数量限制）：
```bash
python split_merge_dataset.py merge \
  --sources_dir /path/to/datasets_root \
  --output /path/to/merged \
  --start_entries 5000 \
  --max_episodes 500 \
  --fps 20 \
  --max_dim 32
```

### 自动发现规则
- 1级：检查 `sources_dir` 的直接子目录是否包含 `meta/info.json`；存在则视为数据集
- 2级：若 1 级子目录不是数据集，则检查其子目录是否包含 `meta/info.json`
- 结果集合会去重并排序；若未发现任何有效数据集则报错

## 输出内容

生成的数据集目录包含：
- `meta/episodes.jsonl`：所选 episode 列表与重编号
- `meta/episodes_stats.jsonl`：逐 episode 的统计（维度对齐）
- `meta/tasks.jsonl`：仅保留实际使用的任务（task_index 过滤）
- `meta/stats.json`：全局统计，合并并基于 episode 统计重算（均值/计数）
- `meta/info.json`：汇总信息（总帧数、总 episode、splits、features.shape、fps、chunks_size、total_videos 等）
- `videos/`：根据 `info.json` 的 `video_path` 模板拷贝的视频
- `data/chunk-XXX/episode_YYYYYY.parquet`：按 chunk 分组的帧数据，维度与索引已对齐

## 目录结构框架

```plaintext
/path/to/output_dataset
.
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── tasks.jsonl
│   └── stats.json
├── videos/
│   └── chunk-000/
│       └── {video_key}/episode_{episode_index:06d}.mp4
└── data/
    └── chunk-000/
        └── episode_{episode_index:06d}.parquet
```

- meta/info.json
  - 关键字段：`total_episodes`、`total_frames`、`total_tasks`、`total_chunks`、`fps`、`features`、`chunks_size`、`total_videos`
  - `splits` 自动设置为 `{"train": "0:total_episodes"}`（包含全部 episode）
  - `video_path` 示例：`videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4`
  - `features` 中 `observation.state` 和 `action` 的 `shape` 对齐为 `[max_dim]`

- meta/episodes.jsonl
  - 每行一个 episode 对象，`episode_index` 已重编号为从 0 连续递增；保留 `length`、`task_index` 等。

- meta/episodes_stats.jsonl
  - 每行一个 episode 的统计；包含 `min/max/mean/std/count`，并对 `observation.state` 与 `action` 填充到 `max_dim`。

- meta/tasks.jsonl
  - 任务列表（仅保留实际使用的 `task_index`）；多源任务按描述去重并重编号。

- meta/stats.json
  - 全局统计：合并各源统计并基于逐 episode 统计重算（均值/计数），维度统一到 `max_dim`。

- videos/
  - 目录结构与 `info["video_path"]` 一致，按 `chunk` 与 `video_key` 组织；文件名包含合并后的 `episode_index`。

- data/
  - `data/chunk-XXX/episode_YYYYYY.parquet`：按 `chunks_size` 分块；`observation.state` 和 `action` 已零填充到 `max_dim`；`index` 为全局帧索引，`episode_index` 为合并后编号。

### 命名与编号约定
- `episode_index`：0 基，文件名零填充到 6 位（如 `episode_000123`）
- `chunk`：按 `episode_index // chunks_size` 计算（例如 `chunk-000`）
- `splits`：默认 `train: 0:total_episodes`，如需 `val`/`test` 可自行修改 `info.json`

## 注意事项

- 运行拷贝与统计时需要安装 `pandas` 与 Parquet 后端（`pyarrow` 或 `fastparquet`）。
- 如果某源数据集 `meta/info.json` 中缺少 `video_path`，视频拷贝会跳过；请确认该键存在或在库中添加保护逻辑。
- 保证不同源数据集的 `features` 定义一致，特别是视频键（`dtype == "video"`）、`chunks_size` 等，以避免拷贝路径问题。
- 其它自动项：`tasks.jsonl`按实际使用过滤；`splits`自动设置为`train: 0:total_episodes`；`chunks_size`从源 `info.json`继承。
