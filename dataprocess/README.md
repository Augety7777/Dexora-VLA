# LeRobot API 版本的 Airbot 数据转换器

这是使用 LeRobot 官方 API 重写的 Airbot 数据转换器，相比原版本有以下主要改进：

## 主要优势

1. **完全使用 LeRobot 官方 API**: 不再手动实现 parquet 写入、视频编码等功能
2. **标准 LeRobot v2.1 格式**: 输出的数据集完全符合 LeRobot v2.1 标准
3. **简化代码**: 删除了大量重复的自定义实现
4. **更好的错误处理**: 利用 LeRobot 的内置验证机制
5. **自动视频编码**: 使用 LeRobot 的高效视频编码管道

## 文件结构

```
dataprocess/
├── airbot_lerobot.py           # 新的 LeRobot API 版本处理器
├── test_lerobot_processor.py   # 测试脚本
├── airbot.py                   # 原始版本 (备份)
├── airbot_config.py            # 配置文件 (兼容)
└── README_lerobot.md           # 本文档
```

## 核心变化

### 删除的函数 (已用 LeRobot API 替代)

1. `create_video_from_images()` → `LeRobotDataset.encode_episode_videos()`
2. `update_episode_metadata()` → `LeRobotDatasetMetadata.save_episode()`
3. `create_meta_info()` → `LeRobotDatasetMetadata.create()`
4. `save_device_info()`, `save_label_info()` → LeRobot 自动处理
5. 手动 parquet 文件写入 → `LeRobotDataset.save_episode()`
6. chunk 目录管理 → LeRobot 自动处理

### 新增的函数

1. `create_lerobot_features()` - 定义数据集特征格式
2. `convert_frame_to_lerobot_format()` - 将帧数据转换为 LeRobot 格式
3. `setup_lerobot_dataset()` - 使用 LeRobot API 初始化数据集
4. `process_episode_with_lerobot()` - 使用 LeRobot API 处理 episode

### 修改的核心逻辑

原始流程:
```python
# 手动创建目录结构
# 手动写入 parquet 文件
# 手动创建视频文件
# 手动更新元数据
```

新流程:
```python
# 1. 创建 LeRobot 数据集
dataset = LeRobotDataset.create(repo_id, fps, features, root, robot_type)

# 2. 对每个 episode:
for episode in episodes:
    # 3. 对每一帧:
    for frame_data in episode_frames:
        frame = convert_frame_to_lerobot_format(frame_data)
        dataset.add_frame(frame, task, timestamp)
    
    # 4. 保存 episode (自动处理 parquet + 视频)
    dataset.save_episode()
```

## 使用方法

### 1. 基本使用

```bash
cd /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/dataprocess

# 使用新的 LeRobot API 版本
python airbot_lerobot.py
```

### 2. 运行测试

```bash
# 运行所有测试
python test_lerobot_processor.py

# 运行特定测试
python test_lerobot_processor.py --test features
python test_lerobot_processor.py --test tasks
python test_lerobot_processor.py --test dataset
```

### 3. 配置修改

数据转换器使用相同的 `airbot_config.py` 配置文件，主要配置项：

```python
# 数据路径
source_data_root = "/path/to/source/data"     # BSON + 图像文件夹
output_data_root = "/path/to/output/data"     # LeRobot 格式输出

# 数据集参数
fps = 20.0                                    # 采样频率
robot = "airbot_dexterous"                    # 机器人类型
overwrite = True                              # 是否覆盖现有数据集

# BSON 文件名
robot_bson_name = "episode_0.bson"            # 机械臂数据
hand_bson_name = "xhand_control_data.bson"    # 灵巧手数据
```

## 数据格式

### 输入格式 (保持不变)

```
source_data_root/
├── action8/
│   ├── episode_001/
│   │   ├── episode_0.bson          # 机械臂数据
│   │   ├── xhand_control_data.bson # 灵巧手数据
│   │   ├── camera_4/               # 高位相机
│   │   ├── camera_2/               # 左侧相机
│   │   └── camera_6/               # 右侧相机
│   └── episode_002/
└── action27/
```

### 输出格式 (LeRobot v2.1 标准)

```
output_data_root/airbot_dexterous_bimanual_dexterous_manipulation/
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   └── chunk-001/
├── meta/
│   ├── info.json                   # 数据集信息
│   ├── episodes.jsonl              # Episode 元数据
│   ├── stats.json                  # 数据统计
│   └── tasks.jsonl                 # 任务定义
└── videos/
    ├── chunk-000/
    │   ├── observation.images.camera_high/
    │   │   ├── episode_000000.mp4
    │   │   └── ...
    │   ├── observation.images.camera_left/
    │   ├── observation.images.camera_right/
    │   └── observation.images.camera_front/
    └── chunk-001/
```

## 数据特征定义

- **states**: 36维 (左臂6 + 右臂6 + 左手12 + 右手12)
- **actions**: 36维 (与 states 相同维度)
- **观测图像**: 4个相机 (camera_high, camera_left, camera_right, camera_front)
- **任务映射**: 支持 action_id → task_index 的自动映射

## 兼容性

- **LeRobot 版本**: v0.3.4 (v2.1 格式)
- **Python 版本**: 3.8+
- **保持原有配置**: 完全兼容现有的 `airbot_config.py`
- **数据源格式**: 保持 BSON + 图像文件夹不变

## 性能优化

1. **批量视频编码**: 支持批量编码以提升效率
2. **并行图像写入**: 支持多进程图像写入
3. **内存优化**: 使用 LeRobot 的内存管理机制
4. **自动验证**: 内置数据验证和错误检查

## 故障排除

### 常见问题

1. **导入错误**: 确保安装了正确版本的 lerobot
   ```bash
   pip install lerobot==0.3.4
   ```

2. **特征维度错误**: 检查关节数据是否符合预期维度
   - 机械臂: 每侧 6 DOF
   - 灵巧手: 每侧 12 DOF

3. **相机映射问题**: 检查 `airbot_config.py` 中的相机映射配置

4. **BSON 文件缺失**: 确保每个 episode 目录都包含必要的 BSON 文件

### 调试模式

启用详细日志:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 与原版本对比

| 功能 | 原版本 | LeRobot API 版本 |
|------|--------|------------------|
| parquet 写入 | 手动实现 | LeRobot API |
| 视频编码 | OpenCV + ffmpeg | LeRobot 管道 |
| 元数据管理 | 手动 JSONL | LeRobot 自动 |
| 目录结构 | 手动创建 | LeRobot 标准 |
| 数据验证 | 基础检查 | 完整验证 |
| 错误处理 | 有限 | 全面覆盖 |
| 代码复杂度 | 高 | 显著降低 |

## 迁移指南

如果你正在使用原版本，迁移到 LeRobot API 版本很简单：

1. 备份现有配置文件
2. 使用新的处理器: `python airbot_lerobot.py`
3. 验证输出结果
4. 删除原始版本 (可选)

新版本完全兼容现有的数据源和配置，无需任何修改。
