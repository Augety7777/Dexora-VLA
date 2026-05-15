#!/usr/bin/env python3
"""
将 `data/ours/final_11.29` 下的数据转换为 LeRobot 数据集。

使用:
    cd Dex-RDT
    python -m dataprocess.convert_final_1129_to_lerobot

输出:
    data/ours/lerobot_output_final_11_29/airbot_{pick_and_place,articulation,assemble,dexterous}
"""

from pathlib import Path

from .airbot_lerobot_restructured import AirbotLeRobotRestructuredProcessor
from .airbot_config_final_1129 import AirbotFinal1129Config


def main():
    # 创建配置
    config = AirbotFinal1129Config()
    config.print_summary()

    # 创建处理器
    processor = AirbotLeRobotRestructuredProcessor(config)

    # 这里保留并行处理 4 个数据集的能力
    # 如果机器资源紧张，可以把 parallel_datasets 改为 False
    processor.run(parallel_datasets=True)


if __name__ == "__main__":
    import os

    # 冒烟测试模式:
    #   环境变量 FINAL1129_SMOKE=1 时，只跑 very small case，验证流程不报错。
    if os.environ.get("FINAL1129_SMOKE", "0") == "1":
        config = AirbotFinal1129Config()
        config.print_summary()
        # 只取每个 category 最多 1 个 episode，用于快速检查
        config.max_episodes_per_category = 1
        processor = AirbotLeRobotRestructuredProcessor(config)
        # 随便选一个类别（如果没有数据则会直接跳过，不会报错）
        processor.process_single_category("assemble")
    else:
        main()


