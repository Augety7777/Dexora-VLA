#!/usr/bin/env python3
"""
脚本用于可视化AIRBOT BSON数据中指定关节的状态(state)和动作(action)轨迹。

python scripts/plot_bson_trajectories.py --bson-path data/ours/action1/episode_0.bson --output trajectory.png
"""

import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import bson

def load_bson(bson_path):
    """Load a bson file and return its contents."""
    with open(bson_path, "rb") as f:
        data = bson.decode(f.read())
    return data

def extract_state_and_action_from_bson(bson_data):
    left_obs_arm_key = "/observation/left_arm/joint_state"
    right_obs_arm_key = "/observation/right_arm/joint_state"
    left_act_arm_key = "/action/left_arm/joint_state"
    right_act_arm_key = "/action/right_arm/joint_state"
    left_obs_eef_key = "/observation/left_arm_eef/joint_state"
    right_obs_eef_key = "/observation/right_arm_eef/joint_state"
    left_act_eef_key = "/action/left_arm_eef/joint_state"
    right_act_eef_key = "/action/right_arm_eef/joint_state"
    
    # Get the number of frames
    frame_num = len(bson_data["data"][left_obs_arm_key])
    
    # Initialize state and action arrays
    state = np.zeros((frame_num, 14), dtype=np.float32)  # 14 motors (7 for each arm)
    velocity = np.zeros((frame_num, 14), dtype=np.float32)
    effort = np.zeros((frame_num, 14), dtype=np.float32)
    
    action = np.zeros((frame_num, 12), dtype=np.float32)
    
    for modality, name in zip([state, velocity, effort], ["pos", "vel", "eff"]):
        # Extract joint positions and gripper data
        for i in range(frame_num):
            # Extract joint positions for left arm observation
            modality[i, 0:6] = bson_data["data"][left_obs_arm_key][i]["data"][name]
            
            # Extract joint positions for right arm observation
            modality[i, 7:13] = bson_data["data"][right_obs_arm_key][i]["data"][name]
            
            # Extract gripper position for left arm observation
            modality[i, 6:7] = bson_data["data"][left_obs_eef_key][i]["data"][name]

            # Extract gripper position for right arm observation
            modality[i, 13:14] = bson_data["data"][right_obs_eef_key][i]["data"][name]
            
    # for i in range(frame_num):
    #     # Extract joint positions for left arm action
    #     action[i, 0:5] = bson_data["data"][left_act_arm_key][i]["data"]["pos"]
        
    #     # Extract joint positions for right arm action
    #     action[i, 5:10] = bson_data["data"][right_act_arm_key][i]["data"]["pos"]
        
    #     # Extract gripper position for left arm action
    #     action[i, 10:11] = bson_data["data"][left_act_eef_key][i]["data"]["pos"]
        
    #     # Extract gripper position for right arm action
    #     action[i, 11:12] = bson_data["data"][right_act_eef_key][i]["data"]["pos"]
    
    return state, velocity, effort, action

# 定义关节名称和其在数组中的索引映射
JOINT_INDICES = {
    "left_waist": 0,
    "left_shoulder": 1,
    "left_elbow": 2,
    "left_forearm_roll": 3,
    "left_wrist_angle": 4,
    "left_wrist_rotate": 5,
    "left_gripper": 6,
    "right_waist": 7,
    "right_shoulder": 8,
    "right_elbow": 9,
    "right_forearm_roll": 10,
    "right_wrist_angle": 11,
    "right_wrist_rotate": 12,
    "right_gripper": 13,
}


def plot_joint_trajectories(bson_path, joints=None, output_file=None, show_plot=True):
    """
    绘制指定关节的状态和动作轨迹。
    
    参数:
        bson_path: BSON数据文件路径
        joints: 要绘制的关节名称列表，如果为None则绘制所有关节
        output_file: 输出图像文件路径，如果为None则不保存
        show_plot: 是否显示图像
    """
    # 确保bson_path是Path对象
    bson_path = Path(bson_path)
    
    # 如果输入是目录，查找data.bson文件
    if bson_path.is_dir():
        bson_file = bson_path / "data.bson"
    else:
        bson_file = bson_path
    
    # 检查文件是否存在
    if not bson_file.exists():
        raise FileNotFoundError(f"找不到BSON文件: {bson_file}")
    
    print(f"加载BSON数据: {bson_file}")
    bson_data = load_bson(bson_file)
    
    # 提取状态和动作数据
    state, velocity, effort, action = extract_state_and_action_from_bson(bson_data)

    state = state[:100]
    action = action[:100]

    print("first state:", state[0])
    
    # 如果没有指定关节，使用所有关节
    if joints is None:
        joints = list(JOINT_INDICES.keys())
    
    # 验证所有指定的关节名称是否有效
    for joint in joints:
        if joint not in JOINT_INDICES:
            raise ValueError(f"未知关节名称: {joint}。可用关节: {', '.join(JOINT_INDICES.keys())}")
    
    # 准备绘图
    fig, axes = plt.subplots(len(joints), 1, figsize=(12, 3*len(joints)), sharex=True)
    if len(joints) == 1:
        axes = [axes]  # 确保axes始终是列表
    
    # 创建时间数组
    time = np.arange(len(state))
    
    # 为每个关节绘制状态和动作
    for i, joint in enumerate(joints):
        joint_idx = JOINT_INDICES[joint]
        ax = axes[i]
        
        # 绘制状态和动作
        ax.plot(time, state[:, joint_idx], label='State', color='blue')
        # ax.plot(time, action[:, joint_idx], label='Action', color='red', linestyle='--')
        
        # 添加图例和标签
        ax.set_title(f'Joint: {joint}')
        ax.set_ylabel('Position (radians)')
        ax.legend()
        ax.grid(True)
    
    # 为最后一个子图添加x轴标签
    axes[-1].set_xlabel('Time steps')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {output_file}")
    
    # 显示图像
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='绘制BSON数据中指定关节的状态和动作轨迹。'
    )
    parser.add_argument(
        '--bson-path', 
        type=str, 
        required=True, 
        help='BSON数据文件或包含data.bson的目录路径'
    )
    parser.add_argument(
        '--joints', 
        type=str, 
        nargs='+', 
        default=None, 
        help='要绘制的关节名称，如果未指定则绘制所有关节'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None, 
        help='输出图像文件路径，如果未指定则不保存'
    )
    parser.add_argument(
        '--no-show', 
        action='store_true', 
        help='不显示图像，只保存'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 绘制轨迹
    plot_joint_trajectories(
        args.bson_path,
        args.joints,
        args.output,
        not args.no_show
    )


if __name__ == "__main__":
    main()
