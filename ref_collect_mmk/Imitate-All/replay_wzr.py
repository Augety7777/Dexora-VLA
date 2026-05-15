import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from mmk2_types.types import MMK2Components, ImageTypes, ControllerTypes
from mmk2_types.grpc_msgs import (
    JointState,
    TrajectoryParams,
    MoveServoParams,
    GoalStatus,
    BaseControlParams,
    BuildMapParams,
    Pose3D,
    Twist3D,
    BaseChargeStationParams,
    ArrayStamped,
)
from airbot_py.airbot_mmk2 import AirbotMMK2
from pprint import pprint
import logging
import time
from bson import BSON
import json
from airbot_data.io import save_bson, load_bson
import sys


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create an instance of the AirbotMMK2 class
# change the ip address to the ip address of the robot
# mmk2 = AirbotMMK2(port=50055)
mmk2 = AirbotMMK2(ip="192.168.11.200")


start_joint_action = {
    MMK2Components.LEFT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, 0.724, 0.0]),
    MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, -0.724, 0.0]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.HEAD: JointState(position=[0.0, 0.18]),
    MMK2Components.SPINE: JointState(position=[0.0]),
}

# end_joint_action = {
#     MMK2Components.LEFT_ARM: JointState(position=[0.0, 0.0, 0.321, 0.005, 0.724, 0.0]),
#     MMK2Components.RIGHT_ARM: JointState(position=[0.0, 0.0, 0.324, 0.0, -0.724, 0.0]),
#     MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
#     MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
#     MMK2Components.HEAD: JointState(position=[0.0, -1]),
#     MMK2Components.SPINE: JointState(position=[0.15001875]),
# }

stop_joint_action = {
    MMK2Components.LEFT_ARM: JointState(position=[1.52, -2.1, 2.0, 1.4, 0.1, -0.62]),
    MMK2Components.RIGHT_ARM: JointState(position=[-1.52, -2.1, 2.0, -1.4, -0.1, 0.62]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=[1.0]),
    MMK2Components.HEAD: JointState(position=[0.0, 0.18]),
    MMK2Components.SPINE: JointState(position=[0.0]),
}


def get_robot_state():
    robot_state = mmk2.get_robot_state()
    if robot_state is None:
        logger.error("Failed to get robot state")
        return
    for _ in range(5):
        print("Robot state:")
        print("stamps:")
        pprint(robot_state.stamp)
        pprint(robot_state.joint_state.header.stamp)
        print("current_time", time.time())
        print("joint states:")
        pprint(robot_state.joint_state.name)
        pprint(robot_state.joint_state.position)
        pprint(robot_state.joint_state.velocity)
        pprint(robot_state.joint_state.effort)
        print("base state:")
        pprint(robot_state.base_state.pose)
        pprint(robot_state.base_state.velocity)
        print("robot poses:")
        pprint(robot_state.robot_pose)
        time.sleep(1)
        print("\n")


def control_trajectory_full():
    if (
        mmk2.set_goal(start_joint_action, TrajectoryParams()).value
        != GoalStatus.Status.SUCCESS
    ):
        logger.error("Failed to set goal")

    # time.sleep(1)

    # if (
    #     mmk2.set_goal(end_joint_action, TrajectoryParams()).value
    #     != GoalStatus.Status.SUCCESS
    # ):
    #     logger.error("Failed to set goal")
    # time.sleep(1)    
"""******run the functions******"""

# get_robot_state()
control_trajectory_full()

# 使用示例
file_path = 'data/raw/example/episode_0.bson'
bson_data = load_bson(file_path)
left_arm_frames = bson_data["data"]["/observation/left_arm/joint_state"]
right_arm_frames = bson_data["data"]["/observation/right_arm/joint_state"]
left_eef_frames = bson_data["data"]["/observation/left_arm_eef/joint_state"]
right_eef_frames = bson_data["data"]["/observation/right_arm_eef/joint_state"]
head_frames = bson_data["data"]["/observation/head/joint_state"]
spine_frames = bson_data["data"]["/observation/spine/joint_state"]
num_frames = min(
    len(left_arm_frames),
    len(right_arm_frames),
    len(left_eef_frames),
    len(right_eef_frames),
    len(head_frames),
    len(spine_frames),
)
print("num_frames:",num_frames)
joint_action = {
    MMK2Components.LEFT_ARM: JointState(position=left_arm_frames[0]["data"]["pos"]),
    MMK2Components.RIGHT_ARM: JointState(position=right_arm_frames[0]["data"]["pos"]),
    MMK2Components.LEFT_ARM_EEF: JointState(position=left_eef_frames[0]["data"]["pos"]),
    MMK2Components.RIGHT_ARM_EEF: JointState(position=right_eef_frames[0]["data"]["pos"]),
    MMK2Components.HEAD: JointState(position=head_frames[0]["data"]["pos"]),
    MMK2Components.SPINE: JointState(position=spine_frames[0]["data"]["pos"]),
}
mmk2.set_goal(joint_action, TrajectoryParams())

print("[mmk] 初始化完成，等待主控信号...")
print("READY", flush=True)

signal = sys.stdin.readline().strip()

if signal.upper() == "START":
    print("收到 START，开始执行任务")
    start_time = time.time()
    # print("start_time:",start_time)

    timestamp = left_arm_frames[0]['t']
    i = 1
    last_played_frame = -1

    while i < num_frames:
        # 当前已经过的时间（ms）
        elapsed_ms = (time.time() - start_time) * 1000
        target_timestamp = elapsed_ms + timestamp  # 加回原始时间戳基准

        # 向后找到最接近的帧
        while i < num_frames - 1 and left_arm_frames[i+1]["t"] <= target_timestamp:
            i += 1

        if i == last_played_frame:
            continue  # 当前帧已经播放过了，不重复发

        # 构建动作
        joint_action = {
            MMK2Components.LEFT_ARM: JointState(position=left_arm_frames[i]["data"]["pos"]),
            MMK2Components.RIGHT_ARM: JointState(position=right_arm_frames[i]["data"]["pos"]),
            MMK2Components.LEFT_ARM_EEF: JointState(position=left_eef_frames[i]["data"]["pos"]),
            MMK2Components.RIGHT_ARM_EEF: JointState(position=right_eef_frames[i]["data"]["pos"]),
            MMK2Components.HEAD: JointState(position=head_frames[i]["data"]["pos"]),
            MMK2Components.SPINE: JointState(position=spine_frames[i]["data"]["pos"]),
        }

        # print(joint_action)

        t0 = time.perf_counter()
        # mmk2.set_goal(joint_action, MoveServoParams())
        mmk2.set_goal(joint_action, MoveServoParams())
        t1 = time.perf_counter()
        # print(f"第{i}帧 [⏱] Action dispatch took {t1 - t0:.6f} seconds")

        last_played_frame = i
    print("[mmk] 结束任务")