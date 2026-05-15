from bson import BSON
import json
from airbot_data.io import save_bson, load_bson
def print_bson_file(file_path):
    """
    打印 BSON 文件的内容
    :param file_path: BSON 文件的路径
    """
    try:
        # 打开 BSON 文件并读取内容
        with open(file_path, 'rb') as f:
            bson_data = BSON(f.read())
        
        # 解析 BSON 数据为字典
        data = BSON.decode(bson_data)
        
        # 将二进制数据转换为可序列化的格式
        def convert_to_serializable(obj):
            if isinstance(obj, bytes):
                return obj.hex()  # 将二进制数据转换为十六进制字符串
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # 转换数据
        serializable_data = convert_to_serializable(data)
        
        # 打印解析后的数据
        print(json.dumps(serializable_data, indent=4))  # 使用 json.dumps 格式化输出
    except Exception as e:
        print(f"读取或解析 BSON 文件时出错: {str(e)}")

# 使用示例
file_path = 'data/raw/example/episode_0.bson'
# print_bson_file(file_path)
bson = load_bson(file_path)
# print(bson.keys())

# print(bson["metadata"].keys())
# print(bson["id"])
# print(bson["metadata"]["topics"].keys())
print(bson["data"].keys())
# print("timestamp:",bson["timestamp"])
# print(bson["data"]['/observation/left_arm/joint_state'][0]['t'])
# print(bson["data"]['/observation/left_arm/joint_state'][1]['t'])
print(bson["data"]['/observation/left_arm/joint_state'][2])
print(bson["data"]['/action/left_arm/joint_state'][2])
print(bson["data"]['/action/left_arm/joint_position'][2])
# print(bson["data"]['/observation/left_arm/joint_state'][0]['t']-bson["timestamp"])
# print(bson["data"]['/observation/left_arm/joint_state'][1]['t']-bson["data"]['/observation/left_arm/joint_state'][0]['t'])
# print(bson["data"]['/observation/left_arm/joint_state'][2]['t']-bson["data"]['/observation/left_arm/joint_state'][1]['t'])
# print(bson["data"]['/observation/left_arm/joint_state'][0]['data']['pos'])
# print(bson["data"]['/observation/right_arm/joint_state'][0]['data']['pos'])
# print(bson["data"]['/observation/left_arm_eef/joint_state'][0]['data']['pos'])
# print(bson["data"]['/observation/left_arm_eef/joint_state'][0]['data']['pos'])
# print(bson["data"]['/observation/head/joint_state'][0]['data']['pos'])
# print(bson["data"]['/observation/spine/joint_state'][0]['data']['pos'])
# print(bson["data"]['/observation/spine/joint_state'])
# print(bson["data"]['/images/head_camera'])
# left_arm_frames = bson["data"]["/observation/left_arm/joint_state"]
# right_arm_frames = bson["data"]["/observation/right_arm/joint_state"]
# left_eef_frames = bson["data"]["/observation/left_arm_eef/joint_state"]
# right_eef_frames = bson["data"]["/observation/right_arm_eef/joint_state"]
# head_frames = bson["data"]["/observation/head/joint_state"]
# spine_frames = bson["data"]["/observation/spine/joint_state"]
# print(len(left_arm_frames))
# print(len(right_arm_frames))
# print(len(left_eef_frames))
# print(len(left_eef_frames))
# print(len(head_frames))
# print(len(spine_frames))