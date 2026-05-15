
# 用户的使用流程：
#   1. import XHandTeleOps
#   2. 实例化 XHandTeleOps
#   3. 调用它的方法：目前只提供三个方法
#           1. 从 versionpro 或 opentelevison 读数据
#           2. 调用 retarget_data 转化数据为 Xhand 灵巧手格式
#           3. 调用 send_data_xhand() 发送控制灵巧手移动
#           4. 调用 get_hand_full_info("hand_a") 或者 get_hand_full_info("hand_b") 读取手的位置，力矩等信息

# """以下是 receiver_main 的 demo 测试代码"""


import time
from xhand_tele_ops import XHandTeleOps
from open_tele_vision import OpenTeleVisionOps


# 读取 灵巧手数据 说明
#   可以通过 get_hand_full_info("hand_a") 或 get_hand_full_info("hand_b") 读取两只手的 所有数据，包括 位置信息，电流数据，扭矩数据，帕悉尼数据，温度数据等。


if __name__ == "__main__":
    # config.yaml 里的 connect_vision_pro 需要配置为 flase
    node = XHandTeleOps("config.yaml")

    # 获取 左右手
    resp_ht = node.get_hand_type("hand_a")
    if resp_ht and resp_ht["code"] == 0:
        hand_a_type = resp_ht['data']
        print(f"hand_a_type: {hand_a_type}")

    # 获取 SN 码
    resp_sn = node.get_serial_number("hand_a")
    if resp_sn and resp_sn["code"] == 0:
        hand_a_sn = resp_sn['data']
        print(f"hand_a_sn: {hand_a_sn}")

    # # 重置指尖传感器
    # resp_rs = node.reset_all_sensors("hand_a")
    # if resp_rs and resp_rs["code"] == 200:
    #     print(f"Reset hand_a all sensors successfully")

    node_tv = OpenTeleVisionOps()
    while True:
        # 从 opentelevison 读数据
        data = node_tv.get_data_from_open_tele_vision()
        
        # 转换数据，发送数据
        transform_data = node.retarget_data(data)
        node.send_data_xhand(transform_data)

        # 读取手部数据，示例代码
        print("\n\n")
        print("//================================")
        print("//Read various hand states")
        print("//================================")
        # 如果 同时 读写，需要 设置 force_update=False；如果只读不写，需要 设置 force_update=True
        resp = node.get_hand_full_info("hand_a", force_update=False, is_print=False)
        if resp and resp["code"] == 200:
            result = resp['data']
            print(f"关节位置 result['joint_position_dic']: {result['joint_position_dic']}")
            print(f"指尖原始压力 result['raw_pressure_dic']: {result['raw_pressure_dic']}")
            print(f"指尖合力值 result['calc_pressure_dic']: {result['calc_pressure_dic']}")
            print(f"帕西尼 指尖传感器平均温度值 result['sensor_temperature_dic']: {result['sensor_temperature_dic']}")
            print(f"关节驱动板温度 result['temperature_joint_dic']: {result['temperature_joint_dic']}")
            print(f"掌心板驱动板温度 result['temperature_tipboard']: {result['temperature_tipboard']}")
        # resp_b = node.get_hand_full_info("hand_b", is_print=True)
        # if resp_b and resp_b["code"] == 200:
        #     result = resp_b['data']
        #     print(f"关节位置 result['joint_position_dic']: {result['joint_position_dic']}")
        #     print(f"指尖原始压力 result['raw_pressure_dic']: {result['raw_pressure_dic']}")
        #     print(f"指尖合力值 result['calc_pressure_dic']: {result['calc_pressure_dic']}")
        #     print(f"帕西尼 指尖传感器平均温度值 result['sensor_temperature_dic']: {result['sensor_temperature_dic']}")
        #     print(f"关节驱动板温度 result['temperature_joint_dic']: {result['temperature_joint_dic']}")
        #     print(f"掌心板驱动板温度 result['temperature_tipboard']: {result['temperature_tipboard']}")

        # 便于查看 手部读取数据，可添加 延时函数
        # time.sleep(1)
