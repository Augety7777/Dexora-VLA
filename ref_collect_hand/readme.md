# 手的检测

cd ~/teleop_software_pkg/

conda activate xhand_tele_env

sudo setcap cap_net_raw+ep $(readlink -f $(which python3))

密码：air123

python3 receiver_main_demo_vision_pro_teleop.py