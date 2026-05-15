
# 转换数据
cd /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/dataprocess
python convert_action190_to_lerobot.py \
  --source /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/new_action190/action264 \
  --output /baai-cwm-backup/cwm/zongzheng.zhang/Dex-RDT/data/3action190 \
  --fps 20

cd /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/dataprocess
python convert_action190_to_lerobot.py \
  --source /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/action201\
  --output /baai-cwm-backup/cwm/zongzheng.zhang/Dex-RDT/data/action201\
  --fps 20

python convert_action190_to_lerobot.py \
  --source /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/action265 \
  --output /baai-cwm-backup/cwm/zongzheng.zhang/Dex-RDT/data/action265  \
  --fps 20

python convert_action190_to_lerobot.py \
  --source /baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/action266 \
  --output /baai-cwm-backup/cwm/zongzheng.zhang/Dex-RDT/data/action266  \
  --fps 20



python convert_action190_to_lerobot.py \
  --source /bci-vepfs/users/guest1/data/action1 \
  --output /bci-vepfs/users/guest1/dataset/action2


