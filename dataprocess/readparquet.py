import pandas as pd
import os
import logging
from datetime import datetime
import sys

# 设置日志文件
log_filename = f"parquet_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename, 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 指定要读取的parquet文件路径
file_path = r"/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/dataprocess/output/airbot_dexterous_bimanual_dexterous_manipulation/data/chunk-000/episode_000000.parquet"
try:
    # 检查文件是否存在
    if not os.path.exists(file_path):
        logging.error(f"文件不存在: {file_path}")
        print(f"错误: 文件不存在: {file_path}")
        sys.exit(1)
    
    # 获取文件大小
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logging.info(f"文件大小: {file_size_mb:.2f} MB")
    
    # 读取parquet文件
    df = pd.read_parquet(file_path)
    
    # 记录数据框的基本信息
    logging.info(f"数据框形状: {df.shape}")
    logging.info(f"数据框列: {df.columns.tolist()}")
    logging.info(f"数据类型:\n{df.dtypes}")
    
    # 检查数据框大小，如果过大则只记录部分
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logging.info(f"数据框内存使用: {memory_usage_mb:.2f} MB")
    
    if memory_usage_mb > 100:  # 如果数据超过100MB
        logging.info("数据过大，只记录头部和尾部内容")
        logging.info("前10行:\n" + df.head(10).to_string())
        logging.info("后10行:\n" + df.tail(10).to_string())
    else:
        # 记录完整数据框内容
        logging.info("数据框完整内容:")
        logging.info("\n" + df.to_string())
    
    # 记录数值统计
    try:
        logging.info("数值统计:\n" + df.describe().to_string())
    except:
        logging.info("无法生成数值统计")
    
    print(f"数据已成功写入日志文件: {log_filename}")
    
except Exception as e:
    logging.error(f"处理文件时出错: {str(e)}")
    print(f"处理文件时出错: {str(e)}")