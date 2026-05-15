import bson
import os
import logging
from datetime import datetime
import sys
import json

# 设置日志文件
log_filename = f"bson_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename, 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 指定要读取的bson文件路径
file_path = r"/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/action190/episode_0/xhand_control_data.bson"

try:
    # 检查文件是否存在
    if not os.path.exists(file_path):
        logging.error(f"文件不存在: {file_path}")
        print(f"错误: 文件不存在: {file_path}")
        sys.exit(1)
    
    # 获取文件大小
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logging.info(f"文件大小: {file_size_mb:.2f} MB")
    print(f"正在读取文件: {file_path} (大小: {file_size_mb:.2f} MB)")
    
    # 读取bson文件
    with open(file_path, 'rb') as f:
        file_content = f.read()
    
    # 解码BSON数据
    data_list = []
    offset = 0
    document_count = 0
    
    while offset < len(file_content):
        try:
            # 读取单个BSON文档
            doc_size = int.from_bytes(file_content[offset:offset+4], byteorder='little')
            if offset + doc_size > len(file_content):
                break
                
            doc_bytes = file_content[offset:offset+doc_size]
            doc = bson.decode(doc_bytes)
            data_list.append(doc)
            
            offset += doc_size
            document_count += 1
            
        except Exception as e:
            logging.warning(f"解码文档 {document_count} 时出错: {str(e)}")
            break
    
    logging.info(f"成功解码 {document_count} 个BSON文档")
    
    # 记录数据的基本信息
    logging.info(f"文档总数: {len(data_list)}")
    
    if data_list:
        # 记录第一个文档的结构
        first_doc = data_list[0]
        logging.info(f"第一个文档的键: {list(first_doc.keys())}")
        
        # 分析数据结构
        for key, value in first_doc.items():
            logging.info(f"键 '{key}' 的类型: {type(value).__name__}")
    
    # 根据数据大小决定记录策略
    if file_size_mb > 50:  # 如果文件超过50MB
        logging.info("文件过大，只记录前5个和后5个文档")
        
        # 记录前5个文档
        for i, doc in enumerate(data_list[:5]):
            logging.info(f"文档 {i}:")
            try:
                # 将文档转换为JSON格式以便记录
                doc_json = json.dumps(doc, default=str, indent=2, ensure_ascii=False)
                logging.info(doc_json)
            except Exception as e:
                logging.info(f"无法序列化文档 {i}: {str(e)}")
                logging.info(f"文档内容: {str(doc)[:1000]}...")
        
        # 记录后5个文档
        if len(data_list) > 5:
            for i, doc in enumerate(data_list[-5:], start=len(data_list)-5):
                logging.info(f"文档 {i}:")
                try:
                    doc_json = json.dumps(doc, default=str, indent=2, ensure_ascii=False)
                    logging.info(doc_json)
                except Exception as e:
                    logging.info(f"无法序列化文档 {i}: {str(e)}")
                    logging.info(f"文档内容: {str(doc)[:1000]}...")
    else:
        # 记录所有文档
        logging.info("记录所有BSON文档:")
        for i, doc in enumerate(data_list):
            logging.info(f"文档 {i}:")
            try:
                doc_json = json.dumps(doc, default=str, indent=2, ensure_ascii=False)
                logging.info(doc_json)
            except Exception as e:
                logging.info(f"无法序列化文档 {i}: {str(e)}")
                logging.info(f"文档内容: {str(doc)[:1000]}...")
    
    # 统计信息
    if data_list:
        logging.info("数据统计:")
        logging.info(f"总文档数: {len(data_list)}")
        
        # 统计各种数据类型
        type_counts = {}
        for doc in data_list:
            for key, value in doc.items():
                value_type = type(value).__name__
                if value_type not in type_counts:
                    type_counts[value_type] = 0
                type_counts[value_type] += 1
        
        logging.info(f"数据类型统计: {type_counts}")
    
    print(f"BSON数据已成功写入日志文件: {log_filename}")
    print(f"共处理 {document_count} 个文档")
    
except Exception as e:
    logging.error(f"处理BSON文件时出错: {str(e)}")
    print(f"处理BSON文件时出错: {str(e)}")