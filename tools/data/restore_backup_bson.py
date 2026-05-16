#!/usr/bin/env python3
"""
BSON备份文件恢复脚本
将所有的 xhand_control_data_backup.bson 恢复为 xhand_control_data.bson
"""
import os
import shutil
import glob
from datetime import datetime

def find_backup_files(data_folder: str):
    """查找所有备份文件"""
    backup_files = []
    
    # 遍历data文件夹下的ours子文件夹
    ours_folder = os.path.join(data_folder, "ours")
    if not os.path.exists(ours_folder):
        print(f"警告: 未找到 ours 文件夹: {ours_folder}")
        return backup_files
    
    # 查找所有备份文件
    pattern = os.path.join(ours_folder, "action*", "episode_*", "xhand_control_data_backup.bson")
    backup_files = glob.glob(pattern)
    
    return sorted(backup_files)

def restore_backup_files(data_folder: str):
    """恢复所有备份文件"""
    print("=" * 80)
    print("BSON备份文件恢复工具")
    print("=" * 80)
    print(f"数据文件夹: {os.path.abspath(data_folder)}")
    
    # 查找所有备份文件
    backup_files = find_backup_files(data_folder)
    
    if not backup_files:
        print("未找到任何备份文件 (xhand_control_data_backup.bson)")
        return
    
    print(f"\n找到 {len(backup_files)} 个备份文件:")
    for backup_file in backup_files:
        rel_path = os.path.relpath(backup_file, data_folder)
        print(f"  - {rel_path}")
    
    # 确认恢复
    print("\n" + "=" * 80)
    response = input("确认恢复所有备份文件？这将覆盖当前的 xhand_control_data.bson 文件 (y/n): ")
    if response.lower() != 'y':
        print("已取消恢复")
        return
    
    # 执行恢复
    successful = 0
    failed = 0
    
    for backup_file in backup_files:
        try:
            # 构造目标文件路径
            target_file = backup_file.replace("_backup.bson", ".bson")
            
            # 复制备份文件到目标位置
            shutil.copy2(backup_file, target_file)
            
            # 验证文件大小
            backup_size = os.path.getsize(backup_file)
            target_size = os.path.getsize(target_file)
            
            if backup_size == target_size:
                successful += 1
                print(f"✓ 恢复成功: {os.path.relpath(target_file, data_folder)}")
            else:
                failed += 1
                print(f"✗ 恢复失败: {os.path.relpath(target_file, data_folder)} (文件大小不匹配)")
                
        except Exception as e:
            failed += 1
            print(f"✗ 恢复失败: {os.path.relpath(backup_file, data_folder)} - 错误: {e}")
    
    # 打印总结
    print(f"\n{'='*80}")
    print("恢复完成！")
    print(f"{'='*80}")
    print(f"恢复总结:")
    print(f"  总备份文件数: {len(backup_files)}")
    print(f"  成功恢复: {successful}")
    print(f"  恢复失败: {failed}")
    
    if successful > 0:
        print(f"\n✓ 已成功恢复 {successful} 个文件")
        print("现在所有的 xhand_control_data.bson 文件都已恢复为原始版本")
        
    # 保存恢复日志
    ours_folder = os.path.join(data_folder, "ours")
    log_file = os.path.join(ours_folder, f"restore_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("BSON备份文件恢复日志\n")
        f.write(f"恢复时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据文件夹: {os.path.abspath(data_folder)}\n")
        f.write(f"总备份文件数: {len(backup_files)}\n")
        f.write(f"成功恢复: {successful}\n")
        f.write(f"恢复失败: {failed}\n")
        f.write("\n恢复的文件列表:\n")
        for backup_file in backup_files:
            target_file = backup_file.replace("_backup.bson", ".bson")
            f.write(f"{os.path.relpath(target_file, data_folder)}\n")
    
    print(f"\n恢复日志已保存到: {log_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BSON备份文件恢复工具')
    parser.add_argument('data_folder', 
                        nargs='?', 
                        default='data',
                        help='数据文件夹路径 (默认: data)')
    
    args = parser.parse_args()
    
    # 检查数据文件夹是否存在
    if not os.path.exists(args.data_folder):
        print(f"错误: 数据文件夹不存在: {args.data_folder}")
        return
    
    # 执行恢复
    restore_backup_files(args.data_folder)

if __name__ == "__main__":
    main()