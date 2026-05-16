#!/usr/bin/env python3
"""
批量BSON文件时间戳同步脚本
遍历data/ours文件夹中的所有action/episode子文件夹，同步其中的BSON文件
"""
import bson
import os
import numpy as np
from typing import List, Dict, Any, Tuple
import bisect
import glob
import shutil
from datetime import datetime
import sys
import logging
from io import StringIO

class LogCapture:
    """捕获print输出并同时输出到控制台和日志"""
    def __init__(self, logger):
        self.logger = logger
        self.terminal = sys.stdout
        self.log_buffer = StringIO()
        
    def write(self, message):
        # 写入控制台
        self.terminal.write(message)
        # 写入日志
        if message.strip():  # 忽略空行
            self.logger.info(message.strip())
        
    def flush(self):
        self.terminal.flush()

class BSONSynchronizer:
    def __init__(self, logger=None):
        self.episode_data = None
        self.xhand_data = None
        self.episode_timestamps = []
        self.xhand_timestamps = []
        self.logger = logger
        
    def log_print(self, message):
        """同时打印到控制台和日志"""
        print(message)
        if self.logger:
            self.logger.info(message)
        
    def load_episode_data(self, filename: str) -> bool:
        """加载episode_0.bson文件"""
        self.log_print(f"加载episode数据: {filename}")
        
        try:
            with open(filename, 'rb') as f:
                data = f.read()
                self.episode_data = bson.decode(data)
            
            # 提取时间戳
            if 'data' in self.episode_data:
                # 使用任意一个数据流的时间戳作为基准（它们应该是同步的）
                first_key = list(self.episode_data['data'].keys())[0]
                timestamp_data = self.episode_data['data'][first_key]
                
                self.episode_timestamps = [item['t'] for item in timestamp_data]
                
                self.log_print(f"Episode数据统计:")
                self.log_print(f"  总帧数: {len(self.episode_timestamps)}")
                self.log_print(f"  开始时间戳: {self.episode_timestamps[0]}")
                self.log_print(f"  结束时间戳: {self.episode_timestamps[-1]}")
                self.log_print(f"  持续时间: {(self.episode_timestamps[-1] - self.episode_timestamps[0])/1000:.3f} 秒")
                
                return True
                
        except Exception as e:
            self.log_print(f"加载episode数据失败: {e}")
            return False
    
    def load_xhand_data(self, filename: str) -> bool:
        """加载xhand_control_data.bson文件"""
        self.log_print(f"\n加载xhand数据: {filename}")
        
        try:
            with open(filename, 'rb') as f:
                data = f.read()
                self.xhand_data = bson.decode(data)
            
            # 提取时间戳
            if 'frames' in self.xhand_data:
                frames = self.xhand_data['frames']
                self.xhand_timestamps = [frame['t'] for frame in frames]
                
                self.log_print(f"Xhand数据统计:")
                self.log_print(f"  总帧数: {len(self.xhand_timestamps)}")
                self.log_print(f"  开始时间戳: {self.xhand_timestamps[0]:.6f}")
                self.log_print(f"  结束时间戳: {self.xhand_timestamps[-1]:.6f}")
                self.log_print(f"  持续时间: {self.xhand_timestamps[-1] - self.xhand_timestamps[0]:.3f} 秒")
                
                # 分析帧率
                if len(self.xhand_timestamps) > 1:
                    time_diffs = [self.xhand_timestamps[i+1] - self.xhand_timestamps[i] 
                                  for i in range(len(self.xhand_timestamps)-1)]
                    avg_diff = sum(time_diffs) / len(time_diffs)
                    fps = 1.0 / avg_diff if avg_diff > 0 else 0
                    self.log_print(f"  平均帧率: {fps:.1f} FPS")
                    self.log_print(f"  平均帧间隔: {avg_diff*1000:.1f} ms")
                
                return True
                
        except Exception as e:
            self.log_print(f"加载xhand数据失败: {e}")
            return False
    
    def normalize_timestamps(self) -> Tuple[List[float], List[float]]:
        """将两个时间戳序列标准化到相同的时间范围"""
        self.log_print(f"\n标准化时间戳...")
        
        # 将episode时间戳转换为相对时间（毫秒转秒）
        episode_start = self.episode_timestamps[0]
        episode_relative = [(t - episode_start) / 1000.0 for t in self.episode_timestamps]
        
        # xhand时间戳已经是相对时间（秒）
        xhand_start = self.xhand_timestamps[0]
        xhand_relative = [t - xhand_start for t in self.xhand_timestamps]
        
        self.log_print(f"Episode标准化后:")
        self.log_print(f"  开始时间: {episode_relative[0]:.6f} 秒")
        self.log_print(f"  结束时间: {episode_relative[-1]:.6f} 秒")
        self.log_print(f"  持续时间: {episode_relative[-1] - episode_relative[0]:.3f} 秒")
        
        # 分析Episode帧率
        if len(episode_relative) > 1:
            time_diffs = [episode_relative[i+1] - episode_relative[i] 
                          for i in range(len(episode_relative)-1)]
            avg_diff = sum(time_diffs) / len(time_diffs)
            fps = 1.0 / avg_diff if avg_diff > 0 else 0
            self.log_print(f"  平均帧率: {fps:.1f} FPS")
            self.log_print(f"  平均帧间隔: {avg_diff*1000:.1f} ms")
        
        self.log_print(f"Xhand标准化后:")
        self.log_print(f"  开始时间: {xhand_relative[0]:.6f} 秒")
        self.log_print(f"  结束时间: {xhand_relative[-1]:.6f} 秒")
        self.log_print(f"  持续时间: {xhand_relative[-1] - xhand_relative[0]:.3f} 秒")
        
        return episode_relative, xhand_relative
    
    def synchronize_data(self) -> Dict[str, Any]:
        """同步xhand数据到episode时间戳"""
        self.log_print(f"\n开始时间戳同步...")
        
        episode_relative, xhand_relative = self.normalize_timestamps()
        
        # 分析时间范围覆盖情况
        episode_beyond_xhand = sum(1 for t in episode_relative if t > xhand_relative[-1])
        episode_before_xhand = sum(1 for t in episode_relative if t < xhand_relative[0])
        
        self.log_print(f"\n时间范围分析:")
        self.log_print(f"  Episode超出xhand结束时间的帧数: {episode_beyond_xhand}")
        self.log_print(f"  Episode早于xhand开始时间的帧数: {episode_before_xhand}")
        
        # 计算帧率比
        if len(episode_relative) > 1 and len(xhand_relative) > 1:
            episode_duration = episode_relative[-1] - episode_relative[0]
            xhand_duration = xhand_relative[-1] - xhand_relative[0]
            episode_fps = len(episode_relative) / episode_duration if episode_duration > 0 else 0
            xhand_fps = len(xhand_relative) / xhand_duration if xhand_duration > 0 else 0
            
            self.log_print(f"\n帧率分析:")
            self.log_print(f"  Episode有效帧率: {episode_fps:.1f} FPS")
            self.log_print(f"  Xhand有效帧率: {xhand_fps:.1f} FPS")
            if xhand_fps > 0:
                fps_ratio = episode_fps / xhand_fps
                self.log_print(f"  帧率比(Episode/Xhand): {fps_ratio:.2f}")
                self.log_print(f"  理论上每{fps_ratio:.2f}个Episode帧对应1个Xhand帧")
        
        # 为每个episode时间戳找到最接近的xhand帧
        synchronized_frames = []
        boundary_usage = {'first_frame': 0, 'last_frame': 0}
        
        # 详细的匹配记录（只记录部分用于日志）
        detailed_matches = []
        
        for i, target_time in enumerate(episode_relative):
            # 在xhand时间戳中找到最接近的时间点
            pos = bisect.bisect_left(xhand_relative, target_time)
            
            # 找到最接近的索引
            best_idx = 0
            min_diff = float('inf')
            
            # 检查左右两个位置
            candidates = []
            if pos > 0:
                candidates.append(pos - 1)
            if pos < len(xhand_relative):
                candidates.append(pos)
            
            # 如果没有候选者或超出范围，使用边界值
            if not candidates or target_time < xhand_relative[0]:
                best_idx = 0
                boundary_usage['first_frame'] += 1
            elif target_time > xhand_relative[-1]:
                best_idx = len(xhand_relative) - 1
                boundary_usage['last_frame'] += 1
            else:
                # 选择时间差最小的候选者
                for idx in candidates:
                    diff = abs(xhand_relative[idx] - target_time)
                    if diff < min_diff:
                        min_diff = diff
                        best_idx = idx
            
            # 添加对应的帧
            synchronized_frames.append(self.xhand_data['frames'][best_idx])
            
            # 记录详细匹配信息（每50帧或边界情况）
            if i < 10 or i % 50 == 0 or target_time > xhand_relative[-1] or target_time < xhand_relative[0]:
                time_diff = abs(xhand_relative[best_idx] - target_time)
                boundary_marker = ""
                if target_time > xhand_relative[-1]:
                    boundary_marker = " [超出xhand范围]"
                elif target_time < xhand_relative[0]:
                    boundary_marker = " [早于xhand开始]"
                
                match_info = f"Episode帧 {i} (时间: {target_time:.6f}) -> Xhand帧 {best_idx} (时间: {xhand_relative[best_idx]:.6f}), 时间差: {time_diff*1000:.1f}ms{boundary_marker}"
                self.log_print(match_info)
        
        self.log_print(f"\n边界帧使用统计:")
        self.log_print(f"  重复使用第一帧次数: {boundary_usage['first_frame']}")
        self.log_print(f"  重复使用最后一帧次数: {boundary_usage['last_frame']}")
        
        # 详细的使用统计
        used_indices = {}
        for i, target_time in enumerate(episode_relative):
            pos = bisect.bisect_left(xhand_relative, target_time)
            best_idx = 0
            min_diff = float('inf')
            
            candidates = []
            if pos > 0:
                candidates.append(pos - 1)
            if pos < len(xhand_relative):
                candidates.append(pos)
            
            if not candidates or target_time < xhand_relative[0]:
                best_idx = 0
            elif target_time > xhand_relative[-1]:
                best_idx = len(xhand_relative) - 1
            else:
                for idx in candidates:
                    diff = abs(xhand_relative[idx] - target_time)
                    if diff < min_diff:
                        min_diff = diff
                        best_idx = idx
            
            used_indices[best_idx] = used_indices.get(best_idx, 0) + 1
        
        self.log_print(f"\n同步完成:")
        self.log_print(f"  Episode总帧数: {len(episode_relative)}")
        self.log_print(f"  Xhand原始帧数: {len(xhand_relative)}")
        self.log_print(f"  同步后帧数: {len(synchronized_frames)}")
        
        unique_frames_used = len(used_indices)
        total_reuses = sum(count - 1 for count in used_indices.values() if count > 1)
        unused_frames = len(xhand_relative) - unique_frames_used
        
        self.log_print(f"  使用的唯一xhand帧数: {unique_frames_used}")
        self.log_print(f"  未使用的xhand帧数: {unused_frames}")
        self.log_print(f"  重复使用的总次数: {total_reuses}")
        
        # 显示重复使用最多的帧
        max_reuse_count = max(used_indices.values())
        max_reuse_frames = [idx for idx, count in used_indices.items() if count == max_reuse_count]
        if max_reuse_count > 1:
            self.log_print(f"  最大重复使用次数: {max_reuse_count} (帧索引: {max_reuse_frames[:5]}...)")  # 只显示前5个
        
        # 分析重复使用模式
        reuse_distribution = {}
        for count in used_indices.values():
            reuse_distribution[count] = reuse_distribution.get(count, 0) + 1
        
        self.log_print(f"\n重复使用分布:")
        for reuse_count in sorted(reuse_distribution.keys()):
            self.log_print(f"  使用{reuse_count}次的帧数: {reuse_distribution[reuse_count]}")
        
        # 创建新的xhand数据结构
        synchronized_data = {
            'frames': synchronized_frames
        }
        
        return synchronized_data
        
    def save_synchronized_data(self, data: Dict[str, Any], output_filename: str):
        """保存同步后的数据"""
        self.log_print(f"\n保存同步后的数据到: {output_filename}")
        
        try:
            with open(output_filename, 'wb') as f:
                encoded_data = bson.encode(data)
                f.write(encoded_data)
            
            self.log_print(f"保存成功，文件大小: {os.path.getsize(output_filename)} 字节")
            
        except Exception as e:
            self.log_print(f"保存失败: {e}")
    
    def verify_synchronization(self, original_filename: str, synchronized_filename: str):
        """验证同步结果"""
        self.log_print(f"\n验证同步结果...")
        
        try:
            # 加载同步后的数据
            with open(synchronized_filename, 'rb') as f:
                data = f.read()
                sync_data = bson.decode(data)
            
            sync_frame_count = len(sync_data['frames'])
            episode_frame_count = len(self.episode_timestamps)
            
            self.log_print(f"验证结果:")
            self.log_print(f"  Episode帧数: {episode_frame_count}")
            self.log_print(f"  同步后xhand帧数: {sync_frame_count}")
            self.log_print(f"  帧数是否一致: {'✓' if sync_frame_count == episode_frame_count else '✗'}")
            
            if sync_frame_count == episode_frame_count:
                self.log_print(f"✓ 同步成功！两个文件的时间戳数量已一致")
                return True
            else:
                self.log_print(f"✗ 同步后帧数不匹配，相差: {abs(sync_frame_count - episode_frame_count)} 帧")
                return False
                
        except Exception as e:
            self.log_print(f"验证失败: {e}")
            return False


def find_episode_folders(data_folder: str) -> List[str]:
    """查找所有包含episode_0.bson和xhand_control_data.bson的文件夹"""
    episode_folders = []
    
    # 遍历data文件夹下的ours子文件夹
    ours_folder = os.path.join(data_folder, "ours")
    if not os.path.exists(ours_folder):
        print(f"警告: 未找到 ours 文件夹: {ours_folder}")
        return episode_folders
    
    # 遍历ours文件夹下的所有action文件夹
    for action_folder in glob.glob(os.path.join(ours_folder, "action*")):
        if os.path.isdir(action_folder):
            # 查找所有episode文件夹 (直接在action文件夹下)
            for episode_folder in glob.glob(os.path.join(action_folder, "episode_*")):
                if os.path.isdir(episode_folder):
                    # 检查是否包含所需的BSON文件
                    episode_file = os.path.join(episode_folder, "episode_0.bson")
                    xhand_file = os.path.join(episode_folder, "xhand_control_data.bson")
                    
                    if os.path.exists(episode_file) and os.path.exists(xhand_file):
                        episode_folders.append(episode_folder)
                    elif os.path.exists(episode_file):
                        print(f"警告: {episode_folder} 中缺少 xhand_control_data.bson")
                    elif os.path.exists(xhand_file):
                        print(f"警告: {episode_folder} 中缺少 episode_0.bson")
    
    return sorted(episode_folders)


def setup_logging(log_dir: str):
    """设置日志系统"""
    # 将日志保存到 data/ours 文件夹中
    ours_log_dir = os.path.join(log_dir, "ours")
    
    # 确保ours文件夹存在
    if not os.path.exists(ours_log_dir):
        os.makedirs(ours_log_dir, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(ours_log_dir, f"bson_sync_detailed_{timestamp}.log")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
        ]
    )
    
    return logging.getLogger(__name__), log_file


def batch_process_bson_files(data_folder: str, backup: bool = True):
    """批量处理所有BSON文件"""
    # 设置日志
    logger, log_file = setup_logging(data_folder)
    
    print("=" * 80)
    print("批量BSON文件时间戳同步工具")
    print("=" * 80)
    print(f"数据文件夹: {os.path.abspath(data_folder)}")
    print(f"备份原文件: {'是' if backup else '否'}")
    print(f"详细日志文件: {log_file}")
    print("=" * 80)
    
    logger.info("=" * 80)
    logger.info("批量BSON文件时间戳同步工具")
    logger.info("=" * 80)
    logger.info(f"数据文件夹: {os.path.abspath(data_folder)}")
    logger.info(f"备份原文件: {'是' if backup else '否'}")
    
    # 查找所有需要处理的文件夹
    episode_folders = find_episode_folders(data_folder)
    
    if not episode_folders:
        print("错误: 未找到包含 episode_0.bson 和 xhand_control_data.bson 的文件夹")
        logger.error("错误: 未找到包含 episode_0.bson 和 xhand_control_data.bson 的文件夹")
        return
    
    print(f"\n找到 {len(episode_folders)} 个待处理的episode文件夹:")
    logger.info(f"找到 {len(episode_folders)} 个待处理的episode文件夹:")
    for folder in episode_folders:
        rel_path = os.path.relpath(folder, data_folder)
        print(f"  - {rel_path}")
        logger.info(f"  - {rel_path}")
    
    # 确认处理
    print("\n" + "=" * 80)
    response = input("确认开始处理？(y/n): ")
    if response.lower() != 'y':
        print("已取消处理")
        logger.info("用户取消处理")
        return
    
    # 记录处理结果
    successful = 0
    failed = 0
    results = []
    
    # 处理每个文件夹
    for i, episode_folder in enumerate(episode_folders, 1):
        # 检查bson文件数量，如果有3个则跳过
        bson_files = [f for f in os.listdir(episode_folder) if f.endswith('.bson')]
        if len(bson_files) >= 3:
            print(f"\n{'='*80}")
            print(f"跳过已对齐的文件夹: {os.path.relpath(episode_folder, data_folder)} (已存在3个bson文件)")
            logger.info(f"跳过已对齐的文件夹: {os.path.relpath(episode_folder, data_folder)} (已存在3个bson文件)")
            results.append((episode_folder, "已对齐，跳过"))
            continue

        print(f"\n{'='*80}")
        print(f"处理进度: {i}/{len(episode_folders)}")
        print(f"当前文件夹: {os.path.relpath(episode_folder, data_folder)}")
        print(f"{'='*80}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"处理进度: {i}/{len(episode_folders)}")
        logger.info(f"当前文件夹: {os.path.relpath(episode_folder, data_folder)}")
        logger.info(f"{'='*80}")
        
        episode_file = os.path.join(episode_folder, "episode_0.bson")
        xhand_file = os.path.join(episode_folder, "xhand_control_data.bson")
        
        try:
            # 备份原始xhand文件
            if backup:
                backup_file = os.path.join(episode_folder, "xhand_control_data_backup.bson")
                if not os.path.exists(backup_file):  # 只在第一次备份
                    shutil.copy2(xhand_file, backup_file)
                    print(f"已备份原始文件到: {os.path.basename(backup_file)}")
                    logger.info(f"已备份原始文件到: {backup_file}")
            
            # 创建同步器
            synchronizer = BSONSynchronizer(logger)
            
            # 加载数据
            if not synchronizer.load_episode_data(episode_file):
                raise Exception("加载episode数据失败")
            
            if not synchronizer.load_xhand_data(xhand_file):
                raise Exception("加载xhand数据失败")
            
            # 执行同步
            synchronized_data = synchronizer.synchronize_data()
            
            # 保存同步后的数据（覆盖原文件）
            synchronizer.save_synchronized_data(synchronized_data, xhand_file)
            
            # 验证结果
            if synchronizer.verify_synchronization(xhand_file, xhand_file):
                successful += 1
                results.append((episode_folder, "成功"))
                print(f"✓ 处理成功")
                logger.info(f"✓ 处理成功")
            else:
                failed += 1
                results.append((episode_folder, "验证失败"))
                print(f"✗ 处理失败：验证未通过")
                logger.error(f"✗ 处理失败：验证未通过")
            
        except Exception as e:
            failed += 1
            results.append((episode_folder, f"错误: {str(e)}"))
            print(f"✗ 处理失败: {e}")
            logger.error(f"✗ 处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 打印总结
    print(f"\n{'='*80}")
    print("批量处理完成！")
    print(f"{'='*80}")
    print(f"处理总结:")
    print(f"  总文件夹数: {len(episode_folders)}")
    print(f"  成功: {successful}")
    print(f"  失败: {failed}")
    
    logger.info(f"\n{'='*80}")
    logger.info("批量处理完成！")
    logger.info(f"{'='*80}")
    logger.info(f"处理总结:")
    logger.info(f"  总文件夹数: {len(episode_folders)}")
    logger.info(f"  成功: {successful}")
    logger.info(f"  失败: {failed}")
    
    if failed > 0:
        print(f"\n失败详情:")
        logger.info(f"\n失败详情:")
        for folder, status in results:
            if not status.startswith("成功"):
                rel_path = os.path.relpath(folder, data_folder)
                print(f"  - {rel_path}: {status}")
                logger.info(f"  - {rel_path}: {status}")
    
    # 保存简要处理日志
    ours_folder = os.path.join(data_folder, "ours")
    summary_log_file = os.path.join(ours_folder, f"sync_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_log_file, 'w', encoding='utf-8') as f:
        f.write("BSON批量同步处理日志（摘要）\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据文件夹: {os.path.abspath(data_folder)}\n")
        f.write(f"总文件夹数: {len(episode_folders)}\n")
        f.write(f"成功: {successful}\n")
        f.write(f"失败: {failed}\n")
        f.write("\n详细结果:\n")
        for folder, status in results:
            f.write(f"{os.path.relpath(folder, data_folder)}: {status}\n")
    
    print(f"\n摘要日志已保存到: {summary_log_file}")
    print(f"详细日志已保存到: {log_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量BSON文件时间戳同步工具')
    parser.add_argument('data_folder', 
                        nargs='?', 
                        default='data',
                        help='数据文件夹路径 (默认: data)')
    parser.add_argument('--no-backup', 
                        action='store_true',
                        help='不备份原始xhand文件')
    
    args = parser.parse_args()
    
    # 检查数据文件夹是否存在
    if not os.path.exists(args.data_folder):
        print(f"错误: 数据文件夹不存在: {args.data_folder}")
        sys.exit(1)
    
    # 执行批量处理
    batch_process_bson_files(args.data_folder, backup=not args.no_backup)


if __name__ == "__main__":
    main()