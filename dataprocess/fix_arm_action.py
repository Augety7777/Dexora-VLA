import os
import json
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from airbot_config import AirbotConfig
import datetime
import shutil


class ArmActionDimension6Fixer:
    """修复指定action数据中机械臂action第6维的问题"""
    
    def __init__(self, config: AirbotConfig, enable_backup: bool = True):
        self.config = config
        self.enable_backup = enable_backup
        
        # 🔧 先初始化基本属性
        self.dataset_name = f"{self.config.robot}_{self.config.task_name}"
        self.dataset_root = os.path.join(self.config.output_data_root, self.dataset_name)
        self.meta_dir = os.path.join(self.dataset_root, "meta")
        
        # 修改为指定的action列表
        self.actions_to_fix = [
            "action12", "action13", "action15", "action16", "action17", "action18",
            "action19", "action20", "action21", "action22", "action23", "action24"
        ]
        
        # 然后设置日志
        self.setup_logging()
        
    def setup_logging(self):
        """配置日志"""
        log_dir = os.path.join(self.config.log_root, self.config.task_name)
        os.makedirs(log_dir, exist_ok=True)
        
        backup_suffix = "_with_backup" if self.enable_backup else "_no_backup"
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 主日志文件
        log_file = os.path.join(log_dir, f"fix_arm_action_dim6_{timestamp}{backup_suffix}.log")
        
        # 预览专用日志文件
        self.preview_log_file = os.path.join(log_dir, f"fix_arm_action_preview_{timestamp}{backup_suffix}.log")

        # 配置主日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("ArmActionDimension6Fixer")
        
        # 配置预览专用日志记录器
        self.preview_logger = logging.getLogger("PreviewLogger")
        self.preview_logger.setLevel(logging.INFO)
        
        # 创建预览日志的文件处理器
        preview_handler = logging.FileHandler(self.preview_log_file, encoding='utf-8')
        preview_formatter = logging.Formatter("%(asctime)s - %(message)s")
        preview_handler.setFormatter(preview_formatter)
        self.preview_logger.addHandler(preview_handler)
        
        # 防止预览日志传播到根日志记录器
        self.preview_logger.propagate = False
        
        self.logger.info(f"备份模式: {'启用' if self.enable_backup else '禁用'}")
        self.logger.info(f"预览日志将保存到: {self.preview_log_file}")
        
        # 在预览日志文件中写入头信息
        self.preview_logger.info("="*80)
        self.preview_logger.info("机械臂Action第6维修复 - 预览日志")
        self.preview_logger.info(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.preview_logger.info(f"备份模式: {'启用' if self.enable_backup else '禁用'}")
        self.preview_logger.info(f"数据集: {self.dataset_name}")
        self.preview_logger.info(f"需要修复的actions: {self.actions_to_fix}")
        self.preview_logger.info("="*80)

    def get_episodes_to_fix(self):
        """获取需要修复的episode列表"""
        episodes_to_fix = []
        
        episodes_stats_file = os.path.join(self.meta_dir, "episodes_stats.jsonl")
        if not os.path.exists(episodes_stats_file):
            self.logger.error("未找到episodes_stats.jsonl文件")
            return episodes_to_fix
        
        with open(episodes_stats_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    stats = json.loads(line.strip())
                    action_id = stats.get('action_id', '')
                    episode_index = stats.get('episode_index')
                    
                    if action_id in self.actions_to_fix:
                        episodes_to_fix.append({
                            'episode_index': episode_index,
                            'action_id': action_id,
                            'chunk': episode_index // 1000  # 计算chunk
                        })
        
        self.logger.info(f"找到需要修复的episodes: {len(episodes_to_fix)} 个")
        return episodes_to_fix

    def backup_parquet_file(self, parquet_path: str) -> str:
        """备份parquet文件"""
        if not self.enable_backup:
            return None
            
        backup_path = parquet_path.replace('.parquet', '_backup.parquet')
        if not os.path.exists(backup_path):  # 避免重复备份
            shutil.copy2(parquet_path, backup_path)
            self.logger.info(f"已备份: {backup_path}")
        return backup_path

    def preview_episode_changes(self, episode_info: dict, detailed: bool = False):
        """预览单个episode的修改内容"""
        episode_index = episode_info['episode_index']
        chunk = episode_info['chunk']
        action_id = episode_info['action_id']
        
        # 构建parquet文件路径
        chunk_data_dir = os.path.join(self.dataset_root, "data", f"chunk-{chunk:03d}")
        parquet_path = os.path.join(chunk_data_dir, f"episode_{episode_index:06d}.parquet")
        
        if not os.path.exists(parquet_path):
            warning_msg = f"Parquet文件不存在: {parquet_path}"
            self.logger.warning(warning_msg)
            self.preview_logger.warning(warning_msg)
            return
        
        try:
            df = pd.read_parquet(parquet_path)
            
            # 同时记录到主日志和预览日志
            header_msg = f"\n=== 预览 {action_id} episode_{episode_index:06d} ==="
            frame_count_msg = f"总帧数: {len(df)}"
            
            self.logger.info(header_msg)
            self.logger.info(frame_count_msg)
            self.preview_logger.info(header_msg)
            self.preview_logger.info(frame_count_msg)
            
            if detailed:
                # 详细预览：显示所有帧的修改
                detail_header = "所有帧的修改预览:"
                table_header = "帧序号 | 左臂states[5] | 左臂actions[5] | 右臂states[23] | 右臂actions[23] | 是否需要修改"
                separator = "-" * 100
                
                self.logger.info(detail_header)
                self.logger.info(table_header)
                self.logger.info(separator)
                
                self.preview_logger.info(detail_header)
                self.preview_logger.info(table_header)
                self.preview_logger.info(separator)
                
                changes_needed = 0
                for idx, row in df.iterrows():
                    states = row['states']
                    actions = row['actions']
                    
                    if isinstance(states, np.ndarray) and isinstance(actions, np.ndarray):
                        if len(states) >= 36 and len(actions) >= 36:
                            left_arm_change = abs(states[5] - actions[5]) > 1e-6
                            right_arm_change = abs(states[23] - actions[23]) > 1e-6
                            need_change = left_arm_change or right_arm_change
                            
                            if need_change:
                                changes_needed += 1
                            
                            change_status = "是" if need_change else "否"
                            row_msg = f"{idx:6d} | {states[5]:12.6f} | {actions[5]:13.6f} | {states[23]:14.6f} | {actions[23]:15.6f} | {change_status}"
                            
                            self.logger.info(row_msg)
                            self.preview_logger.info(row_msg)
                
                summary_msg = f"\n需要修改的帧数: {changes_needed}/{len(df)}"
                self.logger.info(summary_msg)
                self.preview_logger.info(summary_msg)
            else:
                # 简要预览：只显示前5帧和统计信息
                sample_size = min(5, len(df))
                changes_needed = 0
                
                brief_header = f"前{sample_size}帧预览:"
                table_header = "帧序号 | 左臂states[5] | 左臂actions[5] | 右臂states[23] | 右臂actions[23] | 是否需要修改"
                separator = "-" * 100
                
                self.logger.info(brief_header)
                self.logger.info(table_header)
                self.logger.info(separator)
                
                self.preview_logger.info(brief_header)
                self.preview_logger.info(table_header)
                self.preview_logger.info(separator)
                
                for idx in range(sample_size):
                    states = df.iloc[idx]['states']
                    actions = df.iloc[idx]['actions']
                    
                    if isinstance(states, np.ndarray) and isinstance(actions, np.ndarray):
                        if len(states) >= 36 and len(actions) >= 36:
                            left_arm_change = abs(states[5] - actions[5]) > 1e-6
                            right_arm_change = abs(states[23] - actions[23]) > 1e-6
                            need_change = left_arm_change or right_arm_change
                            
                            change_status = "是" if need_change else "否"
                            row_msg = f"{idx:6d} | {states[5]:12.6f} | {actions[5]:13.6f} | {states[23]:14.6f} | {actions[23]:15.6f} | {change_status}"
                            
                            self.logger.info(row_msg)
                            self.preview_logger.info(row_msg)
                
                # 统计所有帧
                for idx, row in df.iterrows():
                    states = row['states']
                    actions = row['actions']
                    
                    if isinstance(states, np.ndarray) and isinstance(actions, np.ndarray):
                        if len(states) >= 36 and len(actions) >= 36:
                            left_arm_change = abs(states[5] - actions[5]) > 1e-6
                            right_arm_change = abs(states[23] - actions[23]) > 1e-6
                            need_change = left_arm_change or right_arm_change
                            
                            if need_change:
                                changes_needed += 1
                
                summary_msg = f"\n总计需要修改的帧数: {changes_needed}/{len(df)}"
                self.logger.info(summary_msg)
                self.preview_logger.info(summary_msg)
                
        except Exception as e:
            error_msg = f"预览episode {episode_index}失败: {e}"
            self.logger.error(error_msg)
            self.preview_logger.error(error_msg)

    def fix_episode_parquet(self, episode_info: dict) -> bool:
        """修复单个episode的parquet文件"""
        episode_index = episode_info['episode_index']
        chunk = episode_info['chunk']
        action_id = episode_info['action_id']
        
        # 构建parquet文件路径
        chunk_data_dir = os.path.join(self.dataset_root, "data", f"chunk-{chunk:03d}")
        parquet_path = os.path.join(chunk_data_dir, f"episode_{episode_index:06d}.parquet")
        
        if not os.path.exists(parquet_path):
            self.logger.warning(f"Parquet文件不存在: {parquet_path}")
            return False
        
        try:
            # 读取parquet文件
            df = pd.read_parquet(parquet_path)
            self.logger.info(f"处理 {action_id} episode_{episode_index:06d}: {len(df)} 帧")
            
            # 检查数据结构
            if 'states' not in df.columns or 'actions' not in df.columns:
                self.logger.error(f"数据结构异常: {parquet_path}")
                return False
            
            # 备份原文件（如果启用备份）
            if self.enable_backup:
                backup_path = self.backup_parquet_file(parquet_path)
            
            # 修复数据
            fixed_actions = []
            validation_errors = 0
            actual_changes = 0
            
            for idx, row in df.iterrows():
                states = row['states']
                actions = row['actions']
                
                # 确保数据类型正确
                if isinstance(states, np.ndarray) and isinstance(actions, np.ndarray):
                    # 检查维度
                    if len(states) >= 36 and len(actions) >= 36:
                        # 🔧 创建可写的actions副本
                        actions_copy = actions.copy()
                        
                        # 检查是否需要修改
                        left_arm_change = abs(states[5] - actions[5]) > 1e-6
                        right_arm_change = abs(states[23] - actions[23]) > 1e-6
                        
                        if left_arm_change or right_arm_change:
                            actual_changes += 1
                            
                        # 复制observation的第6维到action的第6维
                        # 左臂: observation[5] -> action[5] (第6维，索引为5)
                        # 右臂: observation[23] -> action[23] (右臂第6维，索引为23)
                        actions_copy[5] = states[5]    # 左臂第6维
                        actions_copy[23] = states[23]  # 右臂第6维
                        
                        fixed_actions.append(actions_copy)
                    else:
                        self.logger.error(f"维度异常 episode_{episode_index} frame_{idx}: states={len(states)}, actions={len(actions)}")
                        validation_errors += 1
                        fixed_actions.append(actions.copy())  # 保持原状，但要复制
                else:
                    self.logger.error(f"数据类型异常 episode_{episode_index} frame_{idx}: states={type(states)}, actions={type(actions)}")
                    validation_errors += 1
                    # 尝试转换为numpy数组
                    if hasattr(actions, 'copy'):
                        fixed_actions.append(actions.copy())
                    else:
                        fixed_actions.append(np.array(actions))
            
            # 更新DataFrame
            df['actions'] = fixed_actions
            
            # 保存修复后的文件
            df.to_parquet(parquet_path, index=False)
            
            self.logger.info(f"Episode {episode_index}: 实际修改了 {actual_changes} 帧")
            
            if validation_errors > 0:
                self.logger.warning(f"Episode {episode_index} 修复完成，但有 {validation_errors} 个帧存在异常")
            else:
                self.logger.info(f"✅ Episode {episode_index} 修复完成")
            
            # 验证修复结果
            self.validate_fix(parquet_path, episode_index)
            
            return True
            
        except Exception as e:
            self.logger.error(f"修复episode {episode_index}失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return False

    def validate_fix(self, parquet_path: str, episode_index: int):
        """验证修复结果"""
        try:
            df = pd.read_parquet(parquet_path)
            
            # 抽样检查几帧
            sample_size = min(5, len(df))
            if len(df) > 0:
                sample_indices = np.random.choice(len(df), sample_size, replace=False)
            else:
                self.logger.warning(f"Episode {episode_index} 没有数据帧")
                return
            
            all_correct = True
            for idx in sample_indices:
                states = df.iloc[idx]['states']
                actions = df.iloc[idx]['actions']
                
                # 检查左臂第6维是否已复制
                if abs(states[5] - actions[5]) > 1e-6:
                    self.logger.warning(f"Episode {episode_index} frame {idx}: 左臂第6维未正确复制 (states[5]={states[5]:.6f}, actions[5]={actions[5]:.6f})")
                    all_correct = False
                
                # 检查右臂第6维是否已复制
                if abs(states[23] - actions[23]) > 1e-6:
                    self.logger.warning(f"Episode {episode_index} frame {idx}: 右臂第6维未正确复制 (states[23]={states[23]:.6f}, actions[23]={actions[23]:.6f})")
                    all_correct = False
            
            if all_correct:
                self.logger.info(f"✅ Episode {episode_index} 验证通过")
            else:
                self.logger.warning(f"⚠️ Episode {episode_index} 验证发现问题")
                
        except Exception as e:
            self.logger.error(f"验证episode {episode_index}失败: {e}")

    def run_fix(self):
        """运行修复过程"""
        self.logger.info("=== 开始修复机械臂action第6维问题 ===")
        self.logger.info(f"需要修复的actions: {self.actions_to_fix}")
        self.logger.info(f"备份模式: {'启用' if self.enable_backup else '禁用'}")
        
        # 获取需要修复的episodes
        episodes_to_fix = self.get_episodes_to_fix()
        
        if not episodes_to_fix:
            self.logger.info("没有找到需要修复的episodes")
            return
        
        # 按action分组统计
        action_stats = {}
        for episode in episodes_to_fix:
            action_id = episode['action_id']
            if action_id not in action_stats:
                action_stats[action_id] = 0
            action_stats[action_id] += 1
        
        self.logger.info("需要修复的episodes分布:")
        for action_id, count in sorted(action_stats.items()):
            self.logger.info(f"  {action_id}: {count} episodes")
        
        # 执行修复
        success_count = 0
        failed_count = 0
        
        for episode_info in tqdm(episodes_to_fix, desc="修复episodes"):
            if self.fix_episode_parquet(episode_info):
                success_count += 1
            else:
                failed_count += 1
        
        # 总结
        self.logger.info("=== 修复完成 ===")
        self.logger.info(f"成功修复: {success_count} episodes")
        self.logger.info(f"修复失败: {failed_count} episodes")
        self.logger.info(f"总计处理: {len(episodes_to_fix)} episodes")
        
        if failed_count > 0:
            self.logger.warning("存在修复失败的episodes，请检查日志")
        else:
            self.logger.info("✅ 所有episodes修复成功")

    def preview_all_changes(self, detailed: bool = False):
        """预览所有episode的修改内容"""
        self.logger.info("=== 预览所有修改 ===")
        
        episodes_to_fix = self.get_episodes_to_fix()
        if not episodes_to_fix:
            self.logger.info("没有找到需要修复的episodes")
            return
        
        preview_msg = f"将预览 {len(episodes_to_fix)} 个episodes的修改内容"
        mode_msg = "详细模式：显示每个episode的所有帧" if detailed else "简要模式：显示每个episode的前5帧和统计信息"
        
        self.logger.info(preview_msg)
        self.logger.info(mode_msg)
        
        # 在预览日志中记录统计信息
        self.preview_logger.info(f"\n预览开始 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.preview_logger.info(preview_msg)
        self.preview_logger.info(mode_msg)
        
        # 按action分组统计
        action_stats = {}
        for episode in episodes_to_fix:
            action_id = episode['action_id']
            if action_id not in action_stats:
                action_stats[action_id] = 0
            action_stats[action_id] += 1
        
        self.preview_logger.info("\n需要修复的episodes分布:")
        for action_id, count in sorted(action_stats.items()):
            stats_msg = f"  {action_id}: {count} episodes"
            self.preview_logger.info(stats_msg)
        
        self.preview_logger.info("\n" + "="*80)
        
        # 预览所有episodes
        for i, episode_info in enumerate(episodes_to_fix):
            progress_msg = f"\n进度: {i+1}/{len(episodes_to_fix)}"
            self.logger.info(progress_msg)
            self.preview_logger.info(progress_msg)
            self.preview_episode_changes(episode_info, detailed=detailed)
        
        # 预览结束
        end_msg = f"\n预览完成 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.preview_logger.info(end_msg)
        self.preview_logger.info("="*80)
        
        self.logger.info(f"预览日志已保存到: {self.preview_log_file}")


def main():
    """主函数"""
    config = AirbotConfig()
    
    print("请选择操作模式:")
    print("1. 启用备份模式（安全，但占用更多磁盘空间）")
    print("2. 禁用备份模式（节省空间，但无法回滚）")
    
    backup_choice = input("请输入选择 (1/2): ")
    enable_backup = backup_choice == "1"
    
    print("\n请选择预览详细程度:")
    print("1. 简要预览（显示前5帧和统计信息）")
    print("2. 详细预览（显示所有帧的修改内容）")
    
    detail_choice = input("请输入选择 (1/2): ")
    detailed_preview = detail_choice == "2"
    
    fixer = ArmActionDimension6Fixer(config, enable_backup=enable_backup)
    
    # 预览修改
    fixer.preview_all_changes(detailed=detailed_preview)
    
    # 确认是否继续
    response = input("\n是否继续执行修复? (y/N): ")
    if response.lower() in ['y', 'yes']:
        fixer.run_fix()
    else:
        print("已取消修复操作")


if __name__ == "__main__":
    main()