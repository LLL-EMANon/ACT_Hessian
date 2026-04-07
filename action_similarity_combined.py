"""
动作连续性评估 - 完整可视化版
用于评测动作序列的平滑性和连续性，并生成 PDF 分析报告图表
支持多种评估指标：余弦相似度、速度、加速度、综合评分
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Union
import argparse
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 核心分析类 - ActionContinuityAnalyzer
# ============================================================================

class ActionContinuityAnalyzer:
    """
    分析动作序列的连续性和平滑性，并支持可视化
    """
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.mean = None
        self.std = None
        
    def load_action_from_hdf5(self,
                              hdf5_path: str,
                              dataset_key: str = "actions/joint/position") -> np.ndarray:
        with h5py.File(hdf5_path, "r") as f:
            if dataset_key not in f:
                raise KeyError(f"HDF5中不存在数据路径: {dataset_key}")

            node = f[dataset_key]
            if not isinstance(node, h5py.Dataset):
                raise ValueError(f"数据路径不是Dataset: {dataset_key}")

            action = node[()]  # shape: (T, action_dim)

        if action.ndim != 2:
            raise ValueError(f"数据必须是2D数组(T, action_dim)，当前维度: {action.ndim}, 路径: {dataset_key}")
        return action
    
    def cosine_similarity(self, actions: np.ndarray) -> np.ndarray:
        assert actions.ndim == 2, "动作必须是2D数组(T, action_dim)"
        T = actions.shape[0]
        cos_sims = []
        
        for i in range(T - 1):
            a, b = actions[i], actions[i+1]
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            if norm_a < 1e-10 or norm_b < 1e-10:
                cos_sims.append(1.0)
            else:
                cos_sim = np.dot(a, b) / (norm_a * norm_b)
                cos_sims.append(cos_sim)
        
        return np.array(cos_sims)
    
    def velocity_magnitude(self, actions: np.ndarray) -> np.ndarray:
        assert actions.ndim == 2, "动作必须是2D数组(T, action_dim)"
        velocity = np.diff(actions, axis=0)
        vel_magnitude = np.linalg.norm(velocity, axis=1)
        return vel_magnitude
    
    def acceleration_magnitude(self, actions: np.ndarray) -> np.ndarray:
        assert actions.ndim == 2, "动作必须是2D数组(T, action_dim)"
        accel = np.diff(actions, n=2, axis=0)
        accel_magnitude = np.linalg.norm(accel, axis=1)
        return accel_magnitude
    
    def compute_smoothness_score(self, actions: np.ndarray, 
                                 weights: Dict[str, float] = None) -> Dict[str, Union[float, np.ndarray]]:
        if weights is None:
            weights = {'cos': 0.3, 'vel': 0.3, 'accel': 0.4}
        
        cos_sims = self.cosine_similarity(actions)
        vel_mags = self.velocity_magnitude(actions)
        accel_mags = self.acceleration_magnitude(actions)
        
        cos_sim_score = (cos_sims + 1) / 2
        
        vel_95 = np.percentile(vel_mags, 95) if len(vel_mags) > 0 else 1.0
        accel_95 = np.percentile(accel_mags, 95) if len(accel_mags) > 0 else 1.0
        
        vel_95 = max(vel_95, 1e-6)
        accel_95 = max(accel_95, 1e-6)
        
        vel_score = 1.0 - np.minimum(vel_mags / vel_95, 1.0)
        accel_score = 1.0 - np.minimum(accel_mags / accel_95, 1.0)
        
        min_len = min(len(vel_score), len(accel_score))
        
        combined_score = (
            weights['cos'] * np.mean(cos_sim_score[:min_len]) +
            weights['vel'] * np.mean(vel_score[:min_len]) +
            weights['accel'] * np.mean(accel_score[:min_len])
        )
        
        return {
            'cosine_similarity': cos_sims,
            'velocity_magnitude': vel_mags,
            'acceleration_magnitude': accel_mags,
            'cosine_sim_score': cos_sim_score,
            'velocity_score': vel_score,
            'acceleration_score': accel_score,
            'combined_smoothness_score': combined_score,
            'mean_cosine_similarity': np.mean(cos_sim_score),
            'mean_velocity_magnitude': np.mean(vel_mags),
            'mean_acceleration_magnitude': np.mean(accel_mags),
        }
    
    def analyze_hdf5_file(self,
                          hdf5_path: str,
                          weights: Dict[str, float] = None,
                          dataset_key: str = "actions/joint/position") -> Dict:
        actions = self.load_action_from_hdf5(hdf5_path, dataset_key=dataset_key)
        
        if self.normalize:
            self.mean = np.mean(actions, axis=0)
            self.std = np.std(actions, axis=0)
            self.std[self.std < 1e-6] = 1.0
            actions = (actions - self.mean) / self.std
            
        results = self.compute_smoothness_score(actions, weights)
        results['file_path'] = f"{hdf5_path}::{dataset_key}"
        results['num_timesteps'] = len(actions)
        results['action_dim'] = actions.shape[1]
        
        return results

    def analyze_action_array(self,
                             actions: np.ndarray,
                             weights: Dict[str, float] = None,
                             meta_name: str = "in_memory_segment") -> Dict:
        """直接分析动作数组，便于同文件分段对比场景复用。"""
        if self.normalize:
            mean = np.mean(actions, axis=0)
            std = np.std(actions, axis=0)
            std[std < 1e-6] = 1.0
            actions_for_score = (actions - mean) / std
        else:
            actions_for_score = actions

        results = self.compute_smoothness_score(actions_for_score, weights)
        results['file_path'] = meta_name
        results['num_timesteps'] = len(actions)
        results['action_dim'] = actions.shape[1]
        return results
    
    def analyze_dataset_dir(self, dataset_dir: str, num_episodes: int = None,
                           weights: Dict[str, float] = None) -> Dict:
        dataset_dir = Path(dataset_dir)
        episode_files = sorted(dataset_dir.glob("*.hdf5"))
        
        if num_episodes:
            episode_files = episode_files[:num_episodes]
            
        all_results = []
        episode_scores = []
        
        for hdf5_file in episode_files:
            try:
                result = self.analyze_hdf5_file(str(hdf5_file), weights)
                all_results.append(result)
                episode_scores.append(result['combined_smoothness_score'])
            except Exception as e:
                print(f"处理 {hdf5_file} 时出错: {e}")
                
        episode_scores = np.array(episode_scores)
        
        summary = {
            'total_episodes': len(all_results),
            'mean_smoothness_score': np.mean(episode_scores),
            'std_smoothness_score': np.std(episode_scores),
            'min_smoothness_score': np.min(episode_scores),
            'max_smoothness_score': np.max(episode_scores),
            'episode_results': all_results,
        }
        
        return summary

    def format_analysis_report(self, results: Dict, verbose: bool = True, max_steps: int = 10) -> str:
        lines = []
        if 'episode_results' in results:
            lines.append(f"\n{'='*60}")
            lines.append("数据集分析报告")
            lines.append(f"{'='*60}")
            lines.append(f"总Episode数: {results['total_episodes']}")
            lines.append(f"平均丝滑度评分: {results['mean_smoothness_score']:.4f}")
            lines.append(f"标准差: {results['std_smoothness_score']:.4f}")
            
            if verbose:
                lines.append("\n各Episode评分:")
                for i, ep_result in enumerate(results['episode_results']):
                    score = ep_result['combined_smoothness_score']
                    file_name = Path(ep_result['file_path']).name
                    lines.append(f"  {file_name}: {score:.4f}")
        else:
            lines.append(f"\n{'='*60}")
            lines.append("单个文件分析报告")
            lines.append(f"{'='*60}")
            if 'file_path' in results:
                lines.append(f"文件: {results.get('file_path', 'N/A')}")
            lines.append(f"时间步数: {results['num_timesteps']}")
            lines.append(f"动作维度: {results['action_dim']}")
            lines.append("\n平均指标:")
            lines.append(f"  余弦相似度: {results['mean_cosine_similarity']:.4f}")
            lines.append(f"  速度范数: {results['mean_velocity_magnitude']:.6f}")
            lines.append(f"  加速度范数: {results['mean_acceleration_magnitude']:.6f}")
            lines.append(f"\n综合丝滑度评分: {results['combined_smoothness_score']:.4f}")

        return "\n".join(lines)

    def format_compare_report(self, result_a: Dict, result_b: Dict,
                              label_a: str = "File A", label_b: str = "File B") -> str:
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append("双文件对比分析报告")
        lines.append(f"{'='*60}")
        lines.append(f"{label_a}: {result_a.get('file_path', 'N/A')}")
        lines.append(f"{label_b}: {result_b.get('file_path', 'N/A')}")

        score_a = result_a['combined_smoothness_score']
        score_b = result_b['combined_smoothness_score']
        diff = score_a - score_b

        lines.append("\n综合丝滑度评分:")
        lines.append(f"  {label_a}: {score_a:.4f}")
        lines.append(f"  {label_b}: {score_b:.4f}")
        lines.append(f"  评分差值({label_a} - {label_b}): {diff:+.4f}")

        better_label = label_a if diff > 0 else label_b if diff < 0 else "两者相同"
        if better_label == "两者相同":
            lines.append("  结论: 两个文件综合评分相同")
        else:
            lines.append(f"  结论: {better_label} 更平滑")

        lines.append("\n分项均值对比:")
        lines.append(f"  余弦相似度均值: {label_a}={result_a['mean_cosine_similarity']:.4f}, {label_b}={result_b['mean_cosine_similarity']:.4f}")
        lines.append(f"  速度范数均值:   {label_a}={result_a['mean_velocity_magnitude']:.6f}, {label_b}={result_b['mean_velocity_magnitude']:.6f}")
        lines.append(f"  加速度范数均值: {label_a}={result_a['mean_acceleration_magnitude']:.6f}, {label_b}={result_b['mean_acceleration_magnitude']:.6f}")

        return "\n".join(lines)

    # ---------------- 可视化部分 ----------------
    
    def plot_single_episode(self, actions: np.ndarray, results: Dict, save_dir: str = ".", title_prefix: str = "Episode"):
        """绘制单个 Episode 的时序多子图并保存为 PDF"""
        T = len(actions)
        time_steps = np.arange(T)
        
        plt.style.use('default')
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"{title_prefix} - Action Continuity Analysis\nCombined Score: {results['combined_smoothness_score']:.4f}", fontsize=14)

        axs[0].plot(time_steps, actions, alpha=0.7)
        axs[0].set_ylabel("Action Values")
        axs[0].set_title("Raw Action Trajectories (All Dimensions)")
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(time_steps[:-1], results['cosine_similarity'], color='green', linewidth=2)
        axs[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        axs[1].set_ylabel("Cos Sim")
        axs[1].set_title("Cosine Similarity (closer to 1.0 is smoother)")
        axs[1].set_ylim(-1.1, 1.1)
        axs[1].grid(True, alpha=0.3)

        axs[2].plot(time_steps[:-1], results['velocity_magnitude'], color='orange', linewidth=2)
        axs[2].set_ylabel("Velocity")
        axs[2].set_title("Velocity Magnitude (1st Derivative Norm)")
        axs[2].grid(True, alpha=0.3)

        axs[3].plot(time_steps[:-2], results['acceleration_magnitude'], color='red', linewidth=2)
        axs[3].set_ylabel("Acceleration")
        axs[3].set_title("Acceleration Magnitude (2nd Derivative Norm - Lower is smoother)")
        axs[3].set_xlabel("Time Step")
        axs[3].grid(True, alpha=0.3)

        plt.tight_layout()
        
        # 保存为 PDF
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{title_prefix}_analysis.pdf")
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"📊 时序分析图已保存至: {save_path}")
        plt.close()

    def plot_dataset_distribution(self, summary: Dict, save_dir: str = "."):
        """绘制整个数据集的丝滑度评分分布图并保存为 PDF"""
        scores = [res['combined_smoothness_score'] for res in summary['episode_results']]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(scores, bins=20, kde=True, color='skyblue')
        plt.axvline(np.mean(scores), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(scores):.4f}')
        
        plt.title("Dataset Smoothness Score Distribution", fontsize=14)
        plt.xlabel("Combined Smoothness Score (0 to 1, higher is better)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存为 PDF
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"dataset_distribution_{timestamp}.pdf")
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"📊 数据集分布图已保存至: {save_path}")
        plt.close()

    def plot_compare_two_files(self,
                               actions_a: np.ndarray,
                               result_a: Dict,
                               actions_b: np.ndarray,
                               result_b: Dict,
                               save_dir: str = ".",
                               label_a: str = "File A",
                               label_b: str = "File B"):
        """将两个文件的关键时序指标画在同一张图上进行对比，并保存为 PDF"""
        time_a = np.arange(len(actions_a))
        time_b = np.arange(len(actions_b))

        # 用动作向量范数替代逐维曲线，避免高维动作在同图叠加时可读性过差
        action_norm_a = np.linalg.norm(actions_a, axis=1)
        action_norm_b = np.linalg.norm(actions_b, axis=1)

        plt.style.use('default')
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
        fig.suptitle(
            f"Two-File Action Continuity Comparison\n"
            f"{label_a}: {result_a['combined_smoothness_score']:.4f} | "
            f"{label_b}: {result_b['combined_smoothness_score']:.4f}",
            fontsize=14
        )

        axs[0].plot(time_a, action_norm_a, linewidth=2, alpha=0.85, label=label_a)
        axs[0].plot(time_b, action_norm_b, linewidth=2, alpha=0.85, label=label_b)
        axs[0].set_ylabel("Action Norm")
        axs[0].set_title("Action Magnitude Over Time")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(np.arange(len(result_a['cosine_similarity'])), result_a['cosine_similarity'],
                    color='green', linewidth=2, alpha=0.85, label=label_a)
        axs[1].plot(np.arange(len(result_b['cosine_similarity'])), result_b['cosine_similarity'],
                    color='blue', linewidth=2, alpha=0.85, label=label_b)
        axs[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        axs[1].set_ylabel("Cos Sim")
        axs[1].set_title("Cosine Similarity Comparison")
        axs[1].set_ylim(-1.1, 1.1)
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        axs[2].plot(np.arange(len(result_a['velocity_magnitude'])), result_a['velocity_magnitude'],
                    color='orange', linewidth=2, alpha=0.85, label=label_a)
        axs[2].plot(np.arange(len(result_b['velocity_magnitude'])), result_b['velocity_magnitude'],
                    color='purple', linewidth=2, alpha=0.85, label=label_b)
        axs[2].set_ylabel("Velocity")
        axs[2].set_title("Velocity Magnitude Comparison")
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)

        axs[3].plot(np.arange(len(result_a['acceleration_magnitude'])), result_a['acceleration_magnitude'],
                    color='red', linewidth=2, alpha=0.85, label=label_a)
        axs[3].plot(np.arange(len(result_b['acceleration_magnitude'])), result_b['acceleration_magnitude'],
                    color='brown', linewidth=2, alpha=0.85, label=label_b)
        axs[3].set_ylabel("Acceleration")
        axs[3].set_title("Acceleration Magnitude Comparison")
        axs[3].set_xlabel("Time Step")
        axs[3].legend()
        axs[3].grid(True, alpha=0.3)

        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"two_file_comparison_{timestamp}.pdf")
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"📊 双文件对比图已保存至: {save_path}")
        plt.close()


def save_report_to_txt(report_text: str, output_path: str = ".") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(output_path, f"action_smoothness_report_{timestamp}.txt")
    out_file = Path(full_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report_text + "\n", encoding="utf-8")
    return str(out_file)


def extract_segment(actions: np.ndarray, start: int, end: int, segment_name: str = "segment") -> np.ndarray:
    """提取 [start, end) 区间动作并做合法性校验。"""
    if start < 0 or end < 0:
        raise ValueError(f"{segment_name} 的 start/end 不能为负数: start={start}, end={end}")
    if end <= start:
        raise ValueError(f"{segment_name} 的 end 必须大于 start: start={start}, end={end}")
    if end > len(actions):
        raise ValueError(f"{segment_name} 的 end 超出长度 {len(actions)}: end={end}")

    segment = actions[start:end]
    if len(segment) < 3:
        raise ValueError(f"{segment_name} 长度至少需要 3 个时间步，当前为 {len(segment)}")
    return segment

# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="动作连续性评估工具 (带 PDF 可视化)")
    parser.add_argument("--mode", type=str, required=True, choices=["file", "dataset", "compare", "compare_segments", "compare_keys"], help="运行模式：评估单个文件、整个数据集、对比两个文件、同文件分段对比或同文件不同键对比")
    parser.add_argument("--hdf5", type=str, help="HDF5文件路径 (用于 file / compare_segments / compare_keys 模式)")
    parser.add_argument("--hdf5_a", type=str, help="第一个HDF5文件路径 (用于 compare 模式)")
    parser.add_argument("--hdf5_b", type=str, help="第二个HDF5文件路径 (用于 compare 模式)")
    parser.add_argument("--key", type=str, default="actions/joint/position", help="数据键路径 (用于 file / compare_segments 模式)")
    parser.add_argument("--key_a", type=str, default="actions/joint/position", help="第一个数据键路径 (用于 compare / compare_keys 模式)")
    parser.add_argument("--key_b", type=str, default="actions/joint/position", help="第二个数据键路径 (用于 compare / compare_keys 模式)")
    parser.add_argument("--label_a", type=str, default="File A", help="第一个文件在图表中的显示名称")
    parser.add_argument("--label_b", type=str, default="File B", help="第二个文件在图表中的显示名称")
    parser.add_argument("--seg_a_start", type=int, help="分段A起始下标(包含)，用于 compare_segments")
    parser.add_argument("--seg_a_end", type=int, help="分段A结束下标(不包含)，用于 compare_segments")
    parser.add_argument("--seg_b_start", type=int, help="分段B起始下标(包含)，用于 compare_segments")
    parser.add_argument("--seg_b_end", type=int, help="分段B结束下标(不包含)，用于 compare_segments")
    parser.add_argument("--dataset", type=str, help="数据集目录路径 (用于 dataset 模式)")
    parser.add_argument("--num_episodes", type=int, default=None, help="分析的Episode数量，默认分析全部")
    parser.add_argument("--save_txt", action="store_true", help="是否将文本报告保存为txt")
    parser.add_argument("--no_plot", action="store_true", help="禁用自动画图")
    parser.add_argument("--save_dir", type=str, default=".", help="图表和报告保存的目录，默认为当前目录 (.)")
    
    args = parser.parse_args()
    analyzer = ActionContinuityAnalyzer(normalize=True)
    
    if args.mode == "file":
        if not args.hdf5:
            print("错误: file 模式必须提供 --hdf5 参数")
            exit(1)
            
        result = analyzer.analyze_hdf5_file(args.hdf5, dataset_key=args.key)
        report = analyzer.format_analysis_report(result, verbose=True)
        print(report)
        
        if args.save_txt:
            txt_path = save_report_to_txt(report, args.save_dir)
            print(f"📄 报告已保存至: {txt_path}")
            
        if not args.no_plot:
            actions = analyzer.load_action_from_hdf5(args.hdf5, dataset_key=args.key)
            # 作图前是否归一化展示，这里保持原始数据展示更直观
            file_name = Path(args.hdf5).stem
            analyzer.plot_single_episode(actions, result, save_dir=args.save_dir, title_prefix=file_name)
            
    elif args.mode == "dataset":
        if not args.dataset:
            print("错误: dataset 模式必须提供 --dataset 参数")
            exit(1)
            
        summary = analyzer.analyze_dataset_dir(args.dataset, num_episodes=args.num_episodes)
        report = analyzer.format_analysis_report(summary, verbose=True)
        print(report)
        
        if args.save_txt:
            txt_path = save_report_to_txt(report, args.save_dir)
            print(f"📄 报告已保存至: {txt_path}")
            
        if not args.no_plot:
            analyzer.plot_dataset_distribution(summary, save_dir=args.save_dir)

    elif args.mode == "compare":
        if not args.hdf5_a or not args.hdf5_b:
            print("错误: compare 模式必须同时提供 --hdf5_a 和 --hdf5_b 参数")
            exit(1)

        result_a = analyzer.analyze_hdf5_file(args.hdf5_a, dataset_key=args.key_a)
        result_b = analyzer.analyze_hdf5_file(args.hdf5_b, dataset_key=args.key_b)
        report = analyzer.format_compare_report(result_a, result_b, label_a=args.label_a, label_b=args.label_b)
        print(report)

        if args.save_txt:
            txt_path = save_report_to_txt(report, args.save_dir)
            print(f"📄 报告已保存至: {txt_path}")

        if not args.no_plot:
            actions_a = analyzer.load_action_from_hdf5(args.hdf5_a, dataset_key=args.key_a)
            actions_b = analyzer.load_action_from_hdf5(args.hdf5_b, dataset_key=args.key_b)
            analyzer.plot_compare_two_files(
                actions_a,
                result_a,
                actions_b,
                result_b,
                save_dir=args.save_dir,
                label_a=args.label_a,
                label_b=args.label_b,
            )

    elif args.mode == "compare_segments":
        required_args = [args.seg_a_start, args.seg_a_end, args.seg_b_start, args.seg_b_end]
        if not args.hdf5:
            print("错误: compare_segments 模式必须提供 --hdf5 参数")
            exit(1)
        if any(v is None for v in required_args):
            print("错误: compare_segments 模式必须提供 --seg_a_start --seg_a_end --seg_b_start --seg_b_end")
            exit(1)

        all_actions = analyzer.load_action_from_hdf5(args.hdf5, dataset_key=args.key)

        try:
            actions_a = extract_segment(all_actions, args.seg_a_start, args.seg_a_end, segment_name="Segment A")
            actions_b = extract_segment(all_actions, args.seg_b_start, args.seg_b_end, segment_name="Segment B")
        except ValueError as e:
            print(f"错误: {e}")
            exit(1)

        result_a = analyzer.analyze_action_array(
            actions_a,
            meta_name=f"{args.hdf5} [A:{args.seg_a_start}:{args.seg_a_end}]"
        )
        result_b = analyzer.analyze_action_array(
            actions_b,
            meta_name=f"{args.hdf5} [B:{args.seg_b_start}:{args.seg_b_end}]"
        )

        label_a = args.label_a if args.label_a != "File A" else f"SegmentA[{args.seg_a_start}:{args.seg_a_end}]"
        label_b = args.label_b if args.label_b != "File B" else f"SegmentB[{args.seg_b_start}:{args.seg_b_end}]"
        report = analyzer.format_compare_report(result_a, result_b, label_a=label_a, label_b=label_b)
        print(report)

        if args.save_txt:
            txt_path = save_report_to_txt(report, args.save_dir)
            print(f"📄 报告已保存至: {txt_path}")

        if not args.no_plot:
            analyzer.plot_compare_two_files(
                actions_a,
                result_a,
                actions_b,
                result_b,
                save_dir=args.save_dir,
                label_a=label_a,
                label_b=label_b,
            )

    elif args.mode == "compare_keys":
        if not args.hdf5:
            print("错误: compare_keys 模式必须提供 --hdf5 参数")
            exit(1)

        try:
            actions_a = analyzer.load_action_from_hdf5(args.hdf5, dataset_key=args.key_a)
            actions_b = analyzer.load_action_from_hdf5(args.hdf5, dataset_key=args.key_b)
        except (KeyError, ValueError) as e:
            print(f"错误: {e}")
            exit(1)

        result_a = analyzer.analyze_action_array(actions_a, meta_name=f"{args.hdf5}::{args.key_a}")
        result_b = analyzer.analyze_action_array(actions_b, meta_name=f"{args.hdf5}::{args.key_b}")

        label_a = args.label_a if args.label_a != "File A" else args.key_a
        label_b = args.label_b if args.label_b != "File B" else args.key_b
        report = analyzer.format_compare_report(result_a, result_b, label_a=label_a, label_b=label_b)
        print(report)

        if args.save_txt:
            txt_path = save_report_to_txt(report, args.save_dir)
            print(f"📄 报告已保存至: {txt_path}")

        if not args.no_plot:
            analyzer.plot_compare_two_files(
                actions_a,
                result_a,
                actions_b,
                result_b,
                save_dir=args.save_dir,
                label_a=label_a,
                label_b=label_b,
            )