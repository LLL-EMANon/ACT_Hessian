"""
Hessian Eigenvalue Spectrum 对比可视化工具

用法:
    python compare_hessian.py <npz_file_1> <npz_file_2> [--label1 MODEL_A] [--label2 MODEL_B] [--outdir ./compare_output]

示例:
    python compare_hessian.py \
        /path/to/model_A/hessian_eigenvalues_seed_0.npz \
        /path/to/model_B/hessian_eigenvalues_seed_0.npz \
        --label1 "ACT (kl_weight=10)" \
        --label2 "ACT (kl_weight=100)" \
        --outdir ./hessian_compare

输出:
    1. hessian_spectrum_compare.png/pdf        — 特征值谱柱状图 (线性刻度)
    2. hessian_logscale_compare.png/pdf        — |特征值| 对数刻度折线图
    3. hessian_distribution_compare.png/pdf     — 特征值分布直方图
    4. hessian_compare_summary.txt             — 数值对比摘要
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['mathtext.fontset'] = 'stix'
except Exception:
    pass

# ===================== 配色方案 =====================
# Model A: 蓝色系 (与原代码保持一致)
COLOR_A = '#2B8CBE'
COLOR_A_FILL = '#2B8CBE'
COLOR_A_LIGHT = '#A6BDDB'

# Model B: 红/橙色系 (对比鲜明)
COLOR_B = '#E34A33'
COLOR_B_FILL = '#E34A33'
COLOR_B_LIGHT = '#FDBB84'

MARKER_A = 'o'
MARKER_B = 's'


def load_npz(path):
    """加载 npz 文件并返回关键数据的字典"""
    data = np.load(path)
    result = {
        'eigenvalues': data['eigenvalues'],        # 降序排列的特征值
        'T_matrix': data['T_matrix'],               # Lanczos 三对角矩阵
        'alphas': data['alphas'],                    # 三对角矩阵对角线元素
        'betas': data['betas'],                      # 三对角矩阵次对角线元素
        'condition_number': float(data['condition_number'][0]),
        'lambda_max': float(data['lambda_max'][0]),
        'lambda_min': float(data['lambda_min'][0]),
    }
    return result


def plot_spectrum_compare(data_a, data_b, label_a, label_b, outdir):
    """
    绘图 1: 特征值谱柱状图对比 (线性刻度)
    两个模型的柱状图并排放置
    """
    eigs_a = data_a['eigenvalues']
    eigs_b = data_b['eigenvalues']

    n_a = len(eigs_a)
    n_b = len(eigs_b)
    n_max = max(n_a, n_b)

    indices = np.arange(n_max)
    bar_width = 0.38

    plt.figure(figsize=(9, 6))

    # Model A 柱状图
    plt.bar(indices[:n_a] - bar_width / 2, eigs_a, width=bar_width,
            color=COLOR_A_FILL, edgecolor='w', linewidth=0.3, alpha=0.85,
            label=label_a)

    # Model B 柱状图
    plt.bar(indices[:n_b] + bar_width / 2, eigs_b, width=bar_width,
            color=COLOR_B_FILL, edgecolor='w', linewidth=0.3, alpha=0.85,
            label=label_b)

    plt.axhline(y=0, color='#333333', linewidth=0.8, linestyle='--')
    plt.title('Hessian Eigenvalue Spectrum Comparison', fontsize=14)
    plt.legend(fontsize=16, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        save_path = os.path.join(outdir, f"hessian_spectrum_compare.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Compare] Saved spectrum comparison plot")
    plt.close()


def plot_logscale_compare(data_a, data_b, label_a, label_b, outdir):
    """
    绘图 2: |特征值| 对数刻度折线图对比
    """
    eigs_a = data_a['eigenvalues']
    eigs_b = data_b['eigenvalues']

    abs_a = np.abs(eigs_a) + 1e-16
    abs_b = np.abs(eigs_b) + 1e-16

    plt.figure(figsize=(9, 6))

    # Model A
    plt.semilogy(np.arange(len(abs_a)), abs_a,
                 color=COLOR_A, linewidth=1.5,
                 marker=MARKER_A, markersize=4, markeredgecolor='w',
                 label=label_a)

    # Model B
    plt.semilogy(np.arange(len(abs_b)), abs_b,
                 color=COLOR_B, linewidth=1.5,
                 marker=MARKER_B, markersize=4, markeredgecolor='w',
                 label=label_b)



    plt.legend(fontsize=20, framealpha=0.9, loc='lower left')
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        save_path = os.path.join(outdir, f"hessian_logscale_compare.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Compare] Saved log-scale comparison plot")
    plt.close()


def plot_distribution_compare(data_a, data_b, label_a, label_b, outdir):
    """
    绘图 3: 特征值分布直方图对比 (半透明重叠)
    """
    eigs_a = data_a['eigenvalues']
    eigs_b = data_b['eigenvalues']

    # 统一 bin 范围
    all_eigs = np.concatenate([eigs_a, eigs_b])
    n_bins = min(30, max(len(eigs_a), len(eigs_b)))
    bin_edges = np.linspace(all_eigs.min(), all_eigs.max(), n_bins + 1)

    plt.figure(figsize=(9, 6))

    plt.hist(eigs_a, bins=bin_edges, color=COLOR_A_FILL, edgecolor='w',
             linewidth=0.5, alpha=0.6, label=label_a)
    plt.hist(eigs_b, bins=bin_edges, color=COLOR_B_FILL, edgecolor='w',
             linewidth=0.5, alpha=0.6, label=label_b)

    plt.axvline(x=0, color='#333333', linewidth=1, linestyle='-')
    plt.xlabel('Eigenvalue', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Hessian Eigenvalue Distribution Comparison', fontsize=14)
    plt.legend(fontsize=16, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        save_path = os.path.join(outdir, f"hessian_distribution_compare.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Compare] Saved distribution comparison plot")
    plt.close()


def write_summary(data_a, data_b, label_a, label_b, outdir):
    """将两个模型的数值指标写入文本摘要文件"""
    txt_path = os.path.join(outdir, "hessian_compare_summary.txt")

    eigs_a = data_a['eigenvalues']
    eigs_b = data_b['eigenvalues']

    def _stats(eigs, lbl, d):
        lines = []
        lines.append(f"--- {lbl} ---")
        lines.append(f"  Num eigenvalues:       {len(eigs)}")
        lines.append(f"  lambda_max:            {eigs[0]:+.10e}")
        lines.append(f"  lambda_min:            {eigs[-1]:+.10e}")
        lines.append(f"  |lambda|_max:          {d['lambda_max']:.10e}")
        lines.append(f"  |lambda|_min (nonzero):{d['lambda_min']:.10e}")
        lines.append(f"  Condition Number κ:    {d['condition_number']:.10e}")
        n_pos = int(np.sum(eigs > 1e-10))
        n_neg = int(np.sum(eigs < -1e-10))
        n_zero = int(np.sum(np.abs(eigs) <= 1e-10))
        lines.append(f"  Positive: {n_pos}  Negative: {n_neg}  Near-zero: {n_zero}")
        lines.append(f"  Spectral range:        [{eigs[-1]:+.6e}, {eigs[0]:+.6e}]")
        lines.append(f"  Mean eigenvalue:       {np.mean(eigs):+.6e}")
        lines.append(f"  Std eigenvalue:        {np.std(eigs):.6e}")
        return lines

    with open(txt_path, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("  Hessian Eigenvalue Comparison Report\n")
        f.write("=" * 65 + "\n\n")

        for line in _stats(eigs_a, label_a, data_a):
            f.write(line + "\n")
        f.write("\n")
        for line in _stats(eigs_b, label_b, data_b):
            f.write(line + "\n")
        f.write("\n")

        f.write("--- Comparison ---\n")
        ratio_kappa = data_a['condition_number'] / (data_b['condition_number'] + 1e-30)
        f.write(f"  κ_A / κ_B = {ratio_kappa:.6f}\n")
        ratio_lmax = data_a['lambda_max'] / (data_b['lambda_max'] + 1e-30)
        f.write(f"  |λ|_max_A / |λ|_max_B = {ratio_lmax:.6f}\n")
        f.write("\n")

        f.write("--- Interpretation Guide ---\n")
        f.write("  * 条件数 κ 越大 → 损失曲面越陡峭/各向异性 → 训练越困难\n")
        f.write("  * 条件数 κ 越小 → 损失曲面越平坦/各向同性 → 优化更稳定\n")
        f.write("  * 负特征值越多 → 当前点附近存在更多鞍点方向\n")
        f.write("  * 特征值谱衰减越快 → 有效维度越低，模型更易优化\n")

    print(f"[Compare] Saved comparison summary to {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Hessian eigenvalue spectra from two models")
    parser.add_argument("npz1", type=str, help="Path to first model's hessian_eigenvalues_seed_*.npz")
    parser.add_argument("npz2", type=str, help="Path to second model's hessian_eigenvalues_seed_*.npz")
    parser.add_argument("--label1", type=str, default="Model A",
                        help="Display label for the first model (default: Model A)")
    parser.add_argument("--label2", type=str, default="Model B",
                        help="Display label for the second model (default: Model B)")
    parser.add_argument("--outdir", type=str, default="./hessian_compare",
                        help="Output directory for comparison plots (default: ./hessian_compare)")
    args = parser.parse_args()

    # 验证输入文件
    for p in [args.npz1, args.npz2]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Hessian Eigenvalue Comparison")
    print(f"  Model A: {args.label1}  ({args.npz1})")
    print(f"  Model B: {args.label2}  ({args.npz2})")
    print(f"  Output:  {args.outdir}")
    print(f"{'='*60}\n")

    data_a = load_npz(args.npz1)
    data_b = load_npz(args.npz2)

    # 生成三张对比图 + 文本摘要
    plot_spectrum_compare(data_a, data_b, args.label1, args.label2, args.outdir)
    plot_logscale_compare(data_a, data_b, args.label1, args.label2, args.outdir)
    plot_distribution_compare(data_a, data_b, args.label1, args.label2, args.outdir)
    write_summary(data_a, data_b, args.label1, args.label2, args.outdir)

    print(f"\n[Compare] All done! Results saved to: {args.outdir}/")


if __name__ == "__main__":
    main()
