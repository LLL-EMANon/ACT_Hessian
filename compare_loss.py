"""
Loss Curve 对比可视化工具

支持 ACT / ACT_PAA / DP / DP_PAA 等任意模型的 loss 曲线对比。
可同时加载多个 npz 文件，绘制在同一张图上。

用法:
    python compare_loss.py <npz_file_1> <npz_file_2> [npz_file_3 ...] \
        --labels "Model A" "Model B" ["Model C" ...] \
        [--key loss] [--outdir ./loss_compare]

示例 (ACT vs ACT_PAA):
    python compare_loss.py \
        /path/to/ACT/act_ckpt/.../loss_curves_seed_0.npz \
        /path/to/ACT_PAA/act_ckpt/.../loss_curves_seed_0.npz \
        --labels "ACT" "ACT + PAA" \
        --outdir ./loss_compare

示例 (DP vs DP_PAA):
    python compare_loss.py \
        /path/to/DP_PAA/.../optim_analysis/loss_curves_seed_0.npz \
        /path/to/DP_PAA/.../optim_analysis/loss_curves_seed_0.npz \
        --labels "DP Run1" "DP Run2" \
        --outdir ./loss_compare

输出:
    1. train_loss_compare.png/pdf          — Train Loss 对比 (log scale)
    2. val_loss_compare.png/pdf            — Val Loss 对比 (log scale)
    3. train_val_loss_compare.png/pdf      — Train + Val 同图对比
    4. loss_compare_summary.txt            — 数值对比摘要
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
# 支持最多 6 个模型的对比配色 (与 compare_hessian.py 风格一致)
COLORS = [
    '#2B8CBE',   # 蓝
    '#E34A33',   # 红
    '#31A354',   # 绿
    '#756BB1',   # 紫
    '#FF7F00',   # 橙
    '#E7298A',   # 粉
]

COLORS_LIGHT = [
    '#A6BDDB',
    '#FDBB84',
    '#A1D99B',
    '#BCBDDC',
    '#FFD480',
    '#F4B6D2',
]

MARKERS = ['o', 's', '^', 'D', 'v', 'P']
LINESTYLES_TRAIN = ['-', '-', '-', '-', '-', '-']
LINESTYLES_VAL = ['--', '--', '--', '--', '--', '--']


def load_npz(path):
    """
    加载 npz 文件，自动适配 ACT 格式和 DP 格式。
    
    ACT 格式: train_loss, val_loss, train_loss_epochs, val_loss_epochs, ...
    DP 格式:  epochs, train_loss, val_loss, seed
    
    返回统一格式的字典。
    """
    data = np.load(path)
    keys = list(data.keys())
    
    result = {
        'train': {},   # key -> (epochs, values)
        'val': {},     # key -> (epochs, values)
        'path': path,
    }
    
    # DP format: has 'epochs', 'train_loss', 'val_loss'
    if 'epochs' in keys and 'train_loss' in keys:
        epochs = data['epochs']
        result['train']['loss'] = (epochs, data['train_loss'])
        if 'val_loss' in keys:
            val = data['val_loss']
            if np.isfinite(val).any():
                result['val']['loss'] = (epochs, val)
        return result
    
    # ACT format: train_{key}, val_{key}, train_{key}_epochs, val_{key}_epochs
    processed = set()
    for k in keys:
        if k.startswith('train_') and not k.endswith('_epochs') and k != 'train_loss_epochs':
            base_key = k[6:]  # remove 'train_' prefix
            if base_key in ('loss_epochs',):
                continue
            epoch_key = f"train_{base_key}_epochs"
            if epoch_key in keys:
                result['train'][base_key] = (data[epoch_key], data[k])
                processed.add(base_key)
        elif k.startswith('val_') and not k.endswith('_epochs') and k != 'val_loss_epochs':
            base_key = k[4:]  # remove 'val_' prefix
            if base_key in ('loss_epochs',):
                continue
            epoch_key = f"val_{base_key}_epochs"
            if epoch_key in keys:
                val = data[k]
                if np.isfinite(val).any():
                    result['val'][base_key] = (data[epoch_key], val)
                    processed.add(base_key)
    
    return result


def _smooth(values, window=5):
    """简单移动平均平滑"""
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def plot_single_key_compare(all_data, labels, key, mode, outdir, smooth_window=0):
    """
    绘制单个 loss key 的多模型对比图。
    
    mode: 'train', 'val', or 'both'
    """
    plt.figure(figsize=(9, 6))
    has_data = False
    
    for i, (data, label) in enumerate(zip(all_data, labels)):
        color = COLORS[i % len(COLORS)]
        color_light = COLORS_LIGHT[i % len(COLORS_LIGHT)]
        
        if mode in ('train', 'both') and key in data['train']:
            epochs, values = data['train'][key]
            if smooth_window > 1:
                sm = _smooth(values, smooth_window)
                ep_sm = epochs[:len(sm)]
                plt.plot(epochs, values, color=color, linewidth=0.3, alpha=0.3)
                plt.plot(ep_sm, sm, color=color, linewidth=1.5,
                         label=f"{label}")
            else:
                ls = LINESTYLES_TRAIN[i % len(LINESTYLES_TRAIN)]
                plt.plot(epochs, values, color=color, linewidth=1.5,
                         linestyle=ls, label=f"{label}")
            has_data = True
        
        if mode in ('val', 'both') and key in data['val']:
            epochs, values = data['val'][key]
            if smooth_window > 1:
                sm = _smooth(values, smooth_window)
                ep_sm = epochs[:len(sm)]
                plt.plot(epochs, values, color=color_light, linewidth=0.3, alpha=0.3)
                plt.plot(ep_sm, sm, color=color_light, linewidth=1.5,
                         linestyle='--', label=f"{label} (Val)")
            else:
                ls = LINESTYLES_VAL[i % len(LINESTYLES_VAL)]
                plt.plot(epochs, values, color=color, linewidth=1.5,
                         linestyle=ls, alpha=0.7, label=f"{label} (Val)")
            has_data = True
    
    if not has_data:
        plt.close()
        return
    
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=16)
    plt.legend(fontsize=20, framealpha=0.9, loc='best')
    plt.tight_layout()
    
    # Determine filename
    mode_tag = mode
    filename = f"{mode_tag}_{key}_compare"
    
    for ext in ['png', 'pdf']:
        save_path = os.path.join(outdir, f"{filename}.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Compare] Saved {filename} plot")
    plt.close()


def write_summary(all_data, labels, outdir):
    """将所有模型的数值指标写入文本摘要文件"""
    txt_path = os.path.join(outdir, "loss_compare_summary.txt")
    
    # Collect all keys
    all_keys = set()
    for data in all_data:
        all_keys.update(data['train'].keys())
        all_keys.update(data['val'].keys())
    all_keys = sorted(all_keys)
    
    with open(txt_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  Loss Curve Comparison Report\n")
        f.write("=" * 70 + "\n\n")
        
        for i, (data, label) in enumerate(zip(all_data, labels)):
            f.write(f"--- {label} ---\n")
            f.write(f"  Source: {data['path']}\n")
            
            for key in all_keys:
                if key in data['train']:
                    epochs, values = data['train'][key]
                    f.write(f"  Train {key}:\n")
                    f.write(f"    Initial:  {values[0]:.10e}\n")
                    f.write(f"    Final:    {values[-1]:.10e}\n")
                    f.write(f"    Min:      {np.min(values):.10e} (epoch {epochs[np.argmin(values)]:.0f})\n")
                    f.write(f"    Epochs:   {len(values)} data points, [{epochs[0]:.0f} .. {epochs[-1]:.0f}]\n")
                    reduction = (values[0] - values[-1]) / (values[0] + 1e-30) * 100
                    f.write(f"    Reduction: {reduction:.2f}%\n")
                
                if key in data['val']:
                    epochs, values = data['val'][key]
                    finite_mask = np.isfinite(values)
                    if finite_mask.any():
                        valid = values[finite_mask]
                        valid_ep = epochs[finite_mask]
                        f.write(f"  Val {key}:\n")
                        f.write(f"    Initial:  {valid[0]:.10e}\n")
                        f.write(f"    Final:    {valid[-1]:.10e}\n")
                        f.write(f"    Min:      {np.min(valid):.10e} (epoch {valid_ep[np.argmin(valid)]:.0f})\n")
            f.write("\n")
        
        # Pairwise comparison (if 2 models)
        if len(all_data) == 2:
            f.write("--- Pairwise Comparison ---\n")
            d1, d2 = all_data
            l1, l2 = labels
            
            for key in all_keys:
                if key in d1['train'] and key in d2['train']:
                    _, v1 = d1['train'][key]
                    _, v2 = d2['train'][key]
                    f1, f2 = v1[-1], v2[-1]
                    improvement = (f1 - f2) / (f1 + 1e-30) * 100
                    better = l2 if f2 < f1 else l1
                    f.write(f"  Train {key}: {l1}={f1:.6e}, {l2}={f2:.6e}\n")
                    f.write(f"    Δ = {improvement:+.2f}% ({better} is lower)\n")
                
                if key in d1['val'] and key in d2['val']:
                    _, v1 = d1['val'][key]
                    _, v2 = d2['val'][key]
                    v1_fin = v1[np.isfinite(v1)]
                    v2_fin = v2[np.isfinite(v2)]
                    if len(v1_fin) > 0 and len(v2_fin) > 0:
                        f1, f2 = v1_fin[-1], v2_fin[-1]
                        improvement = (f1 - f2) / (f1 + 1e-30) * 100
                        better = l2 if f2 < f1 else l1
                        f.write(f"  Val {key}: {l1}={f1:.6e}, {l2}={f2:.6e}\n")
                        f.write(f"    Δ = {improvement:+.2f}% ({better} is lower)\n")
            f.write("\n")
        
        f.write("--- Interpretation Guide ---\n")
        f.write("  * Loss 越低 → 模型拟合越好\n")
        f.write("  * Train-Val gap 越小 → 泛化越好，过拟合风险越低\n")
        f.write("  * 收敛速度: 相同 epoch 下 loss 更低 = 收敛更快\n")
        f.write("  * 对数刻度下波动更明显，可发现训练不稳定性\n")
    
    print(f"[Compare] Saved comparison summary to {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare loss curves from multiple models")
    parser.add_argument("npz_files", nargs='+', type=str,
                        help="Paths to loss_curves_seed_*.npz files")
    parser.add_argument("--labels", nargs='+', type=str, default=None,
                        help="Display labels for each model (default: Model A, Model B, ...)")
    parser.add_argument("--key", type=str, default="loss",
                        help="Which loss key to plot (default: loss). "
                             "ACT examples: loss, l1, kl")
    parser.add_argument("--smooth", type=int, default=0,
                        help="Moving average window for smoothing (0=disabled)")
    parser.add_argument("--outdir", type=str, default="./loss_compare",
                        help="Output directory for comparison plots")
    args = parser.parse_args()
    
    # Validate inputs
    for p in args.npz_files:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")
    
    n = len(args.npz_files)
    if n > len(COLORS):
        print(f"WARNING: More than {len(COLORS)} models; colors will repeat.")
    
    # Default labels
    if args.labels is None:
        labels = [chr(65 + i) if i < 26 else f"Model {i+1}" for i in range(n)]
    else:
        if len(args.labels) != n:
            raise ValueError(f"Number of labels ({len(args.labels)}) must match "
                           f"number of npz files ({n})")
        labels = args.labels
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  Loss Curve Comparison")
    for i, (p, l) in enumerate(zip(args.npz_files, labels)):
        print(f"  [{l}]: {p}")
    print(f"  Output:  {args.outdir}")
    print(f"  Key:     {args.key}")
    print(f"{'='*60}\n")
    
    # Load all data
    all_data = [load_npz(p) for p in args.npz_files]
    
    # Report available keys
    for data, label in zip(all_data, labels):
        train_keys = sorted(data['train'].keys())
        val_keys = sorted(data['val'].keys())
        print(f"  [{label}] Train keys: {train_keys}, Val keys: {val_keys}")
    print()
    
    key = args.key
    
    # Plot 1: Train loss comparison
    plot_single_key_compare(all_data, labels, key, 'train', args.outdir, args.smooth)
    
    # Plot 2: Val loss comparison
    plot_single_key_compare(all_data, labels, key, 'val', args.outdir, args.smooth)
    
    # Plot 3: Train + Val combined
    plot_single_key_compare(all_data, labels, key, 'both', args.outdir, args.smooth)
    
    # If ACT data, also plot l1 and kl if available
    extra_keys = set()
    for data in all_data:
        extra_keys.update(data['train'].keys())
    extra_keys.discard(key)  # don't replot the main key
    
    for ek in sorted(extra_keys):
        # Check if at least one model has this key
        has_key = any(ek in d['train'] for d in all_data)
        if has_key:
            plot_single_key_compare(all_data, labels, ek, 'train', args.outdir, args.smooth)
            plot_single_key_compare(all_data, labels, ek, 'val', args.outdir, args.smooth)
    
    # Write summary
    write_summary(all_data, labels, args.outdir)
    
    print(f"\n[Compare] All done! Results saved to: {args.outdir}/")


if __name__ == "__main__":
    main()
