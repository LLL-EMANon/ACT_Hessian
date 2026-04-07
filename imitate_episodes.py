import os
import torch
import numpy as np
import pickle
import argparse
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import IPython
import math
import matplotlib.ticker as ticker

# Set rendering backend for MuJoCo
os.environ["MUJOCO_GL"] = "egl"
matplotlib.use("Agg")  # 用于后台绘图

# 假设这些是你项目中的现有模块
from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data  # data functions
from utils import sample_box_pose, sample_insertion_pose  # robot functions
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from act_policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos
from sim_env import BOX_POSE

mpl_settings_applied = False
try:
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl_settings_applied = True
except:
    pass

e = IPython.embed

# =============================================
# Parameter Flattening Helpers (必备工具函数)
# =============================================
def _flatten_grads(grads, params):
    """将梯度展平为一维向量"""
    flats = []
    for g, p in zip(grads, params):
        if g is None:
            flats.append(torch.zeros_like(p).reshape(-1))
        else:
            flats.append(g.reshape(-1))
    return torch.cat(flats)

def _flatten_params(params):
    """将参数展平为一维向量"""
    return torch.cat([p.detach().reshape(-1) for p in params])

def _assign_flat_to_params(flat, params):
    """将一维向量赋值回参数"""
    idx = 0
    for p in params:
        num = p.numel()
        with torch.no_grad():
            p.copy_(flat[idx:idx + num].view_as(p))
        idx += num

# =============================================
# NEW: True Loss Landscape Visualization
# =============================================
def compute_loss_components(data, policy):
    """
    计算 l1 和 kl loss 的独立值（供梯度分析使用）
    返回：l1_loss, kl_loss, total_loss
    """
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    
    forward_dict = policy(qpos_data, image_data, action_data, is_pad)
    return forward_dict.get('l1', forward_dict['loss']), forward_dict.get('kl', torch.tensor(0.0).cuda()), forward_dict['loss']


def plot_gradient_dynamics(angle_hist, grad_ratio_hist, epochs_recorded, ckpt_dir, seed):
    """
    绘制梯度动态曲线：l1 和 kl 梯度的夹角、梯度范数比例
    类似于 exp3_optm_dynamic.py 中的功能
    """
    epochs_arr = np.array(epochs_recorded)
    
    # 1. 绘制梯度夹角曲线 (cos(angle))
    plt.figure(figsize=(6, 5))
    plt.plot(epochs_arr, angle_hist, color='#2B8CBE', linewidth=1.5)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        save_path = os.path.join(ckpt_dir, f"gradient_angle_seed_{seed}.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Visualizer] Saved gradient angle plot to {ckpt_dir}")
    plt.close()
    
    # 2. 绘制梯度范数比例曲线
    plt.figure(figsize=(6, 5))
    plt.plot(epochs_arr, grad_ratio_hist, color='#FF6600', linewidth=1.5)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        save_path = os.path.join(ckpt_dir, f"gradient_ratio_seed_{seed}.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Visualizer] Saved gradient ratio plot to {ckpt_dir}")
    plt.close()


def plot_kl_loss_landscape(policy, vis_data, traj_epochs, traj_vecs, ckpt_dir, seed):
    """
    真实计算 KL Loss Landscape。
    与 plot_true_loss_landscape 类似，但只绘制 KL divergence 的 loss landscape。
    """
    if len(traj_vecs) < 3:
        print("Not enough trajectory points to plot KL landscape.")
        return None

    print(f"\n[Visualizer] Starting KL Loss Landscape calculation...")

    # 1. 确定锚点 (Start, Mid, End) 用于定义平面
    p0 = traj_vecs[0]
    p2 = traj_vecs[-1]
    mid_idx = len(traj_vecs) // 2
    p1 = traj_vecs[mid_idx]

    # 2. 构建平面正交基 (Gram-Schmidt)
    u = (p1 - p0).float()
    v = (p2 - p0).float()
    e1 = u / (u.norm() + 1e-12)
    v_orth = v - torch.dot(v, e1) * e1
    e2 = v_orth / (v_orth.norm() + 1e-12)

    # 3. 投影轨迹
    xs, ys = [], []
    for w in traj_vecs:
        d = (w - p0).float()
        xs.append(torch.dot(d, e1).item())
        ys.append(torch.dot(d, e2).item())

    # 4. 定义网格范围
    margin = 0.2
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w_range = max(x_max - x_min, 1e-6)
    h_range = max(y_max - y_min, 1e-6)
    x_min -= w_range * margin
    x_max += w_range * margin
    y_min -= h_range * margin
    y_max += h_range * margin

    grid_res = 25
    x_lin = np.linspace(x_min, x_max, grid_res)
    y_lin = np.linspace(y_min, y_max, grid_res)
    Xg, Yg = np.meshgrid(x_lin, y_lin)
    Z = np.zeros_like(Xg)

    # 5. 备份参数并扫描
    params = [p for p in policy.parameters() if p.requires_grad]
    w_backup = _flatten_params(params).detach().clone()
    policy.eval()
    
    p0_cuda = p0.cuda()
    e1_cuda = e1.cuda()
    e2_cuda = e2.cuda()

    print(f"[Visualizer] Scanning {grid_res}x{grid_res} grid for KL loss...")
    
    with torch.no_grad():
        for i in tqdm(range(grid_res), desc="Scanning KL Grid Y"):
            for j in range(grid_res):
                x_val = Xg[i, j]
                y_val = Yg[i, j]
                w_new = p0_cuda + x_val * e1_cuda + y_val * e2_cuda
                _assign_flat_to_params(w_new, params)
                
                l1_loss, kl_loss, total_loss = compute_loss_components(vis_data, policy)
                Z[i, j] = kl_loss.item()

    _assign_flat_to_params(w_backup.cuda(), params)
    print("[Visualizer] KL landscape calculation finished.")

    # 6. 绘图
    plt.figure(figsize=(8, 6))
    Z_log = np.log10(Z + 1e-16)
    levels = np.linspace(Z_log.min(), Z_log.max(), 35)
    cs = plt.contourf(Xg, Yg, Z_log, levels=levels, cmap='viridis', alpha=0.9)
    cbar = plt.colorbar(cs)
    cbar.ax.tick_params(labelsize=10)
    
    plt.contour(Xg, Yg, Z_log, levels=levels[::2], colors='white', alpha=0.3, linewidths=0.5)
    plt.plot(xs, ys, 'w-', linewidth=2.0, alpha=0.9)
    plt.scatter(xs, ys, c='#FF6600', s=25, edgecolors='w', linewidths=0.5, zorder=5)
    
    plt.scatter([xs[0]], [ys[0]], c='#FF0000', marker='*', s=150, edgecolors='w', zorder=10)
    plt.scatter([xs[-1]], [ys[-1]], c='#FF0000', marker='X', s=150, edgecolors='w', zorder=10)

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    
    for ext in ['png', 'pdf']:
        save_path = os.path.join(ckpt_dir, f"kl_loss_landscape_seed_{seed}.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Visualizer] Saved KL landscape to {ckpt_dir}")
    plt.close()

    return (Xg, Yg, Z_log, xs, ys)


def plot_true_loss_landscape(policy, vis_data, traj_epochs, traj_vecs, ckpt_dir, seed):
    """
    真实计算 Loss Landscape。
    原理：使用 Start(p0), Mid(p1), End(p2) 三个点定义一个 2D 平面，
    然后在该平面上网格化扫描，对每个点进行一次 Forward Pass 计算真实 Loss。
    """
    if len(traj_vecs) < 3:
        print("Not enough trajectory points to plot landscape.")
        return None

    print(f"\n[Visualizer] Starting True Loss Landscape calculation...")

    # 1. 确定锚点 (Start, Mid, End) 用于定义平面
    p0 = traj_vecs[0]           # Epoch 0
    p2 = traj_vecs[-1]          # Final Epoch
    mid_idx = len(traj_vecs) // 2
    p1 = traj_vecs[mid_idx]     # Middle Epoch

    # 2. 构建平面正交基 (Gram-Schmidt) - 在 CPU 上计算
    u = (p1 - p0).float()
    v = (p2 - p0).float()
    
    # e1: 归一化的 u
    e1 = u / (u.norm() + 1e-12)
    # v_orth: v 在 e1 上的垂线分量
    v_orth = v - torch.dot(v, e1) * e1
    # e2: 归一化的 v_orth
    e2 = v_orth / (v_orth.norm() + 1e-12)

    # 3. 投影轨迹 (计算所有轨迹点在 2D 平面上的坐标 x, y)
    xs = []
    ys = []
    for w in traj_vecs:
        d = (w - p0).float()
        xs.append(torch.dot(d, e1).item())
        ys.append(torch.dot(d, e2).item())

    # 4. 定义网格范围 (Landscape Range)
    margin = 0.2  # 边界留白 20%
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    w_range = x_max - x_min
    h_range = y_max - y_min
    
    # 防止范围过小（例如只训练了1个epoch）
    if w_range < 1e-6: w_range = 1.0
    if h_range < 1e-6: h_range = 1.0
    
    x_min -= w_range * margin
    x_max += w_range * margin
    y_min -= h_range * margin
    y_max += h_range * margin

    # 网格分辨率 (越高越慢，25x25 = 625次 forward passes)
    grid_res = 25 
    x_lin = np.linspace(x_min, x_max, grid_res)
    y_lin = np.linspace(y_min, y_max, grid_res)
    Xg, Yg = np.meshgrid(x_lin, y_lin)
    Z = np.zeros_like(Xg)

    # 5. 真实计算 Loss Loop
    # 备份当前模型参数
    params = [p for p in policy.parameters() if p.requires_grad]
    w_backup = _flatten_params(params).detach().clone()
    
    # 切换到 eval 模式 (固定 BN/Dropout)
    policy.eval() 
    
    # 将基向量移到 GPU 以便快速计算组合
    p0_cuda = p0.cuda()
    e1_cuda = e1.cuda()
    e2_cuda = e2.cuda()

    print(f"[Visualizer] Scanning {grid_res}x{grid_res} grid with fixed validation batch...")
    
    # 使用 no_grad 极度重要，防止 OOM
    with torch.no_grad():
        for i in tqdm(range(grid_res), desc="Scanning Grid Y"):
            for j in range(grid_res):
                # 计算当前网格点的参数: w = p0 + x*e1 + y*e2
                x_val = Xg[i, j]
                y_val = Yg[i, j]
                
                # 合成新权重
                w_new = p0_cuda + x_val * e1_cuda + y_val * e2_cuda
                
                # 将参数加载到模型
                _assign_flat_to_params(w_new, params)
                
                # 计算 Loss (复用 forward_pass)
                # 注意：forward_pass 内部会将 data 移到 CUDA，所以这里传入 CPU data 也没问题
                out_dict = forward_pass(vis_data, policy) 
                
                # 存入 Z (必须使用 .item() 释放计算图)
                Z[i, j] = out_dict['loss'].item()

    # 恢复原始参数
    _assign_flat_to_params(w_backup.cuda(), params)
    print("[Visualizer] Calculation finished.")

    # 6. 绘图
    plt.figure(figsize=(8, 6))
    
    # 使用 Log 尺度，因为 Loss 在优化初期和末期差异巨大
    Z_log = np.log10(Z + 1e-16)
    
    # 绘制等高线 (Background)
    levels = np.linspace(Z_log.min(), Z_log.max(), 35)
    cs = plt.contourf(Xg, Yg, Z_log, levels=levels, cmap='viridis', alpha=0.9)
    cbar = plt.colorbar(cs)
    cbar.ax.tick_params(labelsize=10)
    
    # 添加等高线线条
    plt.contour(Xg, Yg, Z_log, levels=levels[::2], colors='white', alpha=0.3, linewidths=0.5)

    # 绘制优化轨迹 (Foreground)
    plt.plot(xs, ys, 'w-', linewidth=2.0, alpha=0.9)
    plt.scatter(xs, ys, c='#FF6600', s=25, edgecolors='w', linewidths=0.5, zorder=5)
    
    # 标记关键点
    plt.scatter([xs[0]], [ys[0]], c='#FF0000', marker='*', s=150, edgecolors='w', zorder=10)
    plt.scatter([xs[-1]], [ys[-1]], c='#FF0000', marker='X', s=150, edgecolors='w', zorder=10)

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    
    for ext in ['png', 'pdf']:
        save_path = os.path.join(ckpt_dir, f"true_loss_landscape_seed_{seed}.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Visualizer] Saved true landscape to {ckpt_dir}")
    plt.close()

    return (Xg, Yg, Z_log, xs, ys)

from mpl_toolkits.mplot3d import Axes3D  # 必须导入

def plot_true_3d_loss_landscape(policy, vis_data, traj_epochs, traj_vecs, ckpt_dir, seed):
    """
    V5 Style-Matched: 严格按照 2D plot 的样式参数绘制 3D 图。
    包括：Coolwarm曲面, Viridis轨迹点, Lime色起终点, 线宽1.5等。
    """
    if len(traj_vecs) < 3:
        print("Not enough trajectory points to plot landscape.")
        return

    print(f"\n[Visualizer] Starting True Loss Landscape calculation (Style Matched)...")

    # 1. 锚点与基向量 (保持不变)
    p0 = traj_vecs[0]; p2 = traj_vecs[-1]; mid_idx = len(traj_vecs) // 2; p1 = traj_vecs[mid_idx]
    u = (p1 - p0).float(); v = (p2 - p0).float()
    e1 = u / (u.norm() + 1e-12)
    v_orth = v - torch.dot(v, e1) * e1
    e2 = v_orth / (v_orth.norm() + 1e-12)

    # 2. 投影轨迹 (保持不变)
    xs, ys = [], []
    for w in traj_vecs:
        d = (w - p0).float()
        xs.append(torch.dot(d, e1).item())
        ys.append(torch.dot(d, e2).item())

    # 3. 网格定义 (保持不变)
    margin = 0.2
    x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys)
    w_range = max(x_max - x_min, 1e-6); h_range = max(y_max - y_min, 1e-6)
    x_min -= w_range * margin; x_max += w_range * margin
    y_min -= h_range * margin; y_max += h_range * margin
    grid_res = 25 
    x_lin = np.linspace(x_min, x_max, grid_res); y_lin = np.linspace(y_min, y_max, grid_res)
    Xg, Yg = np.meshgrid(x_lin, y_lin)
    Z = np.zeros_like(Xg)

    # 4. 计算网格曲面 (保持不变)
    params = [p for p in policy.parameters() if p.requires_grad]
    w_backup = _flatten_params(params).detach().clone()
    policy.eval()
    p0_cuda = p0.cuda(); e1_cuda = e1.cuda(); e2_cuda = e2.cuda()

    print(f"[Visualizer] Scanning {grid_res}x{grid_res} grid surface...")
    with torch.no_grad():
        for i in tqdm(range(grid_res), desc="Scanning Grid"):
            for j in range(grid_res):
                x_val = Xg[i, j]; y_val = Yg[i, j]
                w_new = p0_cuda + x_val * e1_cuda + y_val * e2_cuda
                _assign_flat_to_params(w_new, params)
                out_dict = forward_pass(vis_data, policy) 
                Z[i, j] = out_dict['loss'].item()

    # 5. 计算真实 Loss 和 投影 Loss (保持不变)
    zs_true = []; zs_projected = []
    with torch.no_grad():
        for k, w in enumerate(traj_vecs):
            _assign_flat_to_params(w.cuda(), params)
            zs_true.append(np.log10(forward_pass(vis_data, policy)['loss'].item() + 1e-16))
            
            w_proj = p0_cuda + xs[k] * e1_cuda + ys[k] * e2_cuda
            _assign_flat_to_params(w_proj, params)
            zs_projected.append(np.log10(forward_pass(vis_data, policy)['loss'].item() + 1e-16))

    _assign_flat_to_params(w_backup.cuda(), params)

    # 6. 绘图 (样式修改重点)
    Z_log = np.log10(Z + 1e-16) 
    views = [(30, 45, "iso"), (90, -90, "top"), (0, 90, "side_x"), (45, 135, "angle_2")]

    for elev, azim, tag in views:
        fig = plt.figure(figsize=(10, 8)) # 稍微调小一点，接近2D图比例
        ax = fig.add_subplot(111, projection='3d') 

        # [Style 1] 曲面颜色: viridis (蓝绿黄)
        surf = ax.plot_surface(Xg, Yg, Z_log, cmap='viridis', 
                             edgecolor='none', alpha=0.9, rstride=1, cstride=1, antialiased=True)
        
        # [Style 2] 轨迹线: 白色, linewidth=2.0
        ax.plot(xs, ys, zs_true, 'w-', linewidth=2.0, alpha=0.9, zorder=20)
        
        # [Style 3] 轨迹点: 亮橙色
        ax.scatter(xs, ys, zs_true, c='#FF6600', s=25, edgecolors='w', linewidths=0.5, depthshade=False, zorder=21)

        # [Style 4] 起点: 红色星
        ax.scatter(xs[0], ys[0], zs_true[0], c='#FF0000', marker='*', s=150, 
                   edgecolors='w', linewidth=1.5, zorder=22)
        
        # [Style 5] 终点: 红色X
        ax.scatter(xs[-1], ys[-1], zs_true[-1], c='#FF0000', marker='X', s=150, 
                   edgecolors='w', linewidth=1.5, zorder=22)

        # [辅助线] 垂直投影线
        step = max(1, len(xs) // 40) 
        for k in range(0, len(xs), step):
            color = '#FF6600' if zs_true[k] < zs_projected[k] else 'gray'
            ax.plot([xs[k], xs[k]], [ys[k], ys[k]], [zs_projected[k], zs_true[k]], 
                    color=color, linestyle=':', linewidth=0.8, alpha=0.5)

        # [辅助线] 地面投影 (Shadow)
        ax.plot(xs, ys, zs_projected, color='white', linestyle='--', linewidth=1, alpha=0.3, zorder=15)

        # 设置 Z 轴范围，保留数字刻度
        z_min_plot = min(min(zs_true), Z_log.min()) - 0.2
        z_max_plot = max(max(zs_true), Z_log.max()) + 0.2
        ax.set_zlim(z_min_plot, z_max_plot)
        ax.tick_params(axis='both', labelsize=8)
        
        ax.view_init(elev=elev, azim=azim)
        
        for ext in ['png', 'pdf']:
            save_path = os.path.join(ckpt_dir, f"academic_landscape_3d_{tag}_seed_{seed}.{ext}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved matched-style plot {tag}")
        plt.close()


# =============================================
# Zoomed-in Loss Landscape Visualization
# =============================================

def plot_zoomed_loss_landscape(landscape_data, traj_epochs, ckpt_dir, seed, zoom_fraction=0.5):
    """
    局部放大 2D Total Loss Landscape：对原始全局 landscape 进行裁剪放大（等高线形状完全一致）。
    保存格式：PNG + PDF
    """
    if landscape_data is None:
        print("[Visualizer] No total landscape data for zoomed plot.")
        return

    Xg, Yg, Z_log, xs, ys = landscape_data

    # 选取最后部分轨迹点定义放大区域
    n_zoom = max(3, int(len(xs) * zoom_fraction))
    zi = len(xs) - n_zoom
    xs_zoom = xs[zi:]
    ys_zoom = ys[zi:]
    epochs_zoom = traj_epochs[zi:]

    # 定义裁剪范围
    margin = 0.15
    x_min_z, x_max_z = min(xs_zoom), max(xs_zoom)
    y_min_z, y_max_z = min(ys_zoom), max(ys_zoom)
    w_range = max(x_max_z - x_min_z, 1e-6)
    h_range = max(y_max_z - y_min_z, 1e-6)
    x_min_z -= w_range * margin; x_max_z += w_range * margin
    y_min_z -= h_range * margin; y_max_z += h_range * margin

    # 绘图（使用原始全局网格数据，仅裁剪视图 → 等高线形状完全一致）
    plt.figure(figsize=(8, 6))
    levels = np.linspace(Z_log.min(), Z_log.max(), 35)
    cs = plt.contourf(Xg, Yg, Z_log, levels=levels, cmap='viridis', alpha=0.9)
    cbar = plt.colorbar(cs)
    cbar.ax.tick_params(labelsize=10)

    plt.contour(Xg, Yg, Z_log, levels=levels[::2], colors='white', alpha=0.3, linewidths=0.5)

    plt.plot(xs_zoom, ys_zoom, 'w-', linewidth=2.0, alpha=0.9)
    plt.scatter(xs_zoom, ys_zoom, c='#FF6600', s=25, edgecolors='w', linewidths=0.5, zorder=5)

    plt.scatter([xs_zoom[0]], [ys_zoom[0]], c='#FF0000', marker='*', s=150, edgecolors='w', zorder=10)
    plt.scatter([xs_zoom[-1]], [ys_zoom[-1]], c='#FF0000', marker='X', s=150, edgecolors='w', zorder=10)

    plt.xlim(x_min_z, x_max_z)
    plt.ylim(y_min_z, y_max_z)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        save_path = os.path.join(ckpt_dir, f"zoomed_total_loss_landscape_seed_{seed}.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Visualizer] Saved zoomed 2D total loss landscape to {ckpt_dir}")
    plt.close()


def plot_zoomed_kl_loss_landscape(landscape_data, traj_epochs, ckpt_dir, seed, zoom_fraction=0.5):
    """
    局部放大 2D KL Loss Landscape：对原始全局 KL landscape 进行裁剪放大（等高线形状完全一致）。
    保存格式：PNG + PDF
    """
    if landscape_data is None:
        print("[Visualizer] No KL landscape data for zoomed plot.")
        return

    Xg, Yg, Z_log, xs, ys = landscape_data

    # 选取最后部分轨迹点定义放大区域
    n_zoom = max(3, int(len(xs) * zoom_fraction))
    zi = len(xs) - n_zoom
    xs_zoom = xs[zi:]
    ys_zoom = ys[zi:]
    epochs_zoom = traj_epochs[zi:]

    # 定义裁剪范围
    margin = 0.15
    x_min_z, x_max_z = min(xs_zoom), max(xs_zoom)
    y_min_z, y_max_z = min(ys_zoom), max(ys_zoom)
    w_range = max(x_max_z - x_min_z, 1e-6)
    h_range = max(y_max_z - y_min_z, 1e-6)
    x_min_z -= w_range * margin; x_max_z += w_range * margin
    y_min_z -= h_range * margin; y_max_z += h_range * margin

    # 绘图（使用原始全局网格数据，仅裁剪视图 → 等高线形状完全一致）
    plt.figure(figsize=(8, 6))
    levels = np.linspace(Z_log.min(), Z_log.max(), 35)
    cs = plt.contourf(Xg, Yg, Z_log, levels=levels, cmap='viridis', alpha=0.9)
    cbar = plt.colorbar(cs)
    cbar.ax.tick_params(labelsize=10)

    plt.contour(Xg, Yg, Z_log, levels=levels[::2], colors='white', alpha=0.3, linewidths=0.5)

    plt.plot(xs_zoom, ys_zoom, 'w-', linewidth=2.0, alpha=0.9)
    plt.scatter(xs_zoom, ys_zoom, c='#FF6600', s=25, edgecolors='w', linewidths=0.5, zorder=5)

    plt.scatter([xs_zoom[0]], [ys_zoom[0]], c='#FF0000', marker='*', s=150, edgecolors='w', zorder=10)
    plt.scatter([xs_zoom[-1]], [ys_zoom[-1]], c='#FF0000', marker='X', s=150, edgecolors='w', zorder=10)

    plt.xlim(x_min_z, x_max_z)
    plt.ylim(y_min_z, y_max_z)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        save_path = os.path.join(ckpt_dir, f"zoomed_kl_loss_landscape_seed_{seed}.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Visualizer] Saved zoomed 2D KL loss landscape to {ckpt_dir}")
    plt.close()


def plot_zoomed_3d_loss_landscape(policy, vis_data, traj_epochs, traj_vecs, ckpt_dir, seed, zoom_fraction=0.5):
    """
    局部放大 3D Total Loss Landscape：聚焦最后一部分训练轨迹。
    严格匹配 plot_true_3d_loss_landscape 风格 (viridis曲面, 橙色轨迹点, 红色起终点)。
    保存格式：PNG + PDF
    """
    if len(traj_vecs) < 3:
        print("Not enough trajectory points for zoomed 3D landscape.")
        return

    print(f"\n[Visualizer] Starting Zoomed 3D Loss Landscape (last {zoom_fraction*100:.0f}%)...")

    # 1. 构建平面基向量
    p0 = traj_vecs[0]
    p2 = traj_vecs[-1]
    mid_idx = len(traj_vecs) // 2
    p1 = traj_vecs[mid_idx]
    u = (p1 - p0).float()
    v = (p2 - p0).float()
    e1 = u / (u.norm() + 1e-12)
    v_orth = v - torch.dot(v, e1) * e1
    e2 = v_orth / (v_orth.norm() + 1e-12)

    # 2. 投影所有轨迹
    xs_all, ys_all = [], []
    for w in traj_vecs:
        d = (w - p0).float()
        xs_all.append(torch.dot(d, e1).item())
        ys_all.append(torch.dot(d, e2).item())

    # 3. 选取最后部分
    n_zoom = max(3, int(len(traj_vecs) * zoom_fraction))
    zi = len(traj_vecs) - n_zoom
    xs_zoom = xs_all[zi:]
    ys_zoom = ys_all[zi:]
    epochs_zoom = traj_epochs[zi:]
    vecs_zoom = traj_vecs[zi:]

    # 4. 定义放大网格
    margin = 0.15
    x_min, x_max = min(xs_zoom), max(xs_zoom)
    y_min, y_max = min(ys_zoom), max(ys_zoom)
    w_range = max(x_max - x_min, 1e-6)
    h_range = max(y_max - y_min, 1e-6)
    x_min -= w_range * margin; x_max += w_range * margin
    y_min -= h_range * margin; y_max += h_range * margin

    grid_res = 30
    x_lin = np.linspace(x_min, x_max, grid_res)
    y_lin = np.linspace(y_min, y_max, grid_res)
    Xg, Yg = np.meshgrid(x_lin, y_lin)
    Z = np.zeros_like(Xg)

    # 5. 扫描曲面 Loss
    params = [p for p in policy.parameters() if p.requires_grad]
    w_backup = _flatten_params(params).detach().clone()
    policy.eval()
    p0_cuda, e1_cuda, e2_cuda = p0.cuda(), e1.cuda(), e2.cuda()

    print(f"[Visualizer] Scanning {grid_res}x{grid_res} zoomed 3D grid...")
    with torch.no_grad():
        for i in tqdm(range(grid_res), desc="Zoomed 3D Grid"):
            for j in range(grid_res):
                w_new = p0_cuda + Xg[i, j] * e1_cuda + Yg[i, j] * e2_cuda
                _assign_flat_to_params(w_new, params)
                out_dict = forward_pass(vis_data, policy)
                Z[i, j] = out_dict['loss'].item()

    # 6. 计算轨迹真实 Loss 和投影 Loss
    zs_true = []
    zs_projected = []
    with torch.no_grad():
        for k, w in enumerate(vecs_zoom):
            _assign_flat_to_params(w.cuda(), params)
            zs_true.append(np.log10(forward_pass(vis_data, policy)['loss'].item() + 1e-16))

            w_proj = p0_cuda + xs_zoom[k] * e1_cuda + ys_zoom[k] * e2_cuda
            _assign_flat_to_params(w_proj, params)
            zs_projected.append(np.log10(forward_pass(vis_data, policy)['loss'].item() + 1e-16))

    _assign_flat_to_params(w_backup.cuda(), params)

    # 7. 绘图 (严格匹配 plot_true_3d_loss_landscape 风格)
    Z_log = np.log10(Z + 1e-16)
    views = [(30, 45, "iso"), (90, -90, "top"), (0, 90, "side_x"), (45, 135, "angle_2")]

    for elev, azim, tag in views:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # [Style 1] 曲面: viridis (蓝绿黄)
        surf = ax.plot_surface(Xg, Yg, Z_log, cmap='viridis',
                               edgecolor='none', alpha=0.9, rstride=1, cstride=1, antialiased=True)

        # [Style 2] 轨迹线: 白色, linewidth=2.0
        ax.plot(xs_zoom, ys_zoom, zs_true, 'w-', linewidth=2.0, alpha=0.9, zorder=20)

        # [Style 3] 轨迹点: 亮橙色
        ax.scatter(xs_zoom, ys_zoom, zs_true, c='#FF6600', s=25,
                   edgecolors='w', linewidths=0.5, depthshade=False, zorder=21)

        # [Style 4] 起点: 红色星
        ax.scatter(xs_zoom[0], ys_zoom[0], zs_true[0], c='#FF0000', marker='*', s=150,
                   edgecolors='w', linewidth=1.5, zorder=22)

        # [Style 5] 终点: 红色X
        ax.scatter(xs_zoom[-1], ys_zoom[-1], zs_true[-1], c='#FF0000', marker='X', s=150,
                   edgecolors='w', linewidth=1.5, zorder=22)

        # 投影辅助线
        step = max(1, len(xs_zoom) // 40)
        for k in range(0, len(xs_zoom), step):
            color = '#FF6600' if zs_true[k] < zs_projected[k] else 'gray'
            ax.plot([xs_zoom[k], xs_zoom[k]], [ys_zoom[k], ys_zoom[k]],
                    [zs_projected[k], zs_true[k]],
                    color=color, linestyle=':', linewidth=0.8, alpha=0.5)

        # 地面投影
        ax.plot(xs_zoom, ys_zoom, zs_projected, color='white', linestyle='--',
                linewidth=1, alpha=0.3, zorder=15)

        z_min_plot = min(min(zs_true), Z_log.min()) - 0.2
        z_max_plot = max(max(zs_true), Z_log.max()) + 0.2
        ax.set_zlim(z_min_plot, z_max_plot)
        ax.tick_params(axis='both', labelsize=8)
        ax.view_init(elev=elev, azim=azim)

        for ext in ['png', 'pdf']:
            save_path = os.path.join(ckpt_dir, f"zoomed_3d_landscape_{tag}_seed_{seed}.{ext}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualizer] Saved zoomed 3D {tag}")
        plt.close()


# =============================================
# Hessian Eigenvalue Spectrum & Condition Number
# =============================================

def _lanczos_algorithm(hvp_fn, dim, num_steps=50, device='cuda'):
    """
    Lanczos 算法：近似计算对称矩阵 (Hessian) 的特征值。
    Q 向量存储在 CPU 以节省 GPU 显存。

    Args:
        hvp_fn: 矩阵-向量乘积函数 H @ v
        dim: 矩阵维度 (参数总数)
        num_steps: Lanczos 迭代步数
        device: 计算设备

    Returns:
        eigenvalues: numpy array, 近似特征值
        T_matrix: numpy array, 三对角矩阵
        alphas: list, 对角元素
        betas: list, 次对角元素
    """
    num_steps = min(num_steps, dim)

    alphas = []
    betas = []

    v = torch.randn(dim, device=device)
    v = v / (v.norm() + 1e-12)

    v_prev = torch.zeros_like(v)
    beta_prev = 0.0

    # Q 向量存储在 CPU 以节省 GPU 显存
    Q_cpu = [v.cpu()]

    for k in tqdm(range(num_steps), desc="Lanczos Iteration"):
        w = hvp_fn(v)
        w = w - beta_prev * v_prev

        alpha = torch.dot(v, w).item()
        alphas.append(alpha)

        w = w - alpha * v

        # 完全重新正交化 (CPU→GPU 传输，保证数值稳定性)
        for q_cpu in Q_cpu:
            q = q_cpu.to(device)
            w = w - torch.dot(w, q) * q

        beta = w.norm().item()

        if beta < 1e-10:
            print(f"[Lanczos] Converged at step {k+1} (beta={beta:.2e})")
            break

        betas.append(beta)

        v_prev = v.clone()
        v = w / beta
        Q_cpu.append(v.cpu())
        beta_prev = beta

    # 构建三对角矩阵 T
    n = len(alphas)
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = alphas[i]
    for i in range(min(len(betas), n - 1)):
        T[i, i + 1] = betas[i]
        T[i + 1, i] = betas[i]

    eigenvalues = np.linalg.eigvalsh(T)
    return eigenvalues, T, alphas, betas


def compute_and_plot_hessian_analysis(policy, vis_data, ckpt_dir, seed, num_lanczos_steps=50):
    """
    计算并可视化 Hessian 矩阵的特征值谱和条件数。
    对大模型 (>2M 参数) 使用参数子集 + 有限差分 HVP 避免 OOM。

    功能：
    1. 使用 Lanczos 算法近似计算 Hessian 特征值
    2. 绘制特征值谱图 (线性刻度 + 对数刻度)
    3. 计算条件数 κ = |λ_max| / |λ_min|
    4. 绘制特征值分布直方图
    5. 保存所有数值数据到 npz + txt 文件
    """
    print(f"\n{'='*60}")
    print(f"[Visualizer] Hessian Eigenvalue Spectrum Analysis")
    print(f"{'='*60}")

    all_params = [p for p in policy.parameters() if p.requires_grad]
    full_dim = sum(p.numel() for p in all_params)
    print(f"[Visualizer] Model has {full_dim:,} trainable parameters")

    # 大模型自动选择参数子集 (从最后几层选取，最具信息量)
    MAX_HESSIAN_PARAMS = 2_000_000
    if full_dim > MAX_HESSIAN_PARAMS:
        print(f"[Visualizer] Model too large for full Hessian ({full_dim:,} > {MAX_HESSIAN_PARAMS:,}).")
        print(f"[Visualizer] Selecting last-layer parameter subset...")
        selected_params = []
        selected_dim = 0
        for p in reversed(all_params):
            selected_params.append(p)
            selected_dim += p.numel()
            if selected_dim >= MAX_HESSIAN_PARAMS:
                break
        params = list(reversed(selected_params))
        dim = sum(p.numel() for p in params)
        print(f"[Visualizer] Selected {len(params)} param groups ({dim:,} params) for Hessian")
    else:
        params = all_params
        dim = full_dim

    # 自适应 Lanczos 步数
    num_steps = min(num_lanczos_steps, dim, 100)
    if dim > 5_000_000:
        num_steps = min(num_steps, 30)
    elif dim > 1_000_000:
        num_steps = min(num_steps, 40)
    print(f"[Visualizer] Using {num_steps} Lanczos steps (finite-diff HVP)")

    policy.eval()
    torch.cuda.empty_cache()

    try:
        # 使用有限差分法计算 Hessian-向量乘积 (避免 create_graph=True 的 OOM 问题)
        # Hv ≈ [∇L(θ+εv) - ∇L(θ-εv)] / (2ε)  (中心差分，精度 O(ε²))
        w_base = torch.cat([p.detach().reshape(-1) for p in params])

        def _set_hessian_params(flat):
            idx = 0
            for p in params:
                num = p.numel()
                with torch.no_grad():
                    p.copy_(flat[idx:idx + num].view_as(p))
                idx += num

        def hvp_fn(vec):
            eps = 1e-4 * max(1.0, w_base.norm().item()) / (vec.norm().item() + 1e-12)

            # ∇L(θ + εv)
            _set_hessian_params(w_base + eps * vec)
            policy.zero_grad()
            out_p = forward_pass(vis_data, policy)
            grads_p = torch.autograd.grad(out_p['loss'], params, allow_unused=True)
            g_plus = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).detach()
                                for g, p in zip(grads_p, params)])

            # ∇L(θ - εv)
            _set_hessian_params(w_base - eps * vec)
            policy.zero_grad()
            out_m = forward_pass(vis_data, policy)
            grads_m = torch.autograd.grad(out_m['loss'], params, allow_unused=True)
            g_minus = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).detach()
                                 for g, p in zip(grads_m, params)])

            # 恢复原参数 + 清显存
            _set_hessian_params(w_base)
            torch.cuda.empty_cache()
            return (g_plus - g_minus) / (2 * eps)

        print(f"[Visualizer] Running Lanczos algorithm ({num_steps} steps)...")
        eigenvalues, T_matrix, alphas, betas = _lanczos_algorithm(
            hvp_fn, dim, num_steps, device='cuda'
        )

        # 确保恢复原参数
        _set_hessian_params(w_base)

    except RuntimeError as e:
        print(f"[Visualizer] Hessian computation failed: {e}")
        print("[Visualizer] Skipping Hessian analysis.")
        return
    finally:
        policy.zero_grad()
        torch.cuda.empty_cache()

    # ===== 分析特征值 =====
    sorted_eigs = np.sort(eigenvalues)[::-1]  # 降序
    abs_eigs = np.abs(sorted_eigs)

    lambda_max = np.max(abs_eigs)
    # 条件数：排除接近零的特征值
    nonzero_mask = abs_eigs > 1e-10
    if nonzero_mask.sum() > 0:
        lambda_min = np.min(abs_eigs[nonzero_mask])
        condition_number = lambda_max / lambda_min
    else:
        lambda_min = 0.0
        condition_number = float('inf')

    n_positive = np.sum(sorted_eigs > 1e-10)
    n_negative = np.sum(sorted_eigs < -1e-10)
    n_near_zero = np.sum(np.abs(sorted_eigs) <= 1e-10)

    print(f"\n[Hessian Analysis Results]")
    print(f"  Num eigenvalues computed: {len(sorted_eigs)}")
    print(f"  lambda_max = {sorted_eigs[0]:.6e}")
    print(f"  lambda_min = {sorted_eigs[-1]:.6e}")
    print(f"  |lambda|_max = {lambda_max:.6e}")
    print(f"  |lambda|_min (nonzero) = {lambda_min:.6e}")
    print(f"  Condition Number kappa = {condition_number:.6e}")
    print(f"  Positive: {n_positive}, Negative: {n_negative}, Near-zero: {n_near_zero}")

    # ===== 保存数值数据 =====
    # NPZ 格式 (程序化访问)
    npz_path = os.path.join(ckpt_dir, f"hessian_eigenvalues_seed_{seed}.npz")
    np.savez(npz_path,
             eigenvalues=sorted_eigs,
             T_matrix=T_matrix,
             alphas=np.array(alphas),
             betas=np.array(betas),
             condition_number=np.array([condition_number]),
             lambda_max=np.array([lambda_max]),
             lambda_min=np.array([lambda_min]))
    print(f"[Visualizer] Saved eigenvalue data to {npz_path}")

    # TXT 格式 (人类可读)
    txt_path = os.path.join(ckpt_dir, f"hessian_analysis_seed_{seed}.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Hessian Eigenvalue Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model Parameters: {dim:,}\n")
        f.write(f"Lanczos Steps: {num_steps}\n")
        f.write(f"Num Eigenvalues Computed: {len(sorted_eigs)}\n\n")
        f.write(f"--- Summary ---\n")
        f.write(f"lambda_max (largest eigenvalue)    = {sorted_eigs[0]:.10e}\n")
        f.write(f"lambda_min (smallest eigenvalue)   = {sorted_eigs[-1]:.10e}\n")
        f.write(f"|lambda|_max                       = {lambda_max:.10e}\n")
        f.write(f"|lambda|_min (nonzero)              = {lambda_min:.10e}\n")
        f.write(f"Condition Number kappa = |lambda_max|/|lambda_min| = {condition_number:.10e}\n")
        f.write(f"Positive eigenvalues: {n_positive}\n")
        f.write(f"Negative eigenvalues: {n_negative}\n")
        f.write(f"Near-zero eigenvalues (|lambda|<=1e-10): {n_near_zero}\n\n")
        f.write(f"--- All Eigenvalues (descending) ---\n")
        for i, ev in enumerate(sorted_eigs):
            f.write(f"  lambda_{i:3d} = {ev:+.10e}\n")
        f.write(f"\n--- Tridiagonal Matrix T ({len(alphas)}x{len(alphas)}) ---\n")
        f.write(f"Diagonal (alphas):\n")
        for i, a in enumerate(alphas):
            f.write(f"  alpha_{i:3d} = {a:+.10e}\n")
        f.write(f"Off-diagonal (betas):\n")
        for i, b in enumerate(betas):
            f.write(f"  beta_{i:3d}  = {b:+.10e}\n")
    print(f"[Visualizer] Saved analysis report to {txt_path}")

    # ===== 绘图 1: 特征值谱 (柱状图, 线性刻度) =====
    indices = np.arange(len(sorted_eigs))
    # 使用 viridis (蓝绿黄) 渐变色为柱状图着色
    cmap_viridis = plt.cm.viridis
    norm_vals = (sorted_eigs - sorted_eigs.min()) / (sorted_eigs.max() - sorted_eigs.min() + 1e-16)
    colors_bar = [cmap_viridis(v) for v in norm_vals]

    plt.figure(figsize=(8, 6))
    plt.bar(indices, sorted_eigs, color=colors_bar, edgecolor='w', linewidth=0.3, width=0.8)
    plt.axhline(y=0, color='#FF6600', linewidth=0.8, linestyle='--')
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        save_path = os.path.join(ckpt_dir, f"hessian_eigenvalue_spectrum_seed_{seed}.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Visualizer] Saved eigenvalue spectrum plot")
    plt.close()

    # ===== 绘图 2: 特征值绝对值 (对数刻度) =====
    plt.figure(figsize=(8, 6))
    abs_eigs_plot = abs_eigs + 1e-16
    plt.semilogy(indices, abs_eigs_plot, color='#2B8CBE', linewidth=1.5, marker='o', markersize=4, markeredgecolor='w')

    # 标记最大和最小 (非零) 特征值
    plt.axhline(y=lambda_max, color='#FF6600', linestyle='--', linewidth=1, alpha=0.7)
    if lambda_min > 0:
        plt.axhline(y=lambda_min, color='#ADDD8E', linestyle='--', linewidth=1, alpha=0.7)

    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        save_path = os.path.join(ckpt_dir, f"hessian_eigenvalue_logscale_seed_{seed}.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Visualizer] Saved eigenvalue log-scale plot")
    plt.close()

    # ===== 绘图 3: 特征值分布直方图 =====
    plt.figure(figsize=(8, 6))
    n_bins = min(30, len(sorted_eigs))
    plt.hist(sorted_eigs, bins=n_bins, color='#2B8CBE', edgecolor='w', linewidth=0.5, alpha=0.8)
    plt.axvline(x=0, color='#FF6600', linewidth=1, linestyle='-')
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        save_path = os.path.join(ckpt_dir, f"hessian_eigenvalue_distribution_seed_{seed}.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Visualizer] Saved eigenvalue distribution plot")
    plt.close()

    print(f"[Visualizer] Hessian analysis complete.\n")


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]

    # get task parameters
    is_sim = task_name[:4] == "sim-"
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # fixed parameters
    state_dim = 14  # yiheng
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "save_freq": args['save_freq']
    }

    if is_eval:
        ckpt_names = [f"policy_best.ckpt"]
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train,
                                                           batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    
    # Start Training
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers  # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha

        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env

        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if "sim_transfer_cube" in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif "sim_insertion" in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if "images" in obs:
                    image_list.append(obs["images"])
                else:
                    image_list.append({"main": obs["image"]})
                qpos_numpy = np.array(obs["qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config["policy_class"] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1))
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers(
                [env.puppet_bot_left, env.puppet_bot_right],
                [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                move_time=0.5,
            )  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )

        if save_episode:
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(ckpt_dir, f"video{rollout_id}.mp4"),
            )

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    # save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    # ### NEW: 固定一批验证数据用于 Loss Landscape 计算 ###
    # 这样可以保证画地形图时 Loss 是可比的，且速度较快
    print("Fetching fixed validation batch for landscape visualization...")
    try:
        fixed_vis_data = next(iter(val_dataloader))
    except StopIteration:
        # Fallback if val_dataloader is empty, use train
        fixed_vis_data = next(iter(train_dataloader))
    # ### END NEW ###

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    # ===== Trajectory Logging Init =====
    params = [p for p in policy.parameters() if p.requires_grad]
    
    # 记录参数轨迹
    # stride: 控制记录频率，避免存太多点。总 epoch 很大时建议增大 stride。
    traj_stride = max(1, num_epochs // 50) 
    traj_epochs = []
    traj_vecs = [] # 存储 CPU 上的 Tensor 以节省显存
    
    # ===== NEW: 梯度动态分析 (来自 exp3_optm_dynamic.py) =====
    # 记录 l1 和 kl 梯度之间的夹角余弦值和梯度范数比例
    angle_hist = []       # cos(angle) between grad_l1 and grad_kl
    grad_ratio_hist = []  # ||grad_l1|| / ||grad_kl||
    gradient_record_epochs = []  # 记录梯度的 epoch 编号
    gradient_record_stride = max(1, num_epochs // 100)  # 梯度记录频率
    
    # 记录初始参数 (Epoch 0)
    w0 = _flatten_params(params).cpu()
    traj_epochs.append(0)
    traj_vecs.append(w0)

    for epoch in tqdm(range(num_epochs)):
        # print(f"\nEpoch {epoch}") # Tqdm handles progress
        
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        
        # print(f"Val loss:   {epoch_val_loss:.5f}")
        
        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            loss = forward_dict["loss"]
            
            # ===== NEW: 梯度夹角分析 (类似 exp3_optm_dynamic.py) =====
            # 仅在 ACT 策略且满足记录频率时计算梯度夹角
            if policy_class == "ACT" and ((epoch + 1) % gradient_record_stride == 0 or epoch == 0):
                if batch_idx == 0:  # 每个 epoch 只记录第一个 batch 的梯度信息
                    l1_loss = forward_dict.get('l1', None)
                    kl_loss = forward_dict.get('kl', None)
                    
                    if l1_loss is not None and kl_loss is not None and kl_loss.item() > 1e-12:
                        # 计算 l1 和 kl 的梯度
                        grads_l1 = torch.autograd.grad(l1_loss, params, retain_graph=True, 
                                                        create_graph=False, allow_unused=True)
                        g_l1 = _flatten_grads(grads_l1, params).detach()
                        
                        grads_kl = torch.autograd.grad(kl_loss, params, retain_graph=True, 
                                                        create_graph=False, allow_unused=True)
                        g_kl = _flatten_grads(grads_kl, params).detach()
                        
                        # 计算夹角余弦值和范数比例
                        dot = torch.dot(g_l1, g_kl)
                        norm_l1 = g_l1.norm() + 1e-12
                        norm_kl = g_kl.norm() + 1e-12
                        cos_angle = torch.clamp(dot / (norm_l1 * norm_kl), -1.0, 1.0)
                        ratio = (norm_l1 / norm_kl).item()
                        
                        angle_hist.append(cos_angle.item())
                        grad_ratio_hist.append(ratio)
                        gradient_record_epochs.append(epoch + 1)
            # ===== END NEW =====
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        
        epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
        epoch_train_loss = epoch_summary["loss"]
        # print(f"Train loss: {epoch_train_loss:.5f}")

        # ===== NEW: Record Trajectory =====
        if ((epoch + 1) % traj_stride == 0) or (epoch == num_epochs - 1):
            w = _flatten_params(params).cpu()
            traj_epochs.append(epoch + 1)
            traj_vecs.append(w)
            
        if (epoch + 1) % config['save_freq'] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    # ===== Save analysis data for offline replotting =====
    analysis_data_path = os.path.join(ckpt_dir, f"analysis_data_seed_{seed}.pt")
    torch.save({
        "traj_epochs": traj_epochs,
        "traj_vecs": traj_vecs,
        "angle_hist": angle_hist,
        "grad_ratio_hist": grad_ratio_hist,
        "gradient_record_epochs": gradient_record_epochs,
        "fixed_vis_data": [x.cpu() if isinstance(x, torch.Tensor) else x for x in fixed_vis_data],
        "train_history": [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in d.items()} for d in train_history],
        "validation_history": [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in d.items()} for d in validation_history],
        "num_epochs": num_epochs,
        "seed": seed,
    }, analysis_data_path)
    print(f"[Analysis] Saved analysis data to {analysis_data_path}")

    # ===== NEW: Plot Gradient Dynamics (来自 exp3_optm_dynamic.py) =====
    if len(angle_hist) > 0:
        plot_gradient_dynamics(
            angle_hist, 
            grad_ratio_hist, 
            gradient_record_epochs, 
            ckpt_dir, 
            seed
        )
    
    # ===== NEW: Plot True Loss Landscape =====
    total_landscape_data = plot_true_loss_landscape(
        policy, 
        fixed_vis_data, 
        traj_epochs, 
        traj_vecs, 
        ckpt_dir, 
        seed
    )
    
    # ===== NEW: Plot KL Loss Landscape =====
    kl_landscape_data = plot_kl_loss_landscape(
        policy, 
        fixed_vis_data, 
        traj_epochs, 
        traj_vecs, 
        ckpt_dir, 
        seed
    )

    # ===== NEW: Plot 3D True Loss Landscape =====
    plot_true_3d_loss_landscape(
        policy, 
        fixed_vis_data, 
        traj_epochs, 
        traj_vecs, 
        ckpt_dir, 
        seed
    )

    # ===== NEW: Zoomed-in Loss Landscape (2D Total + 2D KL + 3D Total) =====
    plot_zoomed_loss_landscape(
        total_landscape_data, traj_epochs, ckpt_dir, seed
    )
    plot_zoomed_kl_loss_landscape(
        kl_landscape_data, traj_epochs, ckpt_dir, seed
    )
    plot_zoomed_3d_loss_landscape(
        policy, fixed_vis_data, traj_epochs, traj_vecs, ckpt_dir, seed
    )

    # ===== NEW: Hessian Eigenvalue Spectrum & Condition Number =====
    compute_and_plot_hessian_analysis(
        policy, fixed_vis_data, ckpt_dir, seed
    )

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    all_loss_data = {}  # collect for npz export
    for key in train_history[0]:
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        train_epochs = np.linspace(0, num_epochs - 1, len(train_history))
        val_epochs = np.linspace(0, num_epochs - 1, len(validation_history))

        # Store for npz
        all_loss_data[f"train_{key}"] = np.array(train_values)
        all_loss_data[f"val_{key}"] = np.array(val_values)
        all_loss_data[f"train_{key}_epochs"] = train_epochs
        all_loss_data[f"val_{key}_epochs"] = val_epochs

        plt.figure(figsize=(8, 6))
        plt.plot(train_epochs, train_values, color='#2B8CBE', linewidth=1.5, label='Train')
        plt.plot(val_epochs, val_values, color='#FF6600', linewidth=1.5, label='Val')
        plt.grid(True, alpha=0.3)
        ax = plt.gca()
        ax.tick_params(axis='both', labelsize=10)
        plt.legend(fontsize=11, framealpha=0.9)
        plt.tight_layout()
        for ext in ['png', 'pdf']:
            save_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.{ext}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Save loss curves as npz for offline comparison
    all_loss_data["num_epochs"] = np.array([num_epochs])
    all_loss_data["seed"] = np.array([seed])
    npz_path = os.path.join(ckpt_dir, f"loss_curves_seed_{seed}.npz")
    np.savez(npz_path, **all_loss_data)
    print(f"Saved plots and loss curves to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--batch_size", action="store", type=int, help="batch_size", required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=True)
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT
    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--hidden_dim", action="store", type=int, help="hidden_dim", required=False)
    parser.add_argument("--state_dim", action="store", type=int, help="state dim", required=True)
    parser.add_argument("--save_freq", action="store", type=int, help="save ckpt frequency", required=False, default=6000)
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")

    main(vars(parser.parse_args()))