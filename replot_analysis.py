#!/usr/bin/env python3
"""
Offline re-plotting script for ACT / ACT_PAA analysis.

Usage:
    python replot_analysis.py \
        --ckpt_dir ./act_ckpt/act-beat_block_hammer/demo_clean-100 \
        --seed 0 \
        --policy_class ACT \
        --task_name sim-beat_block_hammer-demo_clean-100 \
        --kl_weight 10 --chunk_size 50 --hidden_dim 512 \
        --dim_feedforward 3200 --lr 1e-5 --state_dim 14

This script loads:
  1. analysis_data_seed_{seed}.pt  (saved by training)
  2. policy_last.ckpt              (model weights)
Then re-runs all visualization functions (landscape, 3D, Hessian, etc.)
without re-training.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")

os.environ["MUJOCO_GL"] = "egl"

from imitate_episodes import (
    _flatten_params,
    _assign_flat_to_params,
    plot_gradient_dynamics,
    plot_true_loss_landscape,
    plot_kl_loss_landscape,
    plot_true_3d_loss_landscape,
    plot_zoomed_loss_landscape,
    plot_zoomed_kl_loss_landscape,
    plot_zoomed_3d_loss_landscape,
    compute_and_plot_hessian_analysis,
    plot_history,
    make_policy,
    forward_pass,
)
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Offline re-plot ACT analysis")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt_name", type=str, default="policy_last.ckpt",
                        help="Checkpoint file to load (default: policy_last.ckpt)")
    
    # Policy construction args (must match training)
    parser.add_argument("--policy_class", type=str, default="ACT")
    parser.add_argument("--task_name", type=str, default=None,
                        help="Task name, used to get camera_names from config. "
                             "If not provided, defaults to ['head_camera']")
    parser.add_argument("--kl_weight", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--state_dim", type=int, default=14)
    
    # Plot selection
    parser.add_argument("--skip_landscape", action="store_true", help="Skip loss landscape plots")
    parser.add_argument("--skip_hessian", action="store_true", help="Skip Hessian analysis")
    parser.add_argument("--skip_gradient", action="store_true", help="Skip gradient dynamics plots")
    parser.add_argument("--skip_history", action="store_true", help="Skip train/val history plots")
    parser.add_argument("--only_hessian", action="store_true", help="Only run Hessian analysis")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as ckpt_dir)")
    
    args = parser.parse_args()
    
    ckpt_dir = args.ckpt_dir
    seed = args.seed
    output_dir = args.output_dir or ckpt_dir
    os.makedirs(output_dir, exist_ok=True)
    
    set_seed(1)
    
    # ===== Load analysis data =====
    analysis_path = os.path.join(ckpt_dir, f"analysis_data_seed_{seed}.pt")
    if not os.path.exists(analysis_path):
        print(f"ERROR: Analysis data not found at {analysis_path}")
        print("This file is saved during training. Please re-train with the updated code first.")
        sys.exit(1)
    
    print(f"Loading analysis data from {analysis_path} ...")
    data = torch.load(analysis_path, map_location="cpu")
    
    traj_epochs = data["traj_epochs"]
    traj_vecs = data["traj_vecs"]
    angle_hist = data["angle_hist"]
    grad_ratio_hist = data["grad_ratio_hist"]
    gradient_record_epochs = data["gradient_record_epochs"]
    fixed_vis_data = data["fixed_vis_data"]
    train_history = data.get("train_history", [])
    validation_history = data.get("validation_history", [])
    num_epochs = data.get("num_epochs", len(train_history))
    
    print(f"  traj_epochs: {len(traj_epochs)} snapshots")
    print(f"  angle_hist: {len(angle_hist)} records")
    print(f"  num_epochs: {num_epochs}")
    
    # ===== Reconstruct policy =====
    if args.task_name:
        is_sim = args.task_name[:4] == "sim-"
        if is_sim:
            from constants import SIM_TASK_CONFIGS
            task_config = SIM_TASK_CONFIGS[args.task_name]
            camera_names = task_config["camera_names"]
        else:
            camera_names = ["head_camera"]
    else:
        camera_names = ["head_camera"]
    
    policy_config = {
        "lr": args.lr,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": camera_names,
    }
    
    ckpt_path = os.path.join(ckpt_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)
    
    print(f"Loading policy from {ckpt_path} ...")
    policy = make_policy(args.policy_class, policy_config)
    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    policy.cuda()
    policy.eval()
    
    # Move vis_data to GPU
    fixed_vis_data = tuple(
        x.cuda() if isinstance(x, torch.Tensor) else x for x in fixed_vis_data
    )
    
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # ===== 1. Train/val history =====
    if not args.skip_history and not args.only_hessian and len(train_history) > 0:
        print("\n[1/6] Plotting training history ...")
        plot_history(train_history, validation_history, num_epochs, output_dir, seed)
    
    # ===== 2. Gradient dynamics =====
    if not args.skip_gradient and not args.only_hessian and len(angle_hist) > 0:
        print("\n[2/6] Plotting gradient dynamics ...")
        plot_gradient_dynamics(angle_hist, grad_ratio_hist, gradient_record_epochs, output_dir, seed)
    
    # ===== 3. Loss landscape 2D =====
    total_landscape_data = None
    kl_landscape_data = None
    if not args.skip_landscape and not args.only_hessian:
        print("\n[3/6] Plotting total loss landscape 2D ...")
        total_landscape_data = plot_true_loss_landscape(
            policy, fixed_vis_data, traj_epochs, traj_vecs, output_dir, seed
        )
        
        print("\n[4/6] Plotting KL loss landscape 2D ...")
        kl_landscape_data = plot_kl_loss_landscape(
            policy, fixed_vis_data, traj_epochs, traj_vecs, output_dir, seed
        )
        
        print("\n[5/6] Plotting 3D loss landscape ...")
        plot_true_3d_loss_landscape(
            policy, fixed_vis_data, traj_epochs, traj_vecs, output_dir, seed
        )
        
        # Zoomed versions
        if total_landscape_data is not None:
            plot_zoomed_loss_landscape(total_landscape_data, traj_epochs, output_dir, seed)
        if kl_landscape_data is not None:
            plot_zoomed_kl_loss_landscape(kl_landscape_data, traj_epochs, output_dir, seed)
        plot_zoomed_3d_loss_landscape(
            policy, fixed_vis_data, traj_epochs, traj_vecs, output_dir, seed
        )
    
    # ===== 6. Hessian analysis =====
    if not args.skip_hessian:
        print("\n[6/6] Running Hessian eigenvalue analysis ...")
        compute_and_plot_hessian_analysis(policy, fixed_vis_data, output_dir, seed)
    
    print("\n" + "=" * 60)
    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
