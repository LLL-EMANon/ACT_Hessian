import os

from additional_expirement.piratenet import PirateNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'  # 让数学符号风格与 Times New Roman 匹配


# ----------------------------
# 1) MLP for u(x,y)
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=80, depth=5, out_dim=1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X):
        return self.net(X)


# ----------------------------
# 2) Dirichlet boundary g(x,y)
# ----------------------------
A, B, C, D = 20.0, 15.0, 8.0, 6.0


def g_fun(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return (A * torch.sin(2.0 * math.pi * x)
            + B * torch.cos(3.0 * math.pi * y)
            + C * x + D * y)


# ----------------------------
# 3) Laplacian and RHS f(x,y)
# ----------------------------
def laplacian(u, X):
    grads = torch.autograd.grad(
        u, X,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]

    u_xx = torch.autograd.grad(
        u_x, X,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        u_y, X,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True, retain_graph=True
    )[0][:, 1:2]

    return u_xx + u_yy


def f_rhs(X):
    x = X[:, 0:1]
    y = X[:, 1:2]

    lap_g = -((2.0 * math.pi) ** 2) * A * torch.sin(2.0 * math.pi * x) \
            - ((3.0 * math.pi) ** 2) * B * torch.cos(3.0 * math.pi * y)
    lap_extra = -2.0 * (math.pi ** 2) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
    return lap_g + lap_extra


# ----------------------------
# 4) Sampling points
# ----------------------------
def sample_interior(n, device):
    X = torch.rand(n, 2, device=device)
    X.requires_grad_(True)
    return X


def sample_boundary(n_each_edge, device):
    t = torch.rand(n_each_edge, 1, device=device)
    X_left = torch.cat([torch.zeros_like(t), t], dim=1)
    X_right = torch.cat([torch.ones_like(t), t], dim=1)
    X_bottom = torch.cat([t, torch.zeros_like(t)], dim=1)
    X_top = torch.cat([t, torch.ones_like(t)], dim=1)
    Xb = torch.cat([X_left, X_right, X_bottom, X_top], dim=0)
    return Xb


# ----------------------------
# 5) grad flatten helper
# ----------------------------
def _flatten_grads(grads, params):
    flats = []
    for g, p in zip(grads, params):
        if g is None:
            flats.append(torch.zeros_like(p).reshape(-1))
        else:
            flats.append(g.reshape(-1))
    return torch.cat(flats)


# ---- trajectory helpers (NEW) ----
def _flatten_params(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def _assign_flat_to_params(flat, params):
    # flat: 1D tensor on same device as params
    idx = 0
    for p in params:
        num = p.numel()
        with torch.no_grad():
            p.copy_(flat[idx:idx + num].view_as(p))
        idx += num


# ----------------------------
# 6) Train + log + plot
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# model = PirateNet().to(device)
model = MLP().to(device)

N_f = 8000
N_b_edge = 600
epochs = 4101
lr = 1e-3
mid_epoch = 2000

w_pde = 1.0
w_bc = 1.0

opt = torch.optim.Adam(model.parameters(), lr=lr)

# logs
loss_pde_hist = []
loss_bc_hist = []
angle_hist = []  # angle in degrees
grad_ratio_hist = []  # ||∇Lpde|| / ||∇Lbc||

params = [p for p in model.parameters() if p.requires_grad]

traj_stride = 1  # 记录轨迹的步长（越小越密但更占内存）；最小改动起见给个温和默认
traj_epochs = []
traj_vecs = []

# epoch0 snapshot (before training)
w0 = _flatten_params(params).cpu()
traj_epochs.append(0)
traj_vecs.append(w0)

for ep in range(1, epochs + 1):
    Xf = sample_interior(N_f, device)
    Xb = sample_boundary(N_b_edge, device)

    # PDE loss
    uf = model(Xf)
    r = laplacian(uf, Xf) - f_rhs(Xf)
    loss_pde = torch.mean(r ** 2)

    # BC loss
    ub = model(Xb)
    gb = g_fun(Xb)
    loss_bc = torch.mean((ub - gb) ** 2)

    loss = w_pde * loss_pde + w_bc * loss_bc

    # --- compute PDE/BC gradient angle + norm ratio ---
    grads_pde = torch.autograd.grad(loss_pde, params, retain_graph=True, create_graph=False, allow_unused=True)
    g_pde = _flatten_grads(grads_pde, params).detach()

    grads_bc = torch.autograd.grad(loss_bc, params, retain_graph=True, create_graph=False, allow_unused=True)
    g_bc = _flatten_grads(grads_bc, params).detach()

    dot = torch.dot(g_pde, g_bc)
    ng1 = g_pde.norm() + 1e-12
    ng2 = g_bc.norm() + 1e-12
    cosang = torch.clamp(dot / (ng1 * ng2), -1.0, 1.0)
    angle_deg = torch.acos(cosang) * (180.0 / math.pi)

    grad_ratio = (ng1 / ng2).item()

    # --- optimize ---
    opt.zero_grad()
    loss.backward()
    opt.step()

    # ---- trajectory logging (NEW) ----
    if (ep % traj_stride == 0) or (ep in [mid_epoch, epochs]):
        w = _flatten_params(params).cpu()
        traj_epochs.append(ep)
        traj_vecs.append(w)

    # logs
    loss_pde_hist.append(loss_pde.item())
    loss_bc_hist.append(loss_bc.item())
    angle_hist.append(cosang.item())
    grad_ratio_hist.append(grad_ratio)

    if ep % 10 == 0 or ep == 1:
        msg = (f"Epoch {ep:5d} | pde={loss_pde.item():.3e} | bc={loss_bc.item():.3e} "
               f"| angle={angle_deg.item():6.2f} deg | grad_ratio={grad_ratio:.3e}")
        print(msg)

# ----------------------------
# 7) Plot curves
# ----------------------------
epochs_arr = torch.arange(1, epochs + 1).cpu().numpy()

import matplotlib.ticker as ticker

plt.figure(figsize=(6, 5))
plt.plot(epochs_arr, loss_pde_hist)
plt.yscale("log")
plt.xlabel("epoch", fontsize=18)
plt.ylabel("L_pde (log)", fontsize=18)
plt.grid(True)

ax = plt.gca()
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

plt.subplots_adjust(bottom=0.13)
plt.savefig("loss_pde.pdf", dpi=300)

plt.figure(figsize=(6, 5))
plt.plot(epochs_arr, loss_bc_hist)
plt.yscale("log")
plt.xlabel("epoch", fontsize=18)
plt.ylabel("L_bc (log)", fontsize=18)
plt.grid(True)

ax = plt.gca()
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

plt.subplots_adjust(bottom=0.13)
plt.savefig("loss_bc.pdf", dpi=300)

plt.figure(figsize=(6, 5))
plt.plot(epochs_arr, angle_hist)
plt.xlabel("epoch", fontsize=18)
plt.ylabel(r"ρ_g", fontsize=18, labelpad=1)
plt.grid(True)

ax = plt.gca()
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

plt.subplots_adjust(bottom=0.13)
plt.savefig("angle.pdf", dpi=300)

plt.figure(figsize=(6, 5))
plt.plot(epochs_arr, grad_ratio_hist)
plt.yscale("log")
plt.xlabel("epoch", fontsize=18)
plt.ylabel(r"cos(ϕ) (log)", fontsize=18)
plt.grid(True)

ax = plt.gca()
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

plt.subplots_adjust(bottom=0.13)
plt.savefig("grad_ratio.pdf", dpi=300)

# ----------------------------
# 7.5) Plot optimization trajectory on plane spanned by epoch0/3000/10000 (NEW)
# ----------------------------
# fetch required anchor points
# (we already have epoch0; ensure epoch3000 & epoch10000 exist in traj logs)
epoch_to_vec = {e: v for e, v in zip(traj_epochs, traj_vecs)}
if mid_epoch not in epoch_to_vec:
    epoch_to_vec[mid_epoch] = traj_vecs[-1]
if epochs not in epoch_to_vec:
    epoch_to_vec[epochs] = traj_vecs[-1]  # fallback (shouldn't happen)

p0 = epoch_to_vec[0]
p1 = epoch_to_vec[mid_epoch]
p2 = epoch_to_vec[epochs]

u = (p1 - p0)
v = (p2 - p0)

# Orthonormal basis (e1, e2) for the plane
e1 = u / (u.norm() + 1e-12)
v_orth = v - torch.dot(v, e1) * e1
e2 = v_orth / (v_orth.norm() + 1e-12)

# project all recorded points
XY = []
for w in traj_vecs:
    d = w - p0
    x = torch.dot(d, e1).item()
    y = torch.dot(d, e2).item()
    XY.append((x, y))

xs = [t[0] for t in XY]
ys = [t[1] for t in XY]

# ===== 只取 ep 4000~4100 的局部轨迹 =====
ep_lo, ep_hi = 3800, 4000

# traj_epochs 与 traj_vecs 一一对应，所以用它来筛选局部点
mask = [(e >= ep_lo) and (e <= ep_hi) for e in traj_epochs]

xs_loc = [x for x, m in zip(xs, mask) if m]
ys_loc = [y for y, m in zip(ys, mask) if m]
traj_epochs_loc = [e for e, m in zip(traj_epochs, mask) if m]


# also project the three anchor points (for marking)
def proj(w):
    d = w - p0
    return torch.dot(d, e1).item(), torch.dot(d, e2).item()


x0, y0 = proj(p0)
x1, y1 = proj(p1)
x2, y2 = proj(p2)

# ----------------------------
# 7.5) Plot PDE residual loss landscape + trajectory on the same plane (NEW)
# ----------------------------
# 1) 先固定一批 interior 点用于景观评估（避免每个网格点重采样导致景观噪声）
N_f_vis = 2000  # 可调：越大越准但越慢
grid_res = 35  # 可调：网格分辨率，越大越细但越慢
landscape_margin = 0.15  # 可调：在轨迹包围盒外再扩一点

Xf_vis = sample_interior(N_f_vis, device)  # requires_grad_(True) inside

# 2) 定义网格范围：根据轨迹投影坐标(xs, ys)的范围来定
# xmin, xmax = min(xs), max(xs)
# ymin, ymax = min(ys), max(ys)


xmin, xmax = min(xs_loc), max(xs_loc)
ymin, ymax = min(ys_loc), max(ys_loc)

dx = xmax - xmin
dy = ymax - ymin
xmin -= landscape_margin * (dx + 1e-12)
xmax += landscape_margin * (dx + 1e-12)
ymin -= landscape_margin * (dy + 1e-12)
ymax += landscape_margin * (dy + 1e-12)

x_lin = torch.linspace(xmin, xmax, grid_res, device=device)
y_lin = torch.linspace(ymin, ymax, grid_res, device=device)

# 3) 保存当前模型参数，评估完景观后再恢复（避免影响后续）
w_backup = _flatten_params(params).detach().clone().to(device)

# 4) 逐网格点计算 L_pde（只算 PDE 残差项）
Z = torch.empty((grid_res, grid_res), device=device)

model.train(False)  # 只影响 Dropout/BN（你这里没有也没关系）
for i in range(grid_res):
    for j in range(grid_res):
        # w = p0 + x*e1 + y*e2
        w_ij = (p0.to(device) + x_lin[i] * e1.to(device) + y_lin[j] * e2.to(device))
        _assign_flat_to_params(w_ij, params)

        uf = model(Xf_vis)
        r = laplacian(uf, Xf_vis) - f_rhs(Xf_vis)
        loss_pde_ij = torch.mean(r ** 2)
        Z[j, i] = loss_pde_ij  # 注意：imshow/contour的行列方向

# 恢复参数
_assign_flat_to_params(w_backup, params)

# 5) 画“景观（等高线）+ 轨迹”
Xg, Yg = torch.meshgrid(x_lin.detach().cpu(), y_lin.detach().cpu(), indexing="xy")
Zcpu = Z.detach().cpu()

plt.figure(figsize=(6, 5))

# 建议用对数尺度画等高线更清晰（PDE loss 通常跨度很大）
Zplot = torch.log10(Zcpu + 1e-16)

# 等高线填充（背景）
cs = plt.contourf(Xg.numpy(), Yg.numpy(), Zplot.numpy(), levels=25)

# 轨迹线（叠加在上面）
# plt.plot(xs, ys, linewidth=1.3)
plt.plot(xs_loc, ys_loc, linewidth=1.3)
# plt.scatter([x0, x1, x2], [y0, y1, y2], s=35)
# plt.text(x0, y0, "  ep0", fontsize=12)
# plt.text(x1, y1, f"  ep{mid_epoch}", fontsize=12)
# plt.text(x2, y2, f"  ep{epochs}", fontsize=12)


x_lo, y_lo = proj(epoch_to_vec[ep_lo])
x_hi, y_hi = proj(epoch_to_vec[ep_hi])

plt.scatter([x_lo, x_hi], [y_lo, y_hi], s=35)
plt.text(x_lo, y_lo, f"  ep{ep_lo}", fontsize=12)
plt.text(x_hi, y_hi, f"  ep{ep_hi}", fontsize=12)


plt.xlabel("proj axis 1", fontsize=18)
plt.ylabel("proj axis 2", fontsize=18)
plt.grid(True)

ax = plt.gca()
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# 色条：标明 log10(L_pde)
cbar = plt.colorbar(cs)
cbar.set_label(r"$\log_{10}(\mathcal{L}_{\text{res}})$", fontsize=14)
cbar.ax.tick_params(labelsize=12)

plt.subplots_adjust(bottom=0.13)
plt.savefig("traj_plane_pde_local.pdf", dpi=300)

plt.show()

# ----------------------------
# 8) Optional sanity check (unchanged)
# ----------------------------
with torch.no_grad():
    Xt = torch.rand(5000, 2, device=device)
    x = Xt[:, 0:1]
    y = Xt[:, 1:2]
    u_true = g_fun(Xt) + torch.sin(math.pi * x) * torch.sin(math.pi * y)
    u_pred = model(Xt)
    rel_l2 = torch.linalg.norm(u_pred - u_true) / torch.linalg.norm(u_true)
    print("Relative L2 error (check only):", rel_l2.item())

    Xb = sample_boundary(2000, device)
    gb = g_fun(Xb)
    print("Boundary g(x,y) range approx:", float(gb.min().cpu()), "to", float(gb.max().cpu()))
