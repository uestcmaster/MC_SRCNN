# -*- coding: utf-8 -*-
"""
步骤二：深度统计分析
- 任务4：分布形态检验
    选取 3 个典型区域：平坦背景(低频) / 清晰边缘(中频) / 复杂纹理(高频)
    对每个区域内所有像素值的 T 次推理结果做直方图 + 高斯拟合，
    并用 scipy.stats.kstest 计算 K-S 检验 P 值。
- 任务5：不确定性有效性验证
    全图平方误差 E(x,y) = (μ(x,y) - Y_GT(x,y))²
    散点图 (σ²(x,y), E(x,y))
    Pearson r：scipy.stats.pearsonr
    判断 r > 0.6 是否成立
- 任务6：收敛性分析
    对 T ∈ {5,10,20,30,50} 计算 σ_T² = (1/HW) Σ σ²(x,y)
    以 T=50 为基准计算相对误差，找最小 T_opt 使 误差 < 5%

依赖：torch / torchvision / numpy / scipy / matplotlib / pillow
运行：python step2_statistical_analysis.py
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.stats import norm, kstest, pearsonr

from model import SRCNN_MC
from dataprocess import load_img, CROP_SIZE


# ===================== 配置 =====================
HERE = os.path.dirname(os.path.abspath(__file__))
IMG_PATH    = os.path.join(HERE, "data", "Set5", "baby.png")
MODEL_PATH  = os.path.join(HERE, "dp20_model", "latest_checkpoint.pth")
SAVE_DIR    = os.path.join(HERE, "results", "step2_stat_analysis")

ZOOM_FACTOR = 4
T_FULL      = 50
T_LIST      = [5, 10, 20, 30, 50]
DROPOUT_P   = 0.20
ROI_SIZE    = 32                       # 每个典型区域大小 32x32
RNG_SEED    = 42

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

print(f"[Info] device     = {device}")
print(f"[Info] image      = {IMG_PATH}")
print(f"[Info] T_FULL     = {T_FULL}, dropout_p = {DROPOUT_P}")
print(f"[Info] T_LIST     = {T_LIST}")
print(f"[Info] save_dir   = {SAVE_DIR}")


# ===================== 数据预处理 =====================
crop_size = CROP_SIZE - (CROP_SIZE % ZOOM_FACTOR)  # 256
input_transform = transforms.Compose([
    transforms.CenterCrop(crop_size),
    transforms.Resize(crop_size // ZOOM_FACTOR),
    transforms.Resize(crop_size, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
])

y_pil    = load_img(IMG_PATH)
lr_tensor = input_transform(y_pil).unsqueeze(0).to(device)        # [1,1,256,256]
gt_tensor = target_transform(y_pil)                               # [1,256,256]
Y_GT      = gt_tensor[0].numpy()                                  # GT Y 通道, (256,256)
H, W      = Y_GT.shape
print(f"[Info] H={H}, W={W}")


# ===================== 加载模型 =====================
model = SRCNN_MC(dropout_p=DROPOUT_P).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.eval()  # forward 内 dropout training=True，eval 下仍生效


# ===================== 蒙特卡洛采样：T_FULL 次 =====================
def mc_sample(T):
    Y = np.zeros((H, W, T), dtype=np.float32)
    with torch.no_grad():
        for i in range(T):
            out = model(lr_tensor)
            Y[:, :, i] = out[0, 0].cpu().numpy()
    return Y

print("\n[Step] MC 采样 T=50 …")
Y = mc_sample(T_FULL)                                # (H,W,T)
mu      = Y.mean(axis=2)                             # μ(x,y)
sigma2  = Y.var(axis=2, ddof=1)                      # σ²(x,y)
print(f"[Info] Y.shape={Y.shape}, μ.mean={mu.mean():.4f}, σ².mean={sigma2.mean():.6f}")


# ===================== 任务4：3 个典型区域 + 分布形态检验 =====================
print("\n[Step] 任务4：分布形态检验（3 个典型区域）…")

# 用 GT 做 5x5 局部方差 → 排序选 3 类区域
def local_variance(img, k=5):
    pad = k // 2
    img_p = np.pad(img, pad, mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    # 简单卷积均值/平方均值
    for dy in range(k):
        for dx in range(k):
            out += img_p[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    mean = out / (k * k)
    out2 = np.zeros_like(img, dtype=np.float32)
    for dy in range(k):
        for dx in range(k):
            out2 += img_p[dy:dy + img.shape[0], dx:dx + img.shape[1]] ** 2
    mean_sq = out2 / (k * k)
    return mean_sq - mean ** 2

lvar = local_variance(Y_GT.astype(np.float32))

# 遍历所有 ROI 中心点，计算 ROI 平均局部方差，再按分位选 3 个 ROI
rs = ROI_SIZE // 2
ys = np.arange(rs, H - rs, ROI_SIZE // 2)
xs = np.arange(rs, W - rs, ROI_SIZE // 2)
roi_scores = []
for y in ys:
    for x in xs:
        roi_lvar = lvar[y - rs:y + rs, x - rs:x + rs].mean()
        roi_scores.append((roi_lvar, y, x))
roi_scores.sort(key=lambda t: t[0])
flat_score, fy, fx = roi_scores[len(roi_scores) // 20]                       # 最低 5%
edge_score, ey, ex = roi_scores[len(roi_scores) // 2]                        # 中位
tex_score,  ty, tx = roi_scores[-(len(roi_scores) // 20 + 1)]                # 最高 5%

regions = {
    "flat (low-freq)":   (fy, fx, flat_score),
    "edge (mid-freq)":   (ey, ex, edge_score),
    "texture (high-freq)": (ty, tx, tex_score),
}
for name, (y, x, sc) in regions.items():
    print(f"  - {name:22s} center=({y:3d},{x:3d})  ROI local_var={sc:.5f}")

# ROI 可视化（在 GT 上画框）
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(Y_GT, cmap="gray")
colors = {"flat (low-freq)": "lime", "edge (mid-freq)": "yellow", "texture (high-freq)": "red"}
for name, (y, x, _) in regions.items():
    rect = Rectangle((x - rs, y - rs), ROI_SIZE, ROI_SIZE,
                     edgecolor=colors[name], facecolor="none", linewidth=2, label=name)
    ax.add_patch(rect)
ax.set_title("Three typical regions on GT")
ax.legend(loc="lower right", fontsize=8)
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "task4_regions_on_GT.png"), dpi=150)
plt.close()

# 对每个 ROI：把 ROI×T 的所有像素当作一个随机变量，画直方图 + 高斯拟合 + K-S
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ks_table = {}
for ax, (name, (y, x, _)) in zip(axes, regions.items()):
    samples = Y[y - rs:y + rs, x - rs:x + rs, :].reshape(-1)   # 长度 = ROI*T
    mu_s, sigma_s = norm.fit(samples)
    ks_stat, p_val = kstest(samples, "norm", args=(mu_s, sigma_s))
    skew = stats.skew(samples)
    kurt = stats.kurtosis(samples)
    ks_table[name] = (mu_s, sigma_s, ks_stat, p_val, skew, kurt, samples.size)

    ax.hist(samples, bins=60, density=True, alpha=0.65, color="steelblue",
            label="histogram")
    xx = np.linspace(samples.min(), samples.max(), 200)
    ax.plot(xx, norm.pdf(xx, mu_s, sigma_s), "r-", lw=2,
            label=f"Gaussian fit\nμ={mu_s:.3f}, σ={sigma_s:.3f}")
    ax.set_title(f"{name}\nKS p={p_val:.3g}  skew={skew:.2f}  kurt={kurt:.2f}",
                 fontsize=10)
    ax.set_xlabel("pixel value")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "task4_distribution_fit.png"), dpi=150)
plt.close()

print("  [K-S 检验结果]  (P<0.05 ⇒ 拒绝高斯)")
for name, (mu_s, sigma_s, kst, p, sk, ku, n) in ks_table.items():
    verdict = "近似高斯" if p > 0.05 else "拒绝高斯"
    print(f"    {name:22s}  N={n:5d}  μ={mu_s:.3f}  σ={sigma_s:.3f}  "
          f"KS_stat={kst:.3f}  P={p:.3g}  → {verdict}")


# ===================== 任务5：不确定性有效性验证 =====================
print("\n[Step] 任务5：不确定性有效性验证 …")
E = (mu - Y_GT) ** 2                            # 全图平方误差
sigma2_flat = sigma2.reshape(-1)
E_flat      = E.reshape(-1)
r, p_r = pearsonr(sigma2_flat, E_flat)
print(f"  Pearson r = {r:.4f}, p = {p_r:.3e}")
print(f"  → r > 0.6 ?  {'是' if r > 0.6 else '否'}")

# 散点图（抽样 10000 点防止点数太密）
idx = np.random.choice(sigma2_flat.size, size=min(10000, sigma2_flat.size),
                       replace=False)
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(sigma2_flat[idx], E_flat[idx], s=2, alpha=0.4, color="steelblue")
# 拟合一条直线
z = np.polyfit(sigma2_flat[idx], E_flat[idx], 1)
xx = np.linspace(sigma2_flat.min(), sigma2_flat.max(), 100)
ax.plot(xx, np.polyval(z, xx), "r--", lw=2, label=f"linear fit: y={z[0]:.2f}x+{z[1]:.3f}")
ax.set_xlabel(r"variance $\sigma^2(x,y)$")
ax.set_ylabel(r"squared error $E(x,y)=(\mu - Y_{GT})^2$")
ax.set_title(f"Uncertainty vs Squared Error (Pearson r={r:.3f}, p={p_r:.2e})")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "task5_scatter_var_vs_squared_err.png"), dpi=150)
plt.close()

# 误差图可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
im0 = axes[0].imshow(mu, cmap="gray");      axes[0].set_title("μ(x,y) (mean SR)");    axes[0].axis("off")
im1 = axes[1].imshow(sigma2, cmap="jet");   axes[1].set_title(r"$\sigma^2(x,y)$");    axes[1].axis("off"); plt.colorbar(im1, ax=axes[1], fraction=0.046)
im2 = axes[2].imshow(E, cmap="hot");        axes[2].set_title(r"$E(x,y)=(\mu-Y_{GT})^2$"); axes[2].axis("off"); plt.colorbar(im2, ax=axes[2], fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "task5_maps_squared_error.png"), dpi=150)
plt.close()


# ===================== 任务6：收敛性分析 =====================
print("\n[Step] 任务6：收敛性分析（T=5,10,20,30,50）…")

# 复用同一份 Y（前 T 个样本即可，避免重复 inference）
sigma2_T_means = {}
for T in T_LIST:
    YT = Y[:, :, :T]
    sigma2_T = YT.var(axis=2, ddof=1) if T > 1 else np.zeros((H, W), dtype=np.float32)
    sigma2_T_means[T] = float(sigma2_T.mean())

baseline = sigma2_T_means[T_FULL]
rel_errors = {T: abs(sigma2_T_means[T] - baseline) / baseline * 100 for T in T_LIST}

print(f"  baseline σ²(T=50) = {baseline:.6f}")
T_opt = None
for T in T_LIST:
    flag = " <- T_opt" if (T_opt is None and rel_errors[T] < 5.0) else ""
    if T_opt is None and rel_errors[T] < 5.0:
        T_opt = T
    print(f"    T={T:2d}  σ²_T={sigma2_T_means[T]:.6f}  rel_err={rel_errors[T]:6.2f}%{flag}")
print(f"  最小 T_opt（误差 < 5%）: {T_opt}")

# 收敛曲线
fig, ax = plt.subplots(figsize=(7, 5))
Ts = T_LIST
errs = [rel_errors[t] for t in Ts]
ax.plot(Ts, errs, "o-", lw=2, color="steelblue", label="relative error")
ax.axhline(5.0, color="red", ls="--", label="5% threshold")
if T_opt:
    ax.axvline(T_opt, color="green", ls=":", label=f"T_opt = {T_opt}")
ax.set_xlabel("T (number of MC samples)")
ax.set_ylabel("relative error of σ² mean (%)")
ax.set_title("Convergence: σ²(T) relative error vs T (baseline T=50)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "task6_convergence.png"), dpi=150)
plt.close()


# ===================== 保存数据 + 报告 =====================
np.savez_compressed(
    os.path.join(SAVE_DIR, "step2_results.npz"),
    Y=Y, mu=mu, sigma2=sigma2, E=E, Y_GT=Y_GT,
    regions={k: (v[0], v[1]) for k, v in regions.items()},
    ks_table=ks_table, sigma2_T_means=sigma2_T_means, rel_errors=rel_errors,
    T_opt=T_opt if T_opt is not None else -1,
    pearson_r=r, pearson_p=p_r,
)

report_path = os.path.join(SAVE_DIR, "step2_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# 步骤二·深度统计分析 实验结果\n\n")
    f.write(f"- 测试图像：`{os.path.relpath(IMG_PATH, HERE)}`\n")
    f.write(f"- T_FULL = {T_FULL}，Dropout p = {DROPOUT_P}\n")
    f.write(f"- ROI 大小：{ROI_SIZE}×{ROI_SIZE}\n\n")

    f.write("## 任务4：分布形态检验\n\n")
    f.write("| 区域 | 中心(y,x) | N | μ | σ | KS_stat | P值 | 偏度 | 峰度 | 结论 |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|\n")
    for name, (y, x, _) in regions.items():
        mu_s, sigma_s, kst, p, sk, ku, n = ks_table[name]
        verdict = "近似高斯" if p > 0.05 else "拒绝高斯"
        f.write(f"| {name} | ({y},{x}) | {n} | {mu_s:.3f} | {sigma_s:.3f} | "
                f"{kst:.3f} | {p:.3g} | {sk:.2f} | {ku:.2f} | {verdict} |\n")
    f.write("\n配图：`task4_regions_on_GT.png`、`task4_distribution_fit.png`\n\n")

    f.write("## 任务5：不确定性有效性验证\n\n")
    f.write("- 误差定义：`E(x,y) = (μ(x,y) - Y_GT(x,y))^2`\n")
    f.write(f"- 全图 Pearson r = **{r:.4f}**，p = {p_r:.3e}\n")
    f.write(f"- 判定：r > 0.6 ⇒ **{'成立，模型不确定性估计有效' if r > 0.6 else '不成立'}**\n")
    if r <= 0.6:
        f.write("- 可能原因：\n"
                "  1) Dropout 概率较低 (p=0.1)，模型采样多样性有限；\n"
                "  2) 图像大部分区域是平坦背景，σ² 与 E 都很小，整体相关性被稀释；\n"
                "  3) MC-Dropout 主要刻画**模型/认知不确定性**，而 (μ-GT)^2 还包含退化误差、模型偏差和重建误差；\n"
                "  4) 仅在 conv2-conv3 之间一处 Dropout，扰动信息传播不充分。\n")
    f.write("\n配图：`task5_maps_squared_error.png`、`task5_scatter_var_vs_squared_err.png`\n\n")

    f.write("## 任务6：收敛性分析\n\n")
    f.write("| T | σ²_T mean | relative error |\n|---|---|---|\n")
    for T in T_LIST:
        f.write(f"| {T} | {sigma2_T_means[T]:.6f} | {rel_errors[T]:.2f}% |\n")
    f.write(f"\n- 基准 σ²(T=50) = {baseline:.6f}\n")
    f.write(f"- **最小 T_opt（误差 < 5%）= {T_opt}**\n\n")
    f.write("配图：`task6_convergence.png`\n")

print(f"\n========== 步骤二完成 ==========")
print(f"输出目录: {SAVE_DIR}")
for fn in ["task4_regions_on_GT.png", "task4_distribution_fit.png",
           "task5_maps_squared_error.png", "task5_scatter_var_vs_squared_err.png",
           "task6_convergence.png", "step2_results.npz", "step2_report.md"]:
    print(f"  - {fn}")