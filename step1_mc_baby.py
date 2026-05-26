# -*- coding: utf-8 -*-
"""
步骤一：算法改造 + 蒙特卡洛仿真 + 统计量计算
- 任务1：使用改造后的 SRCNN_MC（在 conv2 与 conv3 之间插入 Dropout，eval 下仍生效）
- 任务2：对同一张低分辨率图像 baby.png 进行 T=50 次前向传播
         存入矩阵 Y ∈ R^{H×W×T}（每次输出图像的像素值）
- 任务3：计算均值图 μ(x,y) = (1/T) Σ Y_i(x,y)        —— 最终超分图
        计算方差图 σ²(x,y) = (1/(T-1)) Σ (Y_i - μ)²  —— 不确定性热力图
运行方式：在该脚本所在目录执行  python step1_mc_baby.py
"""

import os
import sys

# Windows 控制台默认 GBK，无法打印 σ² / μ 等字符，这里强制 UTF-8
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
matplotlib.use("Agg")  # 无显示环境也可保存图像
import matplotlib.pyplot as plt

from model import SRCNN_MC
from dataprocess import load_img, CROP_SIZE
from tensor_to_RGB import tensor_to_rgb_image


# ===================== 配置 =====================
HERE = os.path.dirname(os.path.abspath(__file__))
IMG_PATH    = os.path.join(HERE, "data", "Set5", "baby.png")
MODEL_PATH  = os.path.join(HERE, "pretrain_model", "best_model_SRCNN.pth")
SAVE_DIR    = os.path.join(HERE, "results", "step1_mc_baby")

ZOOM_FACTOR = 4
T           = 50      # 蒙特卡洛采样次数
DROPOUT_P   = 0.10    # Dropout 概率（0.01~0.20）

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"[Info] device = {device}")
print(f"[Info] image  = {IMG_PATH}")
print(f"[Info] model  = {MODEL_PATH}")
print(f"[Info] T      = {T}, dropout_p = {DROPOUT_P}")


# ===================== 数据预处理（与 dataprocess.py 一致） =====================
crop_size = CROP_SIZE - (CROP_SIZE % ZOOM_FACTOR)  # 256
input_transform = transforms.Compose([
    transforms.CenterCrop(crop_size),
    transforms.Resize(crop_size // ZOOM_FACTOR),
    transforms.Resize(crop_size, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
])

y_channel = load_img(IMG_PATH)                  # PIL, mode 'L' (Y 通道)
lr_tensor = input_transform(y_channel).unsqueeze(0).to(device)  # [1,1,H,W]
H, W = lr_tensor.shape[-2], lr_tensor.shape[-1]
print(f"[Info] LR 输入 tensor 形状: {tuple(lr_tensor.shape)}  (H={H}, W={W})")


# ===================== 加载模型 =====================
model = SRCNN_MC(dropout_p=DROPOUT_P).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state, strict=False)
model.eval()  # 注意：forward 内 dropout 显式 training=True，故 eval 下仍生效


# ===================== 任务2：T=50 次前向，存矩阵 Y =====================
Y = np.zeros((H, W, T), dtype=np.float32)  # Y ∈ R^{H×W×T}
with torch.no_grad():
    for i in range(T):
        out = model(lr_tensor)             # [1,1,H,W]
        Y[:, :, i] = out[0, 0].cpu().numpy()
print(f"[Info] 蒙特卡洛矩阵 Y 形状: {Y.shape}")


# ===================== 任务3：统计量 =====================
mu    = Y.mean(axis=2)                              # μ(x,y)
sigma2 = Y.var(axis=2, ddof=1)                      # σ²(x,y), 无偏估计 (T-1)
print(f"[Info] μ:   min={mu.min():.4f},   max={mu.max():.4f},   mean={mu.mean():.4f}")
print(f"[Info] σ²:  min={sigma2.min():.6f}, max={sigma2.max():.6f}, mean={sigma2.mean():.6f}")


# ===================== 保存结果 =====================
# 1) 均值图（最终超分结果）—— 转 RGB 保存
mu_tensor = torch.from_numpy(mu).unsqueeze(0)   # [1,H,W]
mu_rgb = tensor_to_rgb_image(mu_tensor, IMG_PATH, ZOOM_FACTOR)
mu_rgb.save(os.path.join(SAVE_DIR, "mean_SR.png"))

# 2) 均值图（Y 通道灰度版）
mu_gray = np.clip(mu * 255.0, 0, 255).astype(np.uint8)
Image.fromarray(mu_gray, mode="L").save(os.path.join(SAVE_DIR, "mean_Y.png"))

# 3) 方差图（不确定性热力图）—— 用 matplotlib 颜色映射保存
plt.figure(figsize=(6, 5))
plt.imshow(sigma2, cmap="jet")
plt.colorbar(label="variance")
plt.title(f"Uncertainty (variance) heatmap, T={T}, p={DROPOUT_P}")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "variance_heatmap.png"), dpi=150)
plt.close()

# 4) 标准差归一化灰度图（便于直接查看）
std = np.sqrt(sigma2)
std_norm = (std - std.min()) / (std.max() - std.min() + 1e-12)
std_u8 = (std_norm * 255).astype(np.uint8)
Image.fromarray(std_u8, mode="L").save(os.path.join(SAVE_DIR, "std_gray.png"))

# 5) 保存原始矩阵 Y 与 μ / σ²，便于后续步骤使用
np.savez_compressed(
    os.path.join(SAVE_DIR, "mc_results.npz"),
    Y=Y, mu=mu, sigma2=sigma2,
    T=T, dropout_p=DROPOUT_P,
)

print("\n========== 步骤一完成 ==========")
print(f"输出目录: {SAVE_DIR}")
for fn in ["mean_SR.png", "mean_Y.png", "variance_heatmap.png",
           "std_gray.png", "mc_results.npz"]:
    print(f"  - {fn}")
