# -*- coding: utf-8 -*-
import argparse
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import SRCNN, SRCNN_MC
from metrics import psnr, ssim


# ===================== 自定义数据集：训练随机裁剪，验证中心裁剪 =====================
class SRFolderDataset(Dataset):
    """
    用于 SRCNN 训练的数据集。

    输入：
        HR 图像文件夹

    输出：
        input_img:  先从 HR 裁剪 crop_size，再缩小 zoom_factor 倍，再 bicubic 放回 crop_size
        target:     HR 裁剪 patch

    训练模式：
        RandomCrop(crop_size)

    验证模式：
        CenterCrop(crop_size)
    """

    def __init__(self, image_dir, zoom_factor=4, crop_size=256, train=True):
        super().__init__()

        self.image_dir = image_dir
        self.zoom_factor = zoom_factor
        self.crop_size = crop_size - (crop_size % zoom_factor)
        self.train = train

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for name in sorted(files):
                ext = os.path.splitext(name)[1].lower()
                if ext in exts:
                    self.image_paths.append(os.path.join(root, name))

        self.image_paths = sorted(self.image_paths)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found in {image_dir}")

        if train:
            self.crop = transforms.RandomCrop(self.crop_size)
        else:
            self.crop = transforms.CenterCrop(self.crop_size)

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def _load_y_channel(self, path):
        """
        SRCNN 常用 Y 通道训练。
        """
        img = Image.open(path).convert("YCbCr")
        y, _, _ = img.split()
        return y

    def _ensure_large_enough(self, img):
        """
        如果图片小于 crop_size，就先等比例放大到至少 crop_size。
        """
        w, h = img.size

        if w >= self.crop_size and h >= self.crop_size:
            return img

        scale = self.crop_size / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        return img.resize((new_w, new_h), Image.BICUBIC)

    def __getitem__(self, idx):
        y_img = self._load_y_channel(self.image_paths[idx])
        y_img = self._ensure_large_enough(y_img)

        # 先在 HR 上裁剪 256×256 patch
        target_pil = self.crop(y_img)

        # 构造 LR 输入：256 -> 64 -> 256
        lr_size = self.crop_size // self.zoom_factor

        input_pil = target_pil.resize((lr_size, lr_size), Image.BICUBIC)
        input_pil = input_pil.resize((self.crop_size, self.crop_size), Image.BICUBIC)

        input_img = self.to_tensor(input_pil)
        target = self.to_tensor(target_pil)

        return input_img, target


# ===================== 参数 =====================
def parse_args():
    parser = argparse.ArgumentParser(description="Train SRCNN / SRCNN_MC")

    parser.add_argument("--train-dir", type=str, default="./data/BSDS500/train")
    parser.add_argument("--val-dir", type=str, default="./data/BSDS500/val")
    parser.add_argument("--save-dir", type=str, default="./train_model")

    parser.add_argument("--model", type=str, default="srcnn_mc",
                        choices=["srcnn", "srcnn_mc"])
    parser.add_argument("--dropout-p", type=float, default=0.10)

    parser.add_argument("--zoom-factor", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)

    # 建议 DIV2K 用 4 或 8
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--val-save-every", type=int, default=10)

    parser.add_argument("--lr-conv1", type=float, default=1e-4)
    parser.add_argument("--lr-conv2", type=float, default=1e-4)
    parser.add_argument("--lr-conv3", type=float, default=1e-5)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto / cuda:0 / cpu")

    parser.add_argument("--resume", type=str, default="",
                        help="checkpoint path, e.g. ./train_model/latest_checkpoint.pth")

    parser.add_argument("--auto-resume", action="store_true",
                        help="automatically resume from save-dir/latest_checkpoint.pth if it exists")

    return parser.parse_args()


def build_model(args):
    if args.model == "srcnn":
        return SRCNN()
    return SRCNN_MC(dropout_p=args.dropout_p)


def set_mc_dropout_if_exists(model, enabled: bool):
    """
    只有 SRCNN_MC 有 set_mc_dropout。
    普通 SRCNN 没有这个函数，直接跳过。
    """
    if hasattr(model, "set_mc_dropout"):
        model.set_mc_dropout(enabled)


def resolve_resume_path(args):
    """
    优先使用 --resume。
    如果没写 --resume，但写了 --auto-resume，则尝试读取 save-dir/latest_checkpoint.pth。
    """
    if args.resume:
        return args.resume

    if args.auto_resume:
        latest_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
        if os.path.exists(latest_path):
            return latest_path

    return ""


def load_resume_if_needed(model, optimizer, resume_path, device):
    start_epoch = 0
    best_psnr = 0.0

    if not resume_path:
        return start_epoch, best_psnr

    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    ckpt = torch.load(resume_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = int(ckpt.get("epoch", 0))
        best_psnr = float(ckpt.get("best_psnr", 0.0))

        print(f"[Resume] checkpoint: {resume_path}")
        print(f"[Resume] start_epoch: {start_epoch + 1}, best_psnr: {best_psnr:.2f}")

    else:
        model.load_state_dict(ckpt, strict=False)
        print(f"[Load] model weights from: {resume_path}")

    return start_epoch, best_psnr


def seed_worker(worker_id):
    """
    保证多进程 DataLoader 下随机性可控。
    """
    worker_seed = torch.initial_seed() % 2**32
    import random
    import numpy as np
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloader(dataset, batch_size, shuffle, num_workers, device, seed):
    """
    带 pin_memory / persistent_workers / prefetch_factor 的 DataLoader。
    """
    pin_memory = (device.type == "cuda")
    persistent_workers = num_workers > 0

    generator = torch.Generator()
    generator.manual_seed(seed)

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def validate_deterministic(model, valloader, criterion, device):
    """
    验证阶段：
    - model.eval()
    - mc_dropout=False
    - Dropout 关闭
    - 用确定性输出计算 PSNR / SSIM
    """
    model.eval()
    set_mc_dropout_if_exists(model, False)

    sum_loss = 0.0
    sum_psnr = 0.0
    sum_ssim = 0.0

    with torch.no_grad():
        for batch in valloader:
            input_img = batch[0].to(device, non_blocking=True)
            target = batch[1].to(device, non_blocking=True)

            out = model(input_img)
            loss = criterion(out, target)

            pr = psnr(loss)
            sm = ssim(out, target)

            sum_loss += loss.item()
            sum_psnr += pr
            sum_ssim += sm.item()

    avg_loss = sum_loss / len(valloader)
    avg_psnr = sum_psnr / len(valloader)
    avg_ssim = sum_ssim / len(valloader)

    return avg_loss, avg_psnr, avg_ssim


def save_checkpoint(path, epoch, model, optimizer, best_psnr, avg_epoch_loss,
                    avg_val_loss=None, avg_psnr=None, avg_ssim=None, args=None):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "last_train_loss": avg_epoch_loss,
            "last_val_loss": avg_val_loss,
            "last_val_psnr": avg_psnr,
            "last_val_ssim": avg_ssim,
            "args": vars(args) if args is not None else None,
        },
        path
    )


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # 对固定输入尺寸的卷积有时能加速
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print("Training Config")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"device: {device}")
    print("=" * 70)

    # ===================== 数据加载 =====================
    trainset = SRFolderDataset(
        image_dir=args.train_dir,
        zoom_factor=args.zoom_factor,
        crop_size=args.crop_size,
        train=True
    )

    valset = SRFolderDataset(
        image_dir=args.val_dir,
        zoom_factor=args.zoom_factor,
        crop_size=args.crop_size,
        train=False
    )

    trainloader = build_dataloader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed
    )

    valloader = build_dataloader(
        dataset=valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed + 1
    )

    print(f"[Info] train samples: {len(trainset)}")
    print(f"[Info] val samples:   {len(valset)}")
    print(f"[Info] train batches: {len(trainloader)}")
    print(f"[Info] val batches:   {len(valloader)}")
    print(f"[Info] crop size:     {args.crop_size}x{args.crop_size}")
    print(f"[Info] train crop:    RandomCrop")
    print(f"[Info] val crop:      CenterCrop")
    print(f"[Info] num_workers:   {args.num_workers}")
    print(f"[Info] pin_memory:    {device.type == 'cuda'}")
    print(f"[Info] persistent_workers: {args.num_workers > 0}")

    # ===================== 模型 =====================
    model = build_model(args).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        [
            {"params": model.conv1.parameters(), "lr": args.lr_conv1},
            {"params": model.conv2.parameters(), "lr": args.lr_conv2},
            {"params": model.conv3.parameters(), "lr": args.lr_conv3},
        ]
    )

    resume_path = resolve_resume_path(args)

    start_epoch, best_psnr = load_resume_if_needed(
        model=model,
        optimizer=optimizer,
        resume_path=resume_path,
        device=device
    )

    # ===================== 训练循环 =====================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        set_mc_dropout_if_exists(model, False)
        # 注意：
        # 此时虽然 mc_dropout=False，
        # 但 model.train() 会让 self.training=True，
        # 所以 SRCNN_MC 里的 Dropout 仍然开启。

        epoch_loss = 0.0

        for iteration, batch in enumerate(trainloader):
            input_img = batch[0].to(device, non_blocking=True)
            target = batch[1].to(device, non_blocking=True)

            optimizer.zero_grad()

            out = model(input_img)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(trainloader)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Training Loss: {avg_epoch_loss:.6f}"
        )

        # ===================== 验证 + 保存 =====================
        if (epoch + 1) % args.val_save_every == 0:
            print("-" * 70)
            print("[Validation] deterministic eval, dropout OFF")

            avg_val_loss, avg_val_psnr, avg_val_ssim = validate_deterministic(
                model=model,
                valloader=valloader,
                criterion=criterion,
                device=device
            )

            print(
                f"Validation | "
                f"Loss: {avg_val_loss:.6f} | "
                f"Average PSNR: {avg_val_psnr:.2f} dB | "
                f"Average SSIM: {avg_val_ssim:.4f}"
            )

            # 按验证集确定性 PSNR 保存 best
            if avg_val_psnr >= best_psnr:
                best_psnr = avg_val_psnr

                best_path = os.path.join(args.save_dir, "best_model_SRCNN.pth")
                torch.save(model.state_dict(), best_path)

                print(
                    f"Best model updated! "
                    f"Saved to {best_path} | "
                    f"Best PSNR: {best_psnr:.2f} dB"
                )

            # 保存当前 epoch checkpoint
            current_path = os.path.join(
                args.save_dir,
                f"SRCNN_epoch_{epoch + 1}.pth"
            )

            save_checkpoint(
                path=current_path,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                best_psnr=best_psnr,
                avg_epoch_loss=avg_epoch_loss,
                avg_val_loss=avg_val_loss,
                avg_psnr=avg_val_psnr,
                avg_ssim=avg_val_ssim,
                args=args
            )

            # 额外保存 latest checkpoint，方便 --auto-resume
            latest_path = os.path.join(args.save_dir, "latest_checkpoint.pth")

            save_checkpoint(
                path=latest_path,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                best_psnr=best_psnr,
                avg_epoch_loss=avg_epoch_loss,
                avg_val_loss=avg_val_loss,
                avg_psnr=avg_val_psnr,
                avg_ssim=avg_val_ssim,
                args=args
            )

            print(f"Current epoch checkpoint saved to {current_path}")
            print(f"Latest checkpoint saved to {latest_path}")
            print("-" * 70)

    print(f"\nTraining completed! Best PSNR: {best_psnr:.2f} dB")
    print(f"Saved in: {args.save_dir}")


if __name__ == "__main__":
    main()