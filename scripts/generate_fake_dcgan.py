"""
CIFAKE FAKE 图像生成器 - DCGAN 训练版
用 DCGAN 在 CIFAR-10 上训练，生成对应的 FAKE 图像
与 CIFAKE 原论文 (arXiv:2303.14126) 方法一致
"""
import os
import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm

# ============ 配置 ============
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "raw" / "cifake"
FAKE_DIR = DATA_DIR / "FAKE"
REAL_DIR = DATA_DIR / "REAL"
GENERATED_DIR = DATA_DIR / "generated_fake"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256    # GPU: 更大 batch 加速
LATENT_DIM = 100
NGF = 64
NDF = 64
EPOCHS = 50
LR = 0.0002
BETA1 = 0.5
N_PER_CLASS = 1000  # 每类生成数（完整 CIFAKE 规模）

print(f"[设备] {DEVICE}")
if torch.cuda.is_available():
    print(f"[GPU]  {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
print(f"[配置] batch={BATCH_SIZE}, latent={LATENT_DIM}, epochs={EPOCHS}")

# ============================================================
# DCGAN 模型定义
# ============================================================
class Generator(nn.Module):
    """生成器：Latent vector → 3×32×32 图像"""
    def __init__(self, latent_dim=100, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            # 输入: latent_dim × 1 × 1
            nn.ConvTranspose2d(latent_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 4×4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 8×8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 16×16
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 32×32
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """判别器：3×32×32 图像 → 真/假"""
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # 32×32
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 16×16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 8×8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4×4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)


def weights_init(m):
    """初始化权重（CIFAKE 论文推荐）"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_dcgan():
    """训练 DCGAN"""
    print("\n[Step 1] 加载 CIFAR-10 数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(
        root=str(ROOT_DIR / "data" / "raw"),
        train=True,
        download=False,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"  数据集大小: {len(dataset)} 张, {len(dataloader)} 批次/epoch")

    # 创建目录
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    REAL_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化模型
    print("\n[Step 2] 初始化 DCGAN...")
    netG = Generator(LATENT_DIM, NGF).to(DEVICE)
    netD = Discriminator(NDF).to(DEVICE)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

    # 固定噪声（用于可视化生成进度）
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)

    print(f"\n[Step 3] 开始训练 DCGAN ({EPOCHS} epochs)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        lossD_sum, lossG_sum, D_x_sum, D_g_z_sum = 0, 0, 0, 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=80)
        for real_images, _ in pbar:
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)

            # ===== 训练判别器 =====
            netD.zero_grad()
            label_real = torch.full((batch_size,), 0.9, dtype=torch.float, device=DEVICE)  # label smoothing
            label_fake = torch.full((batch_size,), 0.1, dtype=torch.float, device=DEVICE)

            output_real = netD(real_images)
            D_x = output_real.mean().item()

            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach())
            D_g_z = output_fake.mean().item()

            lossD = criterion(netD(real_images), label_real) + criterion(output_fake, label_fake)
            lossD.backward()
            optimizerD.step()

            # ===== 训练生成器 =====
            netG.zero_grad()
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_images = netG(noise)
            output = netD(fake_images)
            lossG = criterion(output, torch.full((batch_size,), 1.0, dtype=torch.float, device=DEVICE))
            lossG.backward()
            optimizerG.step()

            lossD_sum += lossD.item()
            lossG_sum += lossG.item()
            D_x_sum += D_x
            D_g_z_sum += D_g_z
            n_batches += 1

            pbar.set_postfix({
                "D": f"{lossD.item():.3f}",
                "G": f"{lossG.item():.3f}",
                "D(x)": f"{D_x:.3f}",
                "D(G(z))": f"{D_g_z:.3f}"
            })

        # 每10个epoch保存生成样本
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake_samples = netG(fixed_noise).detach().cpu()
            save_image(
                fake_samples,
                GENERATED_DIR / f"epoch_{epoch+1:03d}.png",
                nrow=8,
                normalize=True
            )
            print(f"  [保存] epoch {epoch+1} 样本 → {GENERATED_DIR / f'epoch_{epoch+1:03d}.png'}")

        # 保存检查点
        torch.save({
            "netG": netG.state_dict(),
            "netD": netD.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "optimizerD": optimizerD.state_dict(),
            "epoch": epoch + 1,
        }, CHECKPOINT_DIR / "dcgan_latest.pt")

        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch+1} 完成: D_loss={lossD_sum/n_batches:.4f}, G_loss={lossG_sum/n_batches:.4f}, D(x)={D_x_sum/n_batches:.3f}, 耗时={elapsed:.0f}s")

    total_time = time.time() - start_time
    print(f"\n[训练完成] 总耗时 {total_time/60:.1f} 分钟")
    return netG


def generate_fake_images(netG, n_per_class=500):
    """用训练好的 Generator 生成 FAKE 图像"""
    print(f"\n[Step 4] 生成 FAKE 图像（每类 {n_per_class} 张）...")

    # 加载 CIFAR-10 获取索引
    dataset = datasets.CIFAR10(
        root=str(ROOT_DIR / "data" / "raw"),
        train=True,
        download=False
    )

    # 按类别分组
    class_indices = {c: [] for c in range(10)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if len(class_indices[label]) < n_per_class:
            class_indices[label].append(idx)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    netG.eval()
    total = 0
    start_time = time.time()

    for class_idx in range(10):
        indices = class_indices[class_idx]
        if not indices:
            continue

        n = len(indices)
        print(f"  Class {class_idx} ({n} 张)...")

        # 批量生成
        with torch.no_grad():
            for batch_start in range(0, n, 100):
                batch_end = min(batch_start + 100, n)
                batch_n = batch_end - batch_start

                noise = torch.randn(batch_n, LATENT_DIM, 1, 1, device=DEVICE)
                fake_batch = netG(noise).detach().cpu()

                for i, fake_img in enumerate(fake_batch):
                    real_idx = indices[batch_start + i]
                    out_path = FAKE_DIR / f"fake_{real_idx:07d}.jpg"
                    save_image(fake_img, out_path, normalize=True)
                    total += 1

        elapsed = time.time() - start_time
        rate = total / elapsed if elapsed > 0 else 1
        remain = ((N_PER_CLASS * 10) - total) / rate / 60 if rate > 0 else 0
        print(f"    已生成 {total}/{N_PER_CLASS * 10} 张, 速率 {rate:.1f}/s, 预计剩余 {remain:.1f}min")

    print(f"\n[完成] 共生成 {total} 张 FAKE 图像！耗时 {(time.time()-start_time)/60:.1f} 分钟")


def copy_real_images(n_per_class=500):
    """复制 CIFAR-10 作为 REAL 图像"""
    print("\n[Step 5] 复制 REAL 图像...")
    REAL_DIR.mkdir(parents=True, exist_ok=True)

    dataset = datasets.CIFAR10(
        root=str(ROOT_DIR / "data" / "raw"),
        train=True,
        download=False
    )

    class_indices = {c: [] for c in range(10)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if len(class_indices[label]) < n_per_class:
            class_indices[label].append(idx)

    count = 0
    for class_idx, indices in class_indices.items():
        for idx in tqdm(indices, desc=f"  REAL class {class_idx}", ncols=70):
            img, _ = dataset[idx]
            out_path = REAL_DIR / f"real_{idx:07d}.jpg"
            if not out_path.exists():
                img.save(out_path)
                count += 1

    print(f"  REAL 图像: {count} 张（已含重复跳过）")


if __name__ == "__main__":
    print("=" * 60)
    print("CIFAKE 数据集生成器 - DCGAN 版本")
    print("=" * 60)

    # Step 1: 训练 DCGAN
    netG = train_dcgan()

    # Step 2: 生成 FAKE 图像
    generate_fake_images(netG, N_PER_CLASS)

    # Step 3: 复制 REAL 图像
    copy_real_images(N_PER_CLASS)

    # 验证
    real_n = len(list(REAL_DIR.glob("*.jpg"))) + len(list(REAL_DIR.glob("*.png")))
    fake_n = len(list(FAKE_DIR.glob("*.jpg"))) + len(list(FAKE_DIR.glob("*.png")))
    print(f"\n{'='*50}")
    print(f"[数据集完成]")
    print(f"  REAL: {real_n} 张")
    print(f"  FAKE: {fake_n} 张")
    print(f"{'='*50}")
