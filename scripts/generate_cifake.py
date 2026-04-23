"""

CIFAKE 数据集生成脚本
- REAL：直接使用已下载的 CIFAR-10 图像
- FAKE：使用 Stable Diffusion v1.5 为每张 CIFAR-10 图像生成 AI 合成版本
"""
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from diffusers import StableDiffusionImg2ImgPipeline
from tqdm import tqdm

# ============ 配置 ============
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "raw" / "cifake"
REAL_DIR = DATA_DIR / "REAL"
FAKE_DIR = DATA_DIR / "FAKE"
CIFAR_DIR = ROOT_DIR / "data" / "raw" / "cifake-raw"

# 生成数量：训练集每类取样数（控制总数据量，完整版可设为5000）
N_PER_CLASS = 1000  # 总共 10000 张（5000 REAL + 5000 FAKE）竞赛够用

def setup_cifake_dirs():
    """创建 CIFAKE 标准目录结构"""
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[目录] REAL={REAL_DIR}")
    print(f"[目录] FAKE={FAKE_DIR}")

def load_cifar10():
    """加载 CIFAR-10 训练集"""
    transform = transforms.ToPILImage()
    train_set = datasets.CIFAR10(root=str(ROOT_DIR / "data" / "raw"), train=True, download=True)
    print(f"[CIFAR-10] 共 {len(train_set)} 张图像，加载完成")
    return train_set

def setup_sd_pipeline():
    """加载 Stable Diffusion Img2Img 管道"""
    print("[SD] 正在加载 Stable Diffusion v1.5（CPU，约2-5分钟）...")
    device = "cpu"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f"[SD] 模型加载完成，设备={device}")
    return pipe

def generate_fake_images(train_set, pipe, n_per_class=1000):
    """为每类生成 n_per_class 张 FAKE 图像"""
    # 按类别分组
    class_indices = {c: [] for c in range(10)}
    for idx, (img, label) in enumerate(train_set):
        if len(class_indices[label]) < n_per_class:
            class_indices[label].append((idx, img))

    # CIFAKE 类别名称
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    total = 0
    start_time = time.time()

    for class_idx in range(10):
        indices = class_indices[class_idx]
        if not indices:
            continue

        prompt = f"a high quality photo of a {class_names[class_idx]}"
        print(f"\n[Class {class_idx}] '{class_names[class_idx]}' - 生成 {len(indices)} 张FAKE图像")
        print(f"[Prompt] {prompt}")

        # 进度条
        pbar = tqdm(indices, desc=f"  Class {class_idx}", ncols=80)

        for i, (idx, real_img) in enumerate(pbar):
            # 生成 FAKE 图像
            fake_img = pipe(
                prompt=prompt,
                image=real_img,
                strength=0.75,   # 变换强度，越高越不像原图
                guidance_scale=7.5,
                num_inference_steps=30,
            ).images[0]

            # 保存：文件名格式 fake_XXXXXXX.jpg
            filename = f"fake_{idx:07d}.jpg"
            fake_img.save(FAKE_DIR / filename)

            total += 1
            if total % 50 == 0:
                elapsed = time.time() - start_time
                rate = total / elapsed
                eta = (10000 - total) / rate if rate > 0 else 0
                pbar.set_postfix({"已生成": f"{total}/10000", "速率": f"{rate:.1f}/s", "预计剩余": f"{eta/60:.1f}min"})

    print(f"\n[完成] 共生成 {total} 张FAKE图像，耗时 {(time.time()-start_time)/60:.1f} 分钟")

def copy_real_images(train_set, n_per_class=1000):
    """将 CIFAR-10 图像复制到 REAL 目录"""
    class_indices = {c: [] for c in range(10)}
    for idx, (img, _) in enumerate(train_set):
        if len(class_indices[train_set[idx][1]]) < n_per_class:
            class_indices[train_set[idx][1]].append(idx)

    count = 0
    for class_idx, indices in class_indices.items():
        for idx in tqdm(indices, desc=f"复制REAL class {class_idx}", ncols=70):
            img, _ = train_set[idx]
            filename = f"real_{idx:07d}.jpg"
            img.save(REAL_DIR / filename)
            count += 1

    print(f"[REAL] 复制完成，共 {count} 张")

def verify_dataset():
    """验证数据集"""
    real_files = list(REAL_DIR.glob("*.jpg")) + list(REAL_DIR.glob("*.png"))
    fake_files = list(FAKE_DIR.glob("*.jpg")) + list(FAKE_DIR.glob("*.png"))
    print(f"\n{'='*50}")
    print(f"[验证] REAL: {len(real_files)} 张")
    print(f"[验证] FAKE: {len(fake_files)} 张")
    print(f"{'='*50}")
    return len(real_files) > 0 and len(fake_files) > 0

if __name__ == "__main__":
    print("=" * 60)
    print("CIFAKE 数据集生成器")
    print("=" * 60)

    setup_cifake_dirs()

    # Step 1: 加载 CIFAR-10
    print("\n[Step 1] 加载 CIFAR-10...")
    train_set = load_cifar10()

    # Step 2: 检查是否已有 FAKE 图像
    existing_fake = len(list(FAKE_DIR.glob("*.jpg"))) + len(list(FAKE_DIR.glob("*.png")))
    if existing_fake >= N_PER_CLASS * 10:
        print(f"[跳过] FAKE 图像已存在 ({existing_fake} 张)，跳过生成")
    else:
        # Step 3: 加载 SD 模型
        print("\n[Step 2] 加载 Stable Diffusion...")
        pipe = setup_sd_pipeline()

        # Step 4: 生成 FAKE 图像
        print("\n[Step 3] 生成 FAKE 图像（使用SD）...")
        generate_fake_images(train_set, pipe, N_PER_CLASS)

    # Step 5: 复制 REAL 图像
    existing_real = len(list(REAL_DIR.glob("*.jpg"))) + len(list(REAL_DIR.glob("*.png")))
    if existing_real >= N_PER_CLASS * 10:
        print(f"[跳过] REAL 图像已存在 ({existing_real} 张)")
    else:
        print("\n[Step 4] 复制 REAL 图像...")
        copy_real_images(train_set, N_PER_CLASS)

    # Step 6: 验证
    print("\n[Step 5] 验证数据集...")
    verify_dataset()
    print("\n[全部完成!]")
