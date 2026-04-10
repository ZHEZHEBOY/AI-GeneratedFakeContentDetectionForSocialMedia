"""
数据集下载与预处理模块
支持 CIFAKE 数据集（Kaggle公开，真实图像 vs Stable Diffusion生成图像）
也支持本地自定义数据集
"""

import os
import shutil
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json


# ============================================================
# 路径配置
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for d in [RAW_DIR, PROCESSED_DIR, DATA_DIR / "features"]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# CIFAKE 数据集下载（Kaggle API）
# ============================================================
def download_cifake(target_dir: Path = RAW_DIR):
    """
    使用 Kaggle API 下载 CIFAKE 数据集。
    需要提前配置 ~/.kaggle/kaggle.json 或设置环境变量
    KAGGLE_USERNAME / KAGGLE_KEY。
    """
    try:
        import kaggle
        print("[INFO] 正在从 Kaggle 下载 CIFAKE 数据集（约 1.3GB）...")
        kaggle.api.dataset_download_files(
            "birdy654/cifake-real-and-ai-generated-synthetic-images",
            path=str(target_dir),
            unzip=True
        )
        print(f"[INFO] 下载完成，数据保存至：{target_dir}")
    except Exception as e:
        print(f"[WARN] Kaggle 下载失败：{e}")
        print("[HINT] 请手动下载：https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
        print("[HINT] 解压后将 REAL/ 和 FAKE/ 文件夹放到 data/raw/ 目录下")


# ============================================================
# 图像加载工具
# ============================================================
def load_images_from_dir(
    real_dir: Path,
    fake_dir: Path,
    max_per_class: int = 5000,
    img_size: tuple = (64, 64),
    grayscale: bool = True,
) -> tuple:
    """
    从真实/虚假两个目录加载图像，返回 numpy 数组和标签。

    Args:
        real_dir:      真实图像目录
        fake_dir:      AI合成图像目录
        max_per_class: 每类最多加载数量（控制内存）
        img_size:      统一缩放尺寸
        grayscale:     是否转为灰度（频域分析通常用灰度）

    Returns:
        images: np.ndarray, shape (N, H, W)
        labels: np.ndarray, shape (N,), 0=真实, 1=AI合成
    """
    def _load_from(folder: Path, label: int, limit: int):
        imgs, labs = [], []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [f for f in folder.iterdir() if f.suffix.lower() in exts]
        files = files[:limit]
        for fpath in tqdm(files, desc=f"加载 {'REAL' if label==0 else 'FAKE'} ({folder.name})"):
            try:
                img = Image.open(fpath)
                img = img.convert("L") if grayscale else img.convert("RGB")
                img = img.resize(img_size, Image.LANCZOS)
                imgs.append(np.array(img, dtype=np.float32) / 255.0)
                labs.append(label)
            except Exception as e:
                print(f"[WARN] 跳过 {fpath.name}: {e}")
        return imgs, labs

    real_imgs, real_labs = _load_from(real_dir, label=0, limit=max_per_class)
    fake_imgs, fake_labs = _load_from(fake_dir, label=1, limit=max_per_class)

    images = np.array(real_imgs + fake_imgs, dtype=np.float32)
    labels = np.array(real_labs + fake_labs, dtype=np.int64)

    # 打乱顺序
    idx = np.random.permutation(len(labels))
    return images[idx], labels[idx]


# ============================================================
# 自动扫描 data/raw 目录结构
# ============================================================
def auto_detect_dataset(raw_dir: Path = RAW_DIR) -> tuple:
    """
    自动识别数据集目录结构，支持以下格式：
    1. raw/REAL/ + raw/FAKE/
    2. raw/train/REAL/ + raw/train/FAKE/
    3. raw/real/ + raw/fake/（不区分大小写）
    """
    candidates = {
        "real": ["REAL", "real", "Real", "genuine", "authentic"],
        "fake": ["FAKE", "fake", "Fake", "synthetic", "ai", "AI", "generated"],
    }

    def find_dir(base, names):
        for n in names:
            p = base / n
            if p.exists() and p.is_dir():
                return p
        return None

    # 直接在 raw/ 下找
    real_dir = find_dir(raw_dir, candidates["real"])
    fake_dir = find_dir(raw_dir, candidates["fake"])
    if real_dir and fake_dir:
        return real_dir, fake_dir

    # 在 raw/train/ 下找
    train_dir = raw_dir / "train"
    if train_dir.exists():
        real_dir = find_dir(train_dir, candidates["real"])
        fake_dir = find_dir(train_dir, candidates["fake"])
        if real_dir and fake_dir:
            return real_dir, fake_dir

    # 在 raw/cifake/ 下找（CIFAKE 标准结构）
    cifake_dir = raw_dir / "cifake"
    if cifake_dir.exists():
        real_dir = find_dir(cifake_dir, candidates["real"])
        fake_dir = find_dir(cifake_dir, candidates["fake"])
        if real_dir and fake_dir:
            return real_dir, fake_dir

    raise FileNotFoundError(
        f"在 {raw_dir} 中未找到真实/虚假图像目录。\n"
        "请确保目录结构为：\n"
        "  data/raw/REAL/  (真实图像)\n"
        "  data/raw/FAKE/  (AI合成图像)\n"
        "或 CIFAKE 标准结构：\n"
        "  data/raw/cifake/REAL/\n"
        "  data/raw/cifake/FAKE/\n"
    )


# ============================================================
# 生成小型演示数据集（无数据时用于测试）
# ============================================================
def generate_demo_dataset(
    n_real: int = 500,
    n_fake: int = 500,
    img_size: tuple = (64, 64),
    save_dir: Path = RAW_DIR,
):
    """
    生成合成演示数据集用于调试。
    - 真实图像：自然纹理（低频成分为主）
    - AI合成图像：注入高频噪声（模拟GAN/扩散模型伪影）
    """
    print("[INFO] 生成演示数据集（真实图像用自然纹理，AI合成图像注入高频噪声）...")

    real_save = save_dir / "REAL"
    fake_save = save_dir / "FAKE"
    real_save.mkdir(exist_ok=True)
    fake_save.mkdir(exist_ok=True)

    rng = np.random.default_rng(42)
    H, W = img_size

    for i in tqdm(range(n_real), desc="生成 REAL"):
        # 低频平滑纹理
        base = rng.uniform(0.3, 0.8, (H, W))
        noise = rng.normal(0, 0.02, (H, W))
        img = np.clip(base + noise, 0, 1)
        # 添加低频渐变
        grad = np.linspace(0, 0.1, W)[np.newaxis, :]
        img = np.clip(img + grad, 0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).save(real_save / f"real_{i:04d}.png")

    for i in tqdm(range(n_fake), desc="生成 FAKE"):
        # 基础纹理
        base = rng.uniform(0.2, 0.7, (H, W))
        # 注入高频网格噪声（模拟GAN伪影）
        hf_noise = rng.normal(0, 0.15, (H, W))
        grid_x = np.sin(np.linspace(0, 8 * np.pi, W))
        grid_y = np.sin(np.linspace(0, 8 * np.pi, H))
        grid = np.outer(grid_y, grid_x) * 0.05
        img = np.clip(base + hf_noise + grid, 0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).save(fake_save / f"fake_{i:04d}.png")

    print(f"[INFO] 演示数据集已保存至 {save_dir}")
    return real_save, fake_save


# ============================================================
# 保存/加载处理后的数据
# ============================================================
def save_processed(images: np.ndarray, labels: np.ndarray, name: str = "dataset"):
    path = PROCESSED_DIR / f"{name}.npz"
    np.savez_compressed(path, images=images, labels=labels)
    meta = {"n_samples": len(labels), "n_real": int((labels == 0).sum()), "n_fake": int((labels == 1).sum()), "image_shape": list(images.shape[1:])}
    with open(PROCESSED_DIR / f"{name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存处理后数据：{path}")
    print(f"[INFO] 数据统计：{meta}")
    return path


def load_processed(name: str = "dataset") -> tuple:
    path = PROCESSED_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"处理后数据不存在：{path}，请先运行预处理。")
    data = np.load(path)
    return data["images"], data["labels"]


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据集预处理")
    parser.add_argument("--demo", action="store_true", help="使用演示数据集（自动生成）")
    parser.add_argument("--download", action="store_true", help="从Kaggle下载CIFAKE数据集")
    parser.add_argument("--max_per_class", type=int, default=5000, help="每类最多加载图像数")
    parser.add_argument("--img_size", type=int, default=64, help="图像缩放尺寸")
    args = parser.parse_args()

    if args.download:
        download_cifake()

    if args.demo:
        real_dir, fake_dir = generate_demo_dataset(n_real=500, n_fake=500)
    else:
        real_dir, fake_dir = auto_detect_dataset()

    print(f"[INFO] 真实图像目录：{real_dir}")
    print(f"[INFO] AI合成图像目录：{fake_dir}")

    images, labels = load_images_from_dir(
        real_dir, fake_dir,
        max_per_class=args.max_per_class,
        img_size=(args.img_size, args.img_size),
        grayscale=True
    )

    print(f"[INFO] 加载完成：{images.shape}，标签分布：真实={( labels==0).sum()}, AI合成={(labels==1).sum()}")
    save_processed(images, labels, name="dataset")
