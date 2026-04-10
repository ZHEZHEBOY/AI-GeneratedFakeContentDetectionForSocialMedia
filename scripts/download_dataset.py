"""
数据集下载脚本
支持：
  1. CIFAKE (Kaggle)  —— 推荐，120,000张
  2. GenImage 子集 (HuggingFace)  —— 大规模
  3. 本地路径导入
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 工具函数
# ============================================================
def check_kaggle_config() -> bool:
    """检查 Kaggle API 配置是否存在"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print(f"[✓] 已找到 Kaggle API 配置：{kaggle_json}")
        return True
    else:
        print(f"[✗] 未找到 Kaggle API 配置：{kaggle_json}")
        return False


def setup_kaggle_api(username: str, key: str):
    """写入 Kaggle API 凭据"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    config = {"username": username, "key": key}
    with open(kaggle_json, "w") as f:
        json.dump(config, f)
    # Windows 不需要修改权限，Linux/Mac 需要
    if sys.platform != "win32":
        kaggle_json.chmod(0o600)
    print(f"[✓] Kaggle API 配置已写入：{kaggle_json}")


def verify_dataset_structure(dataset_dir: Path) -> bool:
    """验证数据集目录结构是否正确"""
    real_dirs = list(dataset_dir.rglob("REAL")) + list(dataset_dir.rglob("real"))
    fake_dirs = list(dataset_dir.rglob("FAKE")) + list(dataset_dir.rglob("fake"))

    if not real_dirs or not fake_dirs:
        return False

    real_count = sum(
        len([f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        for d in real_dirs if d.is_dir()
    )
    fake_count = sum(
        len([f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        for d in fake_dirs if d.is_dir()
    )

    print(f"[✓] 数据集验证通过：真实图像 {real_count} 张，AI合成图像 {fake_count} 张")
    return real_count > 0 and fake_count > 0


# ============================================================
# CIFAKE 下载（Kaggle API）
# ============================================================
def download_cifake_kaggle(target_dir: Path = DATA_DIR / "cifake"):
    """使用 Kaggle API 下载 CIFAKE 数据集"""
    target_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已下载
    if verify_dataset_structure(target_dir):
        print(f"[INFO] CIFAKE 数据集已存在于 {target_dir}，跳过下载。")
        return target_dir

    if not check_kaggle_config():
        print("\n" + "=" * 60)
        print("请按以下步骤配置 Kaggle API：")
        print("  1. 登录 https://www.kaggle.com")
        print("  2. 点击右上角头像 → Settings → API → Create New Token")
        print("  3. 下载 kaggle.json 文件")
        print(f"  4. 将文件放到：{Path.home() / '.kaggle' / 'kaggle.json'}")
        print("  5. 重新运行此脚本")
        print("=" * 60)
        print("\n或者直接运行：")
        print("  python download_dataset.py --kaggle-user <用户名> --kaggle-key <API_KEY>")
        sys.exit(1)

    try:
        import kaggle
        print("[INFO] 开始下载 CIFAKE 数据集（约 1.3 GB，请耐心等待）...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "birdy654/cifake-real-and-ai-generated-synthetic-images",
            path=str(target_dir),
            unzip=True,
            quiet=False,
        )
        print(f"\n[✓] CIFAKE 下载并解压完成：{target_dir}")
        verify_dataset_structure(target_dir)
        return target_dir

    except ImportError:
        print("[ERROR] 未安装 kaggle 库，请运行：pip install kaggle")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 下载失败：{e}")
        print("\n请手动下载：")
        print("  https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
        print(f"  下载后解压到：{target_dir}")
        sys.exit(1)


# ============================================================
# GenImage 子集下载（HuggingFace）
# ============================================================
def download_genimage_subset(
    target_dir: Path = DATA_DIR / "genimage",
    n_per_class: int = 5000,
    generator: str = "midjourney",
):
    """
    从 HuggingFace 下载 GenImage 子集
    generator: midjourney / stable_diffusion / dalle3 等
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        from PIL import Image as PILImage

        print(f"[INFO] 从 HuggingFace 下载 GenImage 子集（{generator}，每类 {n_per_class} 张）...")
        ds = load_dataset(
            "jzousz/GenImage",
            split=f"train[:{n_per_class * 2}]",
            trust_remote_code=True
        )

        real_dir = target_dir / "REAL"
        fake_dir = target_dir / "FAKE"
        real_dir.mkdir(exist_ok=True)
        fake_dir.mkdir(exist_ok=True)

        real_count, fake_count = 0, 0
        for i, item in enumerate(ds):
            img = item.get("image") or item.get("img")
            label = item.get("label", 0)
            if img is None:
                continue
            if label == 0 and real_count < n_per_class:
                img.save(real_dir / f"real_{real_count:05d}.jpg")
                real_count += 1
            elif label == 1 and fake_count < n_per_class:
                img.save(fake_dir / f"fake_{fake_count:05d}.jpg")
                fake_count += 1
            if real_count >= n_per_class and fake_count >= n_per_class:
                break
            if i % 500 == 0:
                print(f"  进度：真实={real_count}/{n_per_class}，AI合成={fake_count}/{n_per_class}")

        print(f"[✓] GenImage 子集保存完成：{target_dir}")
        return target_dir

    except ImportError:
        print("[ERROR] 请先安装：pip install datasets Pillow")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] HuggingFace 下载失败：{e}")
        print("请检查网络连接，或设置 HF_ENDPOINT 镜像：")
        print("  set HF_ENDPOINT=https://hf-mirror.com  (Windows)")
        print("  export HF_ENDPOINT=https://hf-mirror.com  (Linux/Mac)")
        sys.exit(1)


# ============================================================
# 从本地 ZIP 文件导入
# ============================================================
def import_from_zip(zip_path: str, target_dir: Path = DATA_DIR / "cifake"):
    """从本地已下载的 ZIP 文件导入"""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        print(f"[ERROR] ZIP 文件不存在：{zip_path}")
        sys.exit(1)

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 正在解压 {zip_path} → {target_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    print(f"[✓] 解压完成：{target_dir}")
    verify_dataset_structure(target_dir)
    return target_dir


# ============================================================
# 打印数据集统计信息
# ============================================================
def print_dataset_stats(dataset_dir: Path):
    """打印数据集目录下的图像统计"""
    print(f"\n{'=' * 50}")
    print(f"数据集统计：{dataset_dir}")
    print(f"{'=' * 50}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for subdir in sorted(dataset_dir.rglob("*")):
        if subdir.is_dir():
            imgs = [f for f in subdir.iterdir() if f.suffix.lower() in exts]
            if imgs:
                print(f"  {subdir.relative_to(dataset_dir)}: {len(imgs)} 张图像")
    print(f"{'=' * 50}\n")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="数据集下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 使用 Kaggle API 下载 CIFAKE（推荐）
  python download_dataset.py --cifake

  # 指定 Kaggle 凭据并下载
  python download_dataset.py --cifake --kaggle-user your_username --kaggle-key your_api_key

  # 从已下载的 ZIP 文件导入
  python download_dataset.py --from-zip C:/Users/yourname/Downloads/archive.zip

  # 下载 GenImage 子集（需要 HuggingFace）
  python download_dataset.py --genimage --n-per-class 3000

  # 检查现有数据集结构
  python download_dataset.py --check
        """
    )
    parser.add_argument("--cifake", action="store_true", help="下载 CIFAKE 数据集（Kaggle）")
    parser.add_argument("--genimage", action="store_true", help="下载 GenImage 子集（HuggingFace）")
    parser.add_argument("--from-zip", type=str, help="从本地 ZIP 文件导入")
    parser.add_argument("--kaggle-user", type=str, help="Kaggle 用户名")
    parser.add_argument("--kaggle-key", type=str, help="Kaggle API Key")
    parser.add_argument("--n-per-class", type=int, default=5000, help="每类图像数量（GenImage用）")
    parser.add_argument("--check", action="store_true", help="检查 data/raw 目录下已有的数据集")
    args = parser.parse_args()

    # 写入 Kaggle 凭据
    if args.kaggle_user and args.kaggle_key:
        setup_kaggle_api(args.kaggle_user, args.kaggle_key)

    if args.check:
        print_dataset_stats(DATA_DIR)

    elif args.from_zip:
        target = import_from_zip(args.from_zip)
        print_dataset_stats(target)

    elif args.cifake:
        target = download_cifake_kaggle()
        print_dataset_stats(target)

    elif args.genimage:
        target = download_genimage_subset(n_per_class=args.n_per_class)
        print_dataset_stats(target)

    else:
        print("请指定操作，使用 --help 查看帮助。")
        print("\n快速开始（推荐）：")
        print("  python download_dataset.py --cifake")
        parser.print_help()
