"""CIFAKE FAKE 图像生成 - 使用 CPU 卸载模式节省内存"""
import gc
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import datasets
from diffusers import StableDiffusionImg2ImgPipeline

# ============ 配置 ============
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "raw" / "cifake"
FAKE_DIR = DATA_DIR / "FAKE"
REAL_DIR = DATA_DIR / "REAL"
SD_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--runwayml--stable-diffusion-v1-5" / "snapshots" / "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"

N_PER_CLASS = 500   # 每类500张，共5000 REAL + 5000 FAKE
BATCH_SIZE = 1       # CPU模式单张生成
N_INFERENCE_STEPS = 20

# CIFAKE 类别名
CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


def get_pipe():
    """加载 SD 模型（CPU卸载模式，不一次性全部加载到内存）"""
    print("加载 Stable Diffusion v1.5 (CPU 卸载模式)...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        str(SD_CACHE),
        local_files_only=True,
        torch_dtype=torch.float32,
        safety_checker=None,
        use_safetensors=True,
    )
    # CPU 卸载：按需加载各组件
    pipe.enable_sequential_cpu_offload()
    print("模型加载完成（CPU卸载模式）")
    return pipe


def load_cifar10():
    """加载 CIFAR-10"""
    transform = lambda x: Image.fromarray(x)
    train_set = datasets.CIFAR10(root=str(ROOT_DIR / "data" / "raw"), train=True, download=False)
    print(f"CIFAR-10 加载完成：{len(train_set)} 张")
    return train_set


def generate_batch(pipe, images, labels, prompt_suffix=""):
    """批量生成 FAKE 图像"""
    fake_imgs = []
    prompts = [f"a high quality photo of a {CLASS_NAMES[l]}" for l in labels]
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        result = pipe(
            prompt=prompt + prompt_suffix,
            image=img,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=N_INFERENCE_STEPS,
        )
        fake_imgs.append(result.images[0])
        # 每张处理完立即释放显存/CPU缓存
        gc.collect()
    return fake_imgs


def main():
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    # 检查已生成数量
    existing = len(list(FAKE_DIR.glob("*.jpg"))) + len(list(FAKE_DIR.glob("*.png")))
    print(f"已有 FAKE 图像：{existing} 张")

    if existing >= N_PER_CLASS * 10:
        print("FAKE 图像已全部生成完成！")
        return

    # 加载 CIFAR-10
    train_set = load_cifar10()

    # 加载 SD 模型
    pipe = get_pipe()

    # 按类别分组
    class_indices = {c: [] for c in range(10)}
    for idx in range(len(train_set)):
        img, label = train_set[idx]
        if len(class_indices[label]) < N_PER_CLASS:
            class_indices[label].append((idx, img))

    total_generated = 0
    start_time = time.time()

    for class_idx in range(10):
        indices = class_indices[class_idx]
        if not indices:
            continue

        prompt = f"a high quality photo of a {CLASS_NAMES[class_idx]}"
        print(f"\n[Class {class_idx}] '{CLASS_NAMES[class_idx]}' - {len(indices)} 张")

        for i, (idx, img) in enumerate(indices):
            # 检查是否已生成
            out_file = FAKE_DIR / f"fake_{idx:07d}.jpg"
            if out_file.exists():
                continue

            result = pipe(
                prompt=prompt,
                image=img,
                strength=0.75,
                guidance_scale=7.5,
                num_inference_steps=N_INFERENCE_STEPS,
            )
            result.images[0].save(out_file)
            total_generated += 1

            # 每10张打印进度
            if total_generated % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_generated / elapsed
                remain = ((N_PER_CLASS * 10) - total_generated) / rate / 60
                print(f"  已生成 {total_generated} 张, 速率 {rate:.1f}/s, 预计剩余 {remain:.1f}min")

            gc.collect()

    print(f"\n完成！共生成 {total_generated} 张FAKE图像，耗时 {(time.time()-start_time)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
