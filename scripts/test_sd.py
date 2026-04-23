"""快速测试 SD 模型是否可用"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np

print("Step 1: 加载模型...")
cache = r'C:\Users\FAREWELL\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14'
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    cache,
    torch_dtype=torch.float32,
    safety_checker=None,
)
pipe = pipe.to('cpu')
print("模型加载成功！")

print("Step 2: 测试生成...")
test_img = Image.fromarray(
    np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
)
result = pipe(
    prompt="a colorful bird",
    image=test_img,
    num_inference_steps=10,
    guidance_scale=7.5,
)
out_path = r"d:\WorkBuddySpace\AI-GeneratedFakeContentDetectionForSocialMedia\data\test_sd_output.png"
result.images[0].save(out_path)
print(f"测试生成成功！图像已保存: {out_path}")
