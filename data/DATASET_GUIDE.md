# 数据集下载指引

本项目推荐使用以下公开数据集，按**推荐程度**排序。

---

## ✅ 首选：CIFAKE（最易获取，120,000张）

| 项目 | 说明 |
|------|------|
| 规模 | 120,000张（60,000真实 + 60,000 AI合成） |
| 来源 | CIFAR-10真实图像 + Stable Diffusion v1.4生成 |
| 大小 | 约 1.3 GB |
| 获取难度 | ⭐ 简单（Kaggle公开下载，无需申请） |
| 适合场景 | 快速实验、频域特征验证 |

### 方法一：Kaggle 网页下载（推荐新手）

1. 注册/登录 Kaggle：https://www.kaggle.com
2. 访问数据集页面：
   ```
   https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
   ```
3. 点击右上角 **"Download"** 按钮，下载 `archive.zip`（约1.3GB）
4. 解压后将 `train/` 和 `test/` 文件夹放入：
   ```
   data/raw/cifake/
   ├── train/
   │   ├── REAL/   (50,000张)
   │   └── FAKE/   (50,000张)
   └── test/
       ├── REAL/   (10,000张)
       └── FAKE/   (10,000张)
   ```

### 方法二：Kaggle CLI 命令行下载（推荐有Python环境）

```bash
# 第1步：安装 kaggle 工具
pip install kaggle

# 第2步：配置 API Key（从 https://www.kaggle.com/settings 生成并下载）
# 将 kaggle.json 放到：
# Windows: C:\Users\你的用户名\.kaggle\kaggle.json
# Linux/Mac: ~/.kaggle/kaggle.json

# 第3步：下载并解压
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data/raw/cifake --unzip
```

---

## ✅ 次选：GenImage（百万级，HuggingFace可下载）

| 项目 | 说明 |
|------|------|
| 规模 | 268万张（133万真实 + 135万AI合成） |
| 来源 | ImageNet真实图像 + 8种生成模型（Midjourney/SD/DALL-E等） |
| 大小 | 约 500 GB（建议选取子集） |
| 获取难度 | ⭐⭐ 中等（HuggingFace，部分需申请） |
| 适合场景 | 论文级实验，结果更有说服力 |

### HuggingFace 下载（子集）

```python
from datasets import load_dataset

# 只下载一个子集（如midjourney生成的）
ds = load_dataset("jzousz/GenImage", split="train[:5000]")
```

---

## ✅ 补充：Deepfake-Eval-2024（社交媒体真实场景）

| 项目 | 说明 |
|------|------|
| 规模 | 约 5,000 条真实社交媒体内容 |
| 来源 | 社交媒体平台真实流传的AI伪造内容 |
| 获取 | 需填表申请：https://www.truemedia.org |
| 适合场景 | 社交媒体场景真实性验证 |

---

## 推荐策略

对于本次**统计建模大赛**，推荐：

```
CIFAKE (120,000张) 作为主数据集
├── 训练集：100,000张（train/REAL + train/FAKE）
└── 测试集：20,000张（test/REAL + test/FAKE）
```

**理由：**
- 数据规模适中，实验效率高
- 标签清晰（CIFAR-10真实图像 vs Stable Diffusion合成图像）
- 与"社交媒体AI合成虚假内容检测"主题高度契合
- Kaggle公开，无需申请，下载方便

---

## 下载后验证

下载解压后，运行以下命令验证：
```bash
python src/data_preprocessing.py --dataset cifake
```
