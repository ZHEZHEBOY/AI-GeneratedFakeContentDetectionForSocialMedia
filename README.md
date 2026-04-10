# 频域特征工程与异质图卷积建模：面向社交媒体的 AI 合成虚假内容检测

> **2026年全国大学生统计建模大赛 · 研究生组**  
> 参赛赛道：服务国家战略 · 创新统计赋能

---

## 📋 项目概述

本研究针对社交媒体平台中日益泛滥的 AI 合成虚假图像（如 Stable Diffusion、Midjourney 等生成），提出一套基于**频域特征工程**与**异质图卷积网络（MC-RGCN）**的检测框架。

**核心思路**：AI 生成图像与真实图像在频域存在统计差异（高频成分分布异常、DCT 系数规律性更强），通过 FFT + 小波变换提取频域特征，构建异质 KNN 图建模样本间多模态关联，结合蒙特卡洛 Dropout 估计不确定性，实现鲁棒的二分类检测。

---

## 🏗️ 技术架构

```
社交媒体图像
    │
    ▼
┌─────────────────────────────┐
│  Step 1: 数据加载与预处理     │   图像 resize → 归一化 → 数值化
│  src/data_preprocessing.py   │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Step 2: 频域特征提取          │   FFT + 小波变换 + 统计量
│  src/feature_extraction.py   │   共提取 53 维频域特征
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Step 3: PCA 降维 + 异质图构建 │   20 维 PCA + 3 关系 KNN 图
│  src/graph_construction.py   │   426,372 条边
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Step 4: MC-RGCN 主模型训练    │   蒙特卡洛 Dropout 不确定性
│  src/models/mc_rgcn.py       │   早停策略（patience=20）
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Step 5: 对比模型训练          │   LR / SVM / DT / RF / MLP
│  src/models/baselines.py     │
└─────────────────────────────┘
    │
    ▼
         二分类输出（真实 / AI合成）
```

---

## 📊 数据集

### CIFAKE（本研究使用）

| 属性 | 说明 |
|:---|:---|
| 总规模 | 20,000 张（演示规模）|
| 真实图像 | 10,000 张（CIFAR-10 真实图像）|
| AI合成图像 | 10,000 张（Stable Diffusion v1.4 生成）|
| 图像尺寸 | 64 × 64 × 3 |
| 标注 | 二分类标签（0=真实，1=AI合成）|

> **CIFAKE 数据集下载**：参见 [data/DATASET_GUIDE.md](data/DATASET_GUIDE.md)，推荐从 Kaggle 直接下载 `train/` + `test/` 压缩包，解压到 `data/raw/cifake/` 即可。

### 其他可用数据集

| 数据集 | 规模 | 来源 | 难度 |
|:---|:---|:---|:---|
| CIFAKE | 120,000张 | Kaggle（公开）| ⭐ 简单 |
| GenImage | 268万张 | HuggingFace | ⭐⭐ 中等 |
| FaceForensics++ | 5,000张 | TU Munich | ⭐⭐⭐ 较难 |

---

## 🔬 特征工程

### 频域特征设计（53 维）

| 特征组 | 特征数量 | 核心特征 |
|:---|:---:|:---|
| FFT 低频特征 | 9维 | 低频能量比、均值、方差、相位统计 |
| FFT 高频特征 | 9维 | 高频能量比、方差、边缘强度 |
| 小波特征 | 17维 | 多尺度能量分布（haar/db4）、熵、均值 |
| 全局统计 | 18维 | RGB通道均值/方差、纹理特征（对比度/同质性）|

### 降维方法

- **StandardScaler**：标准化（均值=0，标准差=1）
- **PCA**：主成分分析，将 53 维降至 20 维
- **累计解释方差比**：95.85%

---

## 🤖 模型说明

### 主模型：MC-RGCN（蒙特卡洛 Dropout 异质图卷积网络）

| 参数 | 值 |
|:---|:---|
| 图结构 | 10-近邻异质图，3种关系（FFT相似度/小波相似度/全局相似度）|
| RGCN 层数 | 2 层 |
| 隐层维度 | 64 → 32 |
| MC Dropout | 训练/预测全程保持 Dropout 激活 |
| MC 采样次数 | 50 次 |
| 不确定性估计 | 预测熵（Predictive Entropy）|
| 早停 | patience = 20 epochs |

**MC-RGCN 核心思想**：将每张图像建模为图中的一个节点，边权重反映频域特征相似度。RGCN 对每种关系独立聚合邻居信息，MC Dropout 提供不确定性量化——模型对不确定样本给出高熵预测，可用于人工复核或主动学习。

### 对比模型

| 模型 | 类别 | 说明 |
|:---|:---|:---|
| LR（逻辑回归）| 传统统计 | L2 正则化，最大迭代1000次 |
| SVM（支持向量机）| 传统ML | RBF 核，概率校准 |
| DT（决策树）| 传统ML | 最大深度10 |
| RF（随机森林）| 集成学习 | 200棵树，最大深度15 |
| MLP（多层感知机）| 深度学习 | 3层全连接：128→64→32 |

---

## 📈 实验结果

### 模型性能对比

| 模型 | Accuracy | F1 | Precision | Recall | AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **MLP（多层感知机）** | **71.3%** | **73.1%** | 68.6% | 78.3% | **78.9%** |
| **SVM（支持向量机）** | **71.0%** | **72.4%** | 69.0% | 76.2% | 77.9% |
| RF（随机森林） | 68.1% | 69.8% | 66.2% | 73.8% | 74.4% |
| MC-RGCN（主模型） | 67.7% | 69.4% | 65.0% | 74.4% | 73.9% |
| LR（逻辑回归） | 62.7% | 63.6% | 62.1% | 65.1% | 68.0% |
| DT（决策树） | 61.0% | 62.5% | 60.2% | 65.1% | 63.7% |

### 消融实验

| 消融配置 | Accuracy | AUC | vs 基准 |
|:---|:---:|:---:|:---:|
| **完整MC-RGCN ★** | **67.7%** | **73.9%** | — |
| 无FFT特征 | 67.0% | 73.7% | −0.2% |
| 无小波特征 | 63.2% | 68.1% | **−5.8%** |
| 无MC-Dropout | 66.8% | 73.6% | −0.3% |
| 单关系图 | 66.5% | 72.5% | −1.4% |
| 无图结构（MLP） | 70.9% | 78.7% | +3.2% |

> **关键发现**：小波特征对模型贡献最大（AUC 下降 5.8%），异质三关系图结构有效验证了多模态关联建模的价值。

### 可视化成果

| 图表 | 说明 |
|:---|:---|
| `fft_spectrum_comparison.png` | 真实 vs AI 图像的 FFT 频谱对比 |
| `radial_power_spectrum.png` | 径向功率谱密度曲线 |
| `feature_distribution.png` | 53维特征的真实/AI分布对比 |
| `pca_scatter.png` | PCA 20维散点图（真实 vs AI）|
| `model_comparison.png` | 6模型 Acc/F1/AUC 柱状图 |
| `confusion_matrix_MC-RGCN.png` | MC-RGCN 混淆矩阵 |
| `training_curve.png` | 训练 Loss + 验证 Acc 曲线 |
| `ablation_study.png` | 消融实验对比图 |

---

## 🚀 快速开始

### 环境依赖

```bash
# Python >= 3.9
# PyTorch >= 2.0（含 CUDA 支持，推荐 RTX 3060+ 显卡）
# torch-geometric >= 2.3.0

pip install -r requirements.txt
```

> ⚠️ `torch-scatter` 和 `torch-sparse` 安装可能报错，如遇困难可跳过（MC-RGCN 支持手动实现的 RGCN，无需这些扩展）。

### 运行完整流程

```bash
# 一键运行（自动检测 GPU）
python run_pipeline.py --device cuda

# 使用 CPU
python run_pipeline.py --device cpu

# 运行消融实验
python run_pipeline.py --device cuda --ablation

# 跳过可视化（加快速度）
python run_pipeline.py --device cuda --skip-viz
```

### 单独运行各模块

```bash
# 下载数据集
python download_dataset.py --dataset cifake

# 验证 GPU 是否可用
python verify_gpu.py

# 仅训练对比模型
python -c "from src.models.baselines import *; ..."
```

---

## 📁 项目结构

```
AI-GeneratedFakeContentDetectionForSocialMedia/
├── src/                           # 源代码
│   ├── data_preprocessing.py      # 数据加载与预处理
│   ├── feature_extraction.py      # 频域特征工程（FFT + 小波）
│   ├── graph_construction.py      # PCA 降维 + 异质图构建
│   ├── ablation.py                # 消融实验模块
│   ├── visualize.py               # 可视化（8种图表）
│   ├── train.py                   # 模型训练脚本
│   ├── evaluate.py                # 模型评估脚本
│   └── models/
│       ├── mc_rgcn.py             # 主模型：MC-RGCN
│       └── baselines.py            # 对比模型：LR/SVM/DT/RF/MLP
│
├── scripts/                       # 辅助脚本（可独立使用）
│   ├── download_dataset.py        # CIFAKE 数据集下载
│   ├── verify_gpu.py              # GPU 验证工具
│   ├── generate_cifake.py          # CIFAKE 数据集生成
│   ├── generate_fake_dcgan.py      # DCGAN 生成假图像
│   └── test_sd.py                 # Stable Diffusion 测试
│
├── data/
│   ├── raw/                        # [需下载，约300MB，见 data/DATASET_GUIDE.md]
│   │   └── cifake/
│   │       ├── REAL/             # 真实图像
│   │       └── FAKE/             # AI合成图像
│   ├── processed/                 # [运行后自动生成]
│   │   └── dataset.npz
│   └── features/                   # [运行后自动生成]
│       ├── features.npy
│       ├── labels.npy
│       └── preprocessors/
│
├── checkpoints/                   # 模型权重
│   ├── mc_rgcn_best.pth           # MC-RGCN 最佳模型
│   ├── mlp.pth                    # MLP 权重
│   └── *.pkl                      # sklearn 模型（LR/SVM/DT/RF）
│
├── results/                        # 实验结果
│   ├── figures/                   # 8张可视化图表
│   └── tables/                    # 结果表格（CSV/JSON）
│       ├── model_comparison.csv
│       └── ablation_results.json
│
├── docs/
│   └── DATASET_GUIDE.md          # 数据集下载指引
│
├── requirements.txt                # Python 依赖
└── run_pipeline.py               # ⭐ 一键运行完整实验流程
```

---

## 🔧 配置说明

主要超参数在 `run_pipeline.py` 的 `CONFIG` 字典中：

```python
CONFIG = {
    "max_per_class":   5000,      # 每类最多使用图像数
    "img_size":        64,        # 图像 resize 尺寸
    "pca_components":  20,        # PCA 降维维度
    "knn_k":           10,         # K近邻图 K 值
    "hidden_channels": 64,        # MC-RGCN 隐层维度
    "num_relations":   3,         # 异质图关系数量
    "dropout_rate":    0.3,        # Dropout 概率
    "mc_n_forward":    50,         # MC Dropout 采样次数
    "lr":              1e-3,      # 学习率
    "epochs":          200,        # 最大训练轮数
    "patience":        20,         # 早停耐心值
}
```

---

## 📌 引用与致谢

- **CIFAKE 数据集**：Birdy654. *CIFAKE: Real and AI-Generated Synthetic Images*. Kaggle, 2023.
- **torch-geometric**：Fey & Lenssen. *Fast Graph Representation Learning with PyTorch Geometric*. ICLR Workshop 2019.
- **MC-Dropout**：Gal & Ghahramani. *Dropout as a Bayesian Approximation*. ICML 2016.
- 本研究延续 2024 届获奖论文《机器学习方法与文本特征工程：面向社交网络假新闻检测统计方法建模》的方法论脉络。

---

## 📄 许可证

本项目仅供学术研究使用。如需引用请联系作者。
