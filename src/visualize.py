"""

可视化模块
生成论文级图表：
  1. 频域特征分布对比（真实 vs AI合成）
  2. PCA 降维可视化（2D散点图）
  3. 训练曲线（Loss / Val Acc）
  4. 模型性能对比柱状图
  5. ROC 曲线对比
  6. 混淆矩阵
  7. FFT 频谱可视化示例
  8. MC Dropout 不确定性分布
  9. 消融实验结果图
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

# 中文字体配置（Windows）
rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 150

ROOT_DIR   = Path(__file__).parent.parent
RESULT_DIR = ROOT_DIR / "results" / "figures"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 统一配色
COLOR_REAL = "#2196F3"   # 蓝色 = 真实图像
COLOR_FAKE = "#F44336"   # 红色 = AI合成图像
COLOR_LIST = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


# ============================================================
# 1. FFT 频谱可视化
# ============================================================
def plot_fft_spectrum_examples(images: np.ndarray, labels: np.ndarray, n_examples: int = 3, save: bool = True):
    """展示真实图像与AI合成图像的FFT频谱对比"""
    fig, axes = plt.subplots(2, n_examples * 2, figsize=(n_examples * 6, 5))
    fig.suptitle("FFT 频谱对比：真实图像 vs AI 合成图像", fontsize=14, fontweight="bold")

    real_idx = np.where(labels == 0)[0][:n_examples]
    fake_idx = np.where(labels == 1)[0][:n_examples]

    for col, (r_i, f_i) in enumerate(zip(real_idx, fake_idx)):
        for row_offset, (idx, label_name, color) in enumerate([
            (r_i, "真实", COLOR_REAL), (f_i, "AI合成", COLOR_FAKE)
        ]):
            img = images[idx]
            fft = np.fft.fft2(img)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.log1p(np.abs(fft_shift))

            # 原图
            ax_img = axes[row_offset][col * 2]
            ax_img.imshow(img, cmap="gray")
            ax_img.set_title(f"{label_name}图像", color=color, fontsize=10)
            ax_img.axis("off")

            # 频谱
            ax_fft = axes[row_offset][col * 2 + 1]
            im = ax_fft.imshow(magnitude, cmap="hot")
            ax_fft.set_title(f"{label_name} FFT频谱", color=color, fontsize=10)
            ax_fft.axis("off")
            plt.colorbar(im, ax=ax_fft, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save:
        path = RESULT_DIR / "fft_spectrum_comparison.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"[✓] 保存：{path}")
    plt.close()


# ============================================================
# 2. 关键频域特征分布对比
# ============================================================
def plot_feature_distributions(features: np.ndarray, labels: np.ndarray,
                                feature_names: list = None, top_n: int = 8, save: bool = True):
    """绘制关键特征的分布密度图（真实 vs AI合成）"""
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(features.shape[1])]

    # 选取区分度最高的特征（按均值差异排序）
    real = features[labels == 0]
    fake = features[labels == 1]
    diffs = np.abs(real.mean(0) - fake.mean(0)) / (features.std(0) + 1e-8)
    top_idx = np.argsort(diffs)[-top_n:][::-1]

    cols = 4
    rows = (top_n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.ravel()
    fig.suptitle("关键频域特征分布对比（真实 vs AI 合成）", fontsize=13, fontweight="bold")

    for i, feat_idx in enumerate(top_idx):
        ax = axes[i]
        sns.kdeplot(real[:, feat_idx], ax=ax, color=COLOR_REAL, label="真实", fill=True, alpha=0.4)
        sns.kdeplot(fake[:, feat_idx], ax=ax, color=COLOR_FAKE, label="AI合成", fill=True, alpha=0.4)
        ax.set_title(feature_names[feat_idx], fontsize=9)
        ax.set_xlabel("")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save:
        path = RESULT_DIR / "feature_distribution.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"[✓] 保存：{path}")
    plt.close()


# ============================================================
# 3. PCA 2D 降维可视化
# ============================================================
def plot_pca_scatter(features: np.ndarray, labels: np.ndarray, save: bool = True):
    """PCA 降维到2D的散点图"""
    pca2d = PCA(n_components=2, random_state=42)
    X_2d  = pca2d.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_real = ax.scatter(
        X_2d[labels == 0, 0], X_2d[labels == 0, 1],
        c=COLOR_REAL, alpha=0.4, s=8, label="真实图像"
    )
    scatter_fake = ax.scatter(
        X_2d[labels == 1, 0], X_2d[labels == 1, 1],
        c=COLOR_FAKE, alpha=0.4, s=8, label="AI 合成图像"
    )
    ax.set_title("频域特征 PCA 二维可视化", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1（解释方差 {pca2d.explained_variance_ratio_[0]:.2%}）")
    ax.set_ylabel(f"PC2（解释方差 {pca2d.explained_variance_ratio_[1]:.2%}）")
    ax.legend(markerscale=3, fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = RESULT_DIR / "pca_scatter.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"[✓] 保存：{path}")
    plt.close()


# ============================================================
# 4. 训练曲线
# ============================================================
def plot_training_curve(history: dict, save: bool = True):
    """绘制训练 Loss 和验证 Acc 曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MC-RGCN 训练过程", fontsize=13, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], color="#2196F3", linewidth=1.5)
    ax1.set_title("训练损失（Cross-Entropy Loss）")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_acc"], color="#4CAF50", linewidth=1.5, label="Val Accuracy")
    if "val_f1" in history:
        ax2.plot(epochs, history["val_f1"], color="#FF9800", linewidth=1.5, linestyle="--", label="Val F1")
    ax2.set_title("验证集性能")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = RESULT_DIR / "training_curve.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"[✓] 保存：{path}")
    plt.close()


# ============================================================
# 5. 模型性能对比柱状图
# ============================================================
def plot_model_comparison(results: dict, metrics: list = None, save: bool = True):
    """绘制各模型在多个指标上的对比柱状图"""
    if metrics is None:
        metrics = ["accuracy", "f1", "precision", "recall", "auc"]

    # 过滤掉不存在的指标
    model_names = list(results.keys())
    metrics = [m for m in metrics if any(m in v for v in results.values())]

    x = np.arange(len(model_names))
    width = 0.8 / len(metrics)
    colors = COLOR_LIST[:len(metrics)]

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.8), 6))

    for i, metric in enumerate(metrics):
        vals = [results[m].get(metric, 0) for m in model_names]
        bars = ax.bar(x + i * width - (len(metrics) - 1) * width / 2,
                      vals, width * 0.9, label=metric.upper(), color=colors[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([n.split("（")[0] for n in model_names], rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("各模型性能对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    # 标注主模型
    mc_idx = next((i for i, n in enumerate(model_names) if "RGCN" in n), None)
    if mc_idx is not None:
        ax.axvspan(mc_idx - 0.45, mc_idx + 0.45, alpha=0.08, color="gold")
        ax.text(mc_idx, 1.08, "★主模型", ha="center", fontsize=9, color="#FF8F00")

    plt.tight_layout()
    if save:
        path = RESULT_DIR / "model_comparison.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"[✓] 保存：{path}")
    plt.close()


# ============================================================
# 6. ROC 曲线
# ============================================================
def plot_roc_curves(roc_data: dict, save: bool = True):
    """
    roc_data: {model_name: (y_true, y_prob)}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)

    for i, (name, (y_true, y_prob)) in enumerate(roc_data.items()):
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        lw = 2.5 if "RGCN" in name else 1.2
        ls = "-" if "RGCN" in name else "--"
        ax.plot(fpr, tpr, color=COLOR_LIST[i % len(COLOR_LIST)],
                lw=lw, ls=ls, label=f"{name.split('（')[0]}（AUC={roc_auc:.3f}）")

    ax.set_xlabel("假阳率（FPR）", fontsize=11)
    ax.set_ylabel("真阳率（TPR）", fontsize=11)
    ax.set_title("各模型 ROC 曲线对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    if save:
        path = RESULT_DIR / "roc_curves.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"[✓] 保存：{path}")
    plt.close()


# ============================================================
# 7. 混淆矩阵
# ============================================================
def plot_confusion_matrix(cm: list, model_name: str = "MC-RGCN", save: bool = True):
    """绘制混淆矩阵热力图"""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=["真实", "AI合成"],
        yticklabels=["真实", "AI合成"],
        ax=ax, linewidths=0.5
    )
    ax.set_title(f"{model_name} 混淆矩阵", fontsize=12, fontweight="bold")
    ax.set_xlabel("预测标签", fontsize=10)
    ax.set_ylabel("真实标签", fontsize=10)

    plt.tight_layout()
    if save:
        safe_name = model_name.replace("/", "_").replace(" ", "_")
        path = RESULT_DIR / f"confusion_matrix_{safe_name}.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"[✓] 保存：{path}")
    plt.close()


# ============================================================
# 8. MC Dropout 不确定性分布
# ============================================================
def plot_mc_uncertainty(uncertainty: np.ndarray, labels: np.ndarray, save: bool = True):
    """绘制 MC Dropout 不确定性（预测熵）的分布"""
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.kdeplot(uncertainty[labels == 0], ax=ax, color=COLOR_REAL,
                label="真实图像", fill=True, alpha=0.4)
    sns.kdeplot(uncertainty[labels == 1], ax=ax, color=COLOR_FAKE,
                label="AI合成图像", fill=True, alpha=0.4)

    ax.set_title("MC Dropout 预测不确定性分布", fontsize=12, fontweight="bold")
    ax.set_xlabel("预测熵（不确定性）", fontsize=10)
    ax.set_ylabel("密度", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = RESULT_DIR / "mc_uncertainty.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"[✓] 保存：{path}")
    plt.close()


# ============================================================
# 9. 消融实验结果图
# ============================================================
def plot_ablation_study(ablation_results: dict, save: bool = True):
    """
    ablation_results: {
      "无FFT特征":       {"accuracy": ..., "f1": ...},
      "无小波特征":       {"accuracy": ..., "f1": ...},
      "无MC-Dropout":    {"accuracy": ..., "f1": ...},
      "单关系图":         {"accuracy": ..., "f1": ...},
      "完整MC-RGCN":     {"accuracy": ..., "f1": ...},
    }
    """
    names  = list(ablation_results.keys())
    accs   = [ablation_results[n]["accuracy"] for n in names]
    f1s    = [ablation_results[n]["f1"]       for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1s,  width, label="F1",       color="#4CAF50", alpha=0.85)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    # 标注完整模型
    full_idx = len(names) - 1
    ax.axvspan(full_idx - 0.45, full_idx + 0.45, alpha=0.08, color="gold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("消融实验：各模块贡献分析", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        path = RESULT_DIR / "ablation_study.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"[✓] 保存：{path}")
    plt.close()


# ============================================================
# 10. 径向功率谱对比（论文级图）
# ============================================================
def plot_radial_power_spectrum(images: np.ndarray, labels: np.ndarray,
                                n_samples: int = 200, save: bool = True):
    """绘制真实图像与AI合成图像的平均径向功率谱"""
    real_idx = np.where(labels == 0)[0][:n_samples]
    fake_idx = np.where(labels == 1)[0][:n_samples]

    def mean_radial_spectrum(imgs):
        H, W = imgs[0].shape
        cy, cx = H // 2, W // 2
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
        r_max = int(np.sqrt(cy ** 2 + cx ** 2))
        all_spectra = []
        for img in imgs:
            fft = np.fft.fft2(img)
            fft_shift = np.fft.fftshift(fft)
            power = np.abs(fft_shift) ** 2
            spectrum = np.array([power[r == ri].mean() if (r == ri).any() else 0
                                  for ri in range(r_max)])
            all_spectra.append(spectrum)
        return np.array(all_spectra).mean(axis=0)

    real_spec = mean_radial_spectrum(images[real_idx])
    fake_spec = mean_radial_spectrum(images[fake_idx])
    freqs = np.arange(len(real_spec))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("径向功率谱分析：真实图像 vs AI 合成图像", fontsize=13, fontweight="bold")

    # 线性坐标
    ax1.plot(freqs, real_spec, color=COLOR_REAL, label="真实图像", linewidth=1.5)
    ax1.plot(freqs, fake_spec, color=COLOR_FAKE, label="AI合成图像", linewidth=1.5)
    ax1.set_title("线性坐标")
    ax1.set_xlabel("空间频率（像素）")
    ax1.set_ylabel("功率")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 对数坐标（更清晰显示频率差异）
    ax2.semilogy(freqs[1:], real_spec[1:], color=COLOR_REAL, label="真实图像", linewidth=1.5)
    ax2.semilogy(freqs[1:], fake_spec[1:], color=COLOR_FAKE, label="AI合成图像", linewidth=1.5)
    ax2.set_title("对数坐标（显示高频伪影）")
    ax2.set_xlabel("空间频率（像素）")
    ax2.set_ylabel("功率（对数）")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = RESULT_DIR / "radial_power_spectrum.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"[✓] 保存：{path}")
    plt.close()


# ============================================================
# 一键生成所有图表
# ============================================================
def generate_all_figures(
    images: np.ndarray = None,
    labels_raw: np.ndarray = None,
    features: np.ndarray = None,
    labels: np.ndarray = None,
    history: dict = None,
    results: dict = None,
):
    """一键生成所有可视化图表"""
    from src.feature_extraction import ALL_FEATURE_NAMES

    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)

    if images is not None and labels_raw is not None:
        print("[1/5] FFT 频谱示例...")
        plot_fft_spectrum_examples(images, labels_raw)
        print("[2/5] 径向功率谱...")
        plot_radial_power_spectrum(images, labels_raw)

    if features is not None and labels is not None:
        print("[3/5] 特征分布对比...")
        plot_feature_distributions(features, labels, ALL_FEATURE_NAMES)
        print("[4/5] PCA 散点图...")
        plot_pca_scatter(features, labels)

    if history is not None:
        print("[5/5] 训练曲线...")
        plot_training_curve(history)

    if results is not None:
        print("[+] 模型对比柱状图...")
        plot_model_comparison(results)

    print(f"\n[✓] 所有图表已保存至 {RESULT_DIR}")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT_DIR))

    # 加载数据
    feat_dir = ROOT_DIR / "data" / "features"

    print("加载特征数据...")
    features = np.load(feat_dir / "features.npy")
    labels   = np.load(feat_dir / "labels.npy")

    # 加载训练历史
    hist_path = ROOT_DIR / "results" / "tables" / "training_history.json"
    history = None
    if hist_path.exists():
        with open(hist_path, encoding="utf-8") as f:
            history = json.load(f)

    # 加载结果
    result_path = ROOT_DIR / "results" / "tables" / "model_comparison.json"
    results = None
    if result_path.exists():
        with open(result_path, encoding="utf-8") as f:
            results = json.load(f)

    # 加载图像（用于频谱可视化，取前500张节省内存）
    try:
        from src.data_preprocessing import load_processed
        images, labels_raw = load_processed("dataset")
        images = images[:500]
        labels_raw = labels_raw[:500]
    except Exception:
        images, labels_raw = None, None
        print("[WARN] 未能加载原始图像，跳过频谱可视化")

    generate_all_figures(
        images=images, labels_raw=labels_raw,
        features=features, labels=labels,
        history=history, results=results,
    )

    # 如果有模型结果，画混淆矩阵
    if results and "MC-RGCN（主模型）" in results:
        cm = results["MC-RGCN（主模型）"].get("confusion_matrix")
        if cm:
            plot_confusion_matrix(cm, "MC-RGCN")

    # 消融实验示例（实际训练后替换为真实数值）
    ablation_demo = {
        "无FFT特征":    {"accuracy": 0.82, "f1": 0.81},
        "无小波特征":   {"accuracy": 0.84, "f1": 0.83},
        "无MC-Dropout": {"accuracy": 0.86, "f1": 0.85},
        "单关系图":     {"accuracy": 0.87, "f1": 0.86},
        "完整MC-RGCN":  {"accuracy": 0.91, "f1": 0.90},
    }
    plot_ablation_study(ablation_demo)
    print("\n[✓] 可视化全部完成！")
