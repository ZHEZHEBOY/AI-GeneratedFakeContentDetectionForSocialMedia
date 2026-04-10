"""
频域特征工程模块
核心方法：
  1. FFT（快速傅里叶变换）频域统计特征
  2. 小波变换（Daubechies）多尺度频域特征
  3. 高频能量比（AI合成图像的核心判别特征）
  4. PCA 降维

核心依据：
  AI合成图像（GAN/扩散模型）在频域中表现出：
  - 高频成分异常（GAN频谱伪影）
  - 频谱不均匀性（与自然图像统计分布不同）
  - 方位能量不对称
"""

import numpy as np
import pywt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
from tqdm import tqdm


# ============================================================
# FFT 频域特征提取
# ============================================================
def extract_fft_features(image: np.ndarray) -> np.ndarray:
    """
    从单张灰度图像提取 FFT 频域统计特征。

    特征列表（共 17 维）：
      频谱整体统计：均值、方差、偏度、峰度
      频段能量比：低频/中频/高频能量占比
      高频比：高频能量 / 总能量
      径向功率谱：从中心到边缘的功率衰减斜率
      频谱熵
      相位谱统计：均值、方差
      方位能量不对称度

    Args:
        image: 灰度图像，归一化到 [0,1]，shape (H, W)

    Returns:
        features: np.ndarray, shape (17,)
    """
    H, W = image.shape

    # 2D FFT，中心化频谱
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    power = magnitude ** 2

    # --- 1. 频谱整体统计 ---
    log_power = np.log1p(power.ravel())
    feat_mean = np.mean(log_power)
    feat_var  = np.var(log_power)
    feat_skew = float(stats.skew(log_power))
    feat_kurt = float(stats.kurtosis(log_power))

    # --- 2. 频段能量比 ---
    cy, cx = H // 2, W // 2
    # 构建径向距离矩阵
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = np.sqrt(cy ** 2 + cx ** 2)
    r_norm = r / r_max  # 归一化到 [0, 1]

    total_energy = power.sum() + 1e-10
    low_energy  = power[r_norm < 0.2].sum()   / total_energy  # 低频 (<20%)
    mid_energy  = power[(r_norm >= 0.2) & (r_norm < 0.5)].sum() / total_energy  # 中频
    high_energy = power[r_norm >= 0.5].sum()  / total_energy  # 高频 (>50%)

    # --- 3. 高频比（最重要的 AI 检测特征）---
    hf_ratio = power[r_norm >= 0.7].sum() / total_energy

    # --- 4. 径向功率谱斜率（log-log回归） ---
    n_bins = 20
    radial_power = np.zeros(n_bins)
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if mask.sum() > 0:
            radial_power[i] = power[mask].mean()
    # log-log 线性回归求斜率
    freq_bins = np.arange(1, n_bins + 1)
    valid = radial_power > 0
    if valid.sum() >= 3:
        log_freq = np.log(freq_bins[valid])
        log_pow  = np.log(radial_power[valid] + 1e-10)
        slope, _ = np.polyfit(log_freq, log_pow, 1)
    else:
        slope = 0.0

    # --- 5. 频谱熵 ---
    p_norm = power / total_energy
    # 避免 log(0)
    p_norm = np.clip(p_norm, 1e-10, 1.0)
    spectral_entropy = -np.sum(p_norm * np.log(p_norm))
    # 归一化
    spectral_entropy /= np.log(H * W)

    # --- 6. 相位谱统计 ---
    phase = np.angle(fft_shift).ravel()
    phase_mean = np.mean(phase)
    phase_var  = np.var(phase)

    # --- 7. 方位能量不对称（水平 vs 垂直 vs 对角） ---
    # 水平方向：|y - cy| < 5
    horiz_mask = np.abs(yy - cy) < max(H // 16, 3)
    vert_mask  = np.abs(xx - cx) < max(W // 16, 3)
    horiz_energy = power[horiz_mask].sum() / total_energy
    vert_energy  = power[vert_mask].sum() / total_energy
    orient_asym = abs(horiz_energy - vert_energy)

    features = np.array([
        feat_mean, feat_var, feat_skew, feat_kurt,    # 4
        low_energy, mid_energy, high_energy,          # 3
        hf_ratio,                                     # 1
        slope,                                        # 1
        spectral_entropy,                             # 1
        phase_mean, phase_var,                        # 2
        horiz_energy, vert_energy, orient_asym,       # 3
        radial_power.mean(), radial_power.std(),      # 2
    ], dtype=np.float32)

    return features  # 共 17 维


# ============================================================
# 小波变换特征提取
# ============================================================
def extract_wavelet_features(
    image: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> np.ndarray:
    """
    多尺度小波分解频域特征。

    对每一层的 cH/cV/cD 子带提取统计特征：
    均值绝对值、方差、能量、熵（每子带 4 维）
    共 level × 3子带 × 4统计 = 36 维

    Args:
        image:   灰度图像 [0,1], shape (H, W)
        wavelet: 小波基（db4 适合捕捉高频伪影）
        level:   分解层数

    Returns:
        features: np.ndarray, shape (level * 12,)
    """
    features = []
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    # coeffs[0]: 低频近似，coeffs[1..level]: (cH, cV, cD) 细节子带

    for i in range(1, level + 1):
        cH, cV, cD = coeffs[i]
        for subband in [cH, cV, cD]:
            arr = subband.ravel()
            abs_mean = np.mean(np.abs(arr))
            variance = np.var(arr)
            energy   = np.mean(arr ** 2)
            # 归一化小波熵
            p = arr ** 2
            total = p.sum() + 1e-10
            p_norm = p / total
            p_norm = np.clip(p_norm, 1e-10, 1)
            entropy = -np.sum(p_norm * np.log(p_norm)) / np.log(len(p_norm))
            features.extend([abs_mean, variance, energy, entropy])

    return np.array(features, dtype=np.float32)  # 3层 × 3子带 × 4 = 36 维


# ============================================================
# 合并特征
# ============================================================
def extract_all_features(image: np.ndarray) -> np.ndarray:
    """
    提取图像的全部频域特征（FFT + 小波），拼接为一个向量。

    输出维度：17（FFT）+ 36（小波）= 53 维
    """
    fft_feat = extract_fft_features(image)
    wav_feat = extract_wavelet_features(image, wavelet="db4", level=3)
    return np.concatenate([fft_feat, wav_feat])


# ============================================================
# 批量特征提取
# ============================================================
def batch_extract_features(images: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    """
    批量提取图像频域特征。

    Args:
        images: shape (N, H, W)，归一化灰度图像
        n_jobs: 并行核数（-1=全部）

    Returns:
        features: shape (N, D)
    """
    print(f"[INFO] 开始提取频域特征（{len(images)} 张图像，FFT+小波，共53维）...")

    def _extract_one(img):
        return extract_all_features(img)

    if n_jobs == 1 or len(images) < 100:
        features = [_extract_one(img) for img in tqdm(images, desc="特征提取")]
    else:
        features = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
            joblib.delayed(_extract_one)(img)
            for img in tqdm(images, desc="特征提取（并行）")
        )

    features = np.array(features, dtype=np.float32)
    print(f"[INFO] 特征提取完成：shape = {features.shape}")
    return features


# ============================================================
# 标准化 + PCA 降维
# ============================================================
def normalize_and_reduce(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 20,
    save_dir: Path = None,
) -> tuple:
    """
    对训练集拟合 StandardScaler + PCA，并变换测试集。

    Args:
        X_train:      训练集特征，shape (N_train, D)
        X_test:       测试集特征，shape (N_test, D)
        n_components: PCA 目标维度
        save_dir:     保存 scaler/pca 模型的目录

    Returns:
        X_train_pca, X_test_pca, explained_variance_ratio
    """
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    evr = pca.explained_variance_ratio_
    cumvar = evr.cumsum()
    print(f"[INFO] PCA {n_components} 维，累计解释方差：{cumvar[-1]:.4f}")
    print(f"[INFO] 各主成分方差贡献（前10）：{evr[:10].round(4)}")

    # 保存模型
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_dir / "scaler.pkl")
        joblib.dump(pca,    save_dir / "pca.pkl")
        print(f"[INFO] Scaler 和 PCA 模型已保存至：{save_dir}")

    return X_train_pca, X_test_pca, evr


def load_normalizer(model_dir: Path):
    """加载已训练的 scaler 和 pca"""
    scaler = joblib.load(model_dir / "scaler.pkl")
    pca    = joblib.load(model_dir / "pca.pkl")
    return scaler, pca


# ============================================================
# 特征命名（用于可视化）
# ============================================================
FFT_FEATURE_NAMES = [
    "fft_log_mean", "fft_log_var", "fft_log_skew", "fft_log_kurt",
    "fft_low_energy", "fft_mid_energy", "fft_high_energy",
    "fft_hf_ratio",
    "fft_radial_slope",
    "fft_spectral_entropy",
    "fft_phase_mean", "fft_phase_var",
    "fft_horiz_energy", "fft_vert_energy", "fft_orient_asym",
    "fft_radial_mean", "fft_radial_std",
]

WAVELET_FEATURE_NAMES = [
    f"wav_L{level+1}_{sub}_{stat}"
    for level in range(3)
    for sub in ["cH", "cV", "cD"]
    for stat in ["abs_mean", "var", "energy", "entropy"]
]

ALL_FEATURE_NAMES = FFT_FEATURE_NAMES + WAVELET_FEATURE_NAMES


# ============================================================
# 主程序（独立测试）
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_preprocessing import load_processed, PROCESSED_DIR
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("频域特征工程测试")
    print("=" * 60)

    # 加载数据
    images, labels = load_processed("dataset")
    print(f"数据集：{images.shape}，标签分布：真实={(labels==0).sum()}, AI合成={(labels==1).sum()}")

    # 提取特征
    features = batch_extract_features(images, n_jobs=4)
    print(f"\n特征维度：{features.shape}")
    print(f"特征名称（{len(ALL_FEATURE_NAMES)}维）：{ALL_FEATURE_NAMES[:5]}...")

    # 检查特征区分度（均值差异）
    print("\n关键特征差异（真实 vs AI合成）：")
    real_feat = features[labels == 0]
    fake_feat = features[labels == 1]
    for i, name in enumerate(FFT_FEATURE_NAMES[:8]):
        print(f"  {name}: 真实={real_feat[:, i].mean():.4f}, AI合成={fake_feat[:, i].mean():.4f}")

    # 保存特征
    feat_dir = Path(__file__).parent.parent / "data" / "features"
    feat_dir.mkdir(exist_ok=True)
    np.save(feat_dir / "features.npy", features)
    np.save(feat_dir / "labels.npy", labels)
    print(f"\n[✓] 特征已保存至 {feat_dir}")

    # PCA 降维测试
    X_tr, X_te, y_tr, y_te = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_tr_pca, X_te_pca, evr = normalize_and_reduce(
        X_tr, X_te, n_components=20,
        save_dir=feat_dir / "preprocessors"
    )
    print(f"\n降维后：训练={X_tr_pca.shape}，测试={X_te_pca.shape}")
