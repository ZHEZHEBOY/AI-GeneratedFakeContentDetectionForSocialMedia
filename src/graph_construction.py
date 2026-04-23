"""

异质图构建模块
方法：基于降维后的频域特征矩阵，构建样本间K近邻相似度图
图类型：有向图 / 无向图，支持多关系边（异质图）

关系类型（RGCN 的多关系设计）：
  - rel_0: 高频特征相似（特征组1：FFT高频相关特征）
  - rel_1: 小波特征相似（特征组2：小波能量特征）
  - rel_2: 全局特征相似（全部特征）
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from pathlib import Path


# ============================================================
# 特征分组（用于构建多关系边）
# ============================================================
# 与 feature_extraction.py 中的特征顺序对应
# FFT特征 17维，小波特征 36维（降维后使用 PCA 投影特征）
# 此处按 PCA 后维度 n_components=20 进行分组
def get_feature_groups(n_components: int = 20) -> dict:
    """
    将 PCA 降维后的特征分成两组用于多关系边构建。
    前半：偏 FFT 信息；后半：偏小波信息
    """
    half = n_components // 2
    return {
        "fft_group":     list(range(0, half)),           # 前 10 维
        "wavelet_group": list(range(half, n_components)), # 后 10 维
        "all":           list(range(n_components)),       # 全部
    }


# ============================================================
# 构建 KNN 相似度图
# ============================================================
def build_knn_graph(
    features: np.ndarray,
    k: int = 10,
    feature_indices: list = None,
    symmetric: bool = True,
    threshold: float = None,
) -> tuple:
    """
    构建基于余弦相似度的 K 近邻图。

    Args:
        features:        特征矩阵，shape (N, D)
        k:               每个节点的近邻数
        feature_indices: 用哪些特征维度计算相似度（None=全部）
        symmetric:       是否对称化（无向图）
        threshold:       相似度阈值（None=不过滤）

    Returns:
        edge_index: torch.LongTensor, shape (2, E)
        edge_weight: torch.FloatTensor, shape (E,)
    """
    N = features.shape[0]

    if feature_indices is not None:
        feat = features[:, feature_indices]
    else:
        feat = features

    # L2 归一化后计算余弦相似度
    feat_norm = normalize(feat, norm="l2")
    sim_matrix = cosine_similarity(feat_norm)  # (N, N)
    np.fill_diagonal(sim_matrix, -1)            # 排除自环

    # 阈值过滤
    if threshold is not None:
        sim_matrix[sim_matrix < threshold] = -1

    # KNN：取每行 top-k
    src_list, dst_list, weight_list = [], [], []
    for i in range(N):
        top_k_idx = np.argsort(sim_matrix[i])[-k:]
        for j in top_k_idx:
            if sim_matrix[i, j] > 0:
                src_list.append(i)
                dst_list.append(j)
                weight_list.append(float(sim_matrix[i, j]))

    if symmetric:
        # 对称化：加入反向边
        src_ext = src_list + dst_list
        dst_ext = dst_list + src_list
        w_ext   = weight_list + weight_list
        # 去重
        seen = set()
        final_src, final_dst, final_w = [], [], []
        for s, d, w in zip(src_ext, dst_ext, w_ext):
            key = (min(s, d), max(s, d))
            if key not in seen:
                seen.add(key)
                final_src.append(s)
                final_dst.append(d)
                final_w.append(w)
                final_src.append(d)
                final_dst.append(s)
                final_w.append(w)
        src_list, dst_list, weight_list = final_src, final_dst, final_w

    edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)
    return edge_index, edge_weight


# ============================================================
# 构建多关系异质图（RGCN 输入格式）
# ============================================================
def build_heterogeneous_graph(
    features: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    n_components: int = 20,
) -> dict:
    """
    构建三种关系的异质图，供 RGCN 使用。

    Args:
        features:     PCA 降维后的特征矩阵，shape (N, n_components)
        labels:       标签，shape (N,)
        k:            KNN 近邻数
        n_components: PCA 维度

    Returns:
        graph_data: dict，包含：
            - x:           节点特征矩阵 [N, n_components]
            - y:           节点标签 [N]
            - edge_index_list:  List[Tensor] 每种关系的边
            - edge_weight_list: List[Tensor] 每种关系的边权重
            - num_relations:    关系数量
            - num_nodes:        节点数
    """
    N = features.shape[0]
    groups = get_feature_groups(n_components)
    relation_names = ["fft_similarity", "wavelet_similarity", "global_similarity"]
    feature_groups = [groups["fft_group"], groups["wavelet_group"], groups["all"]]

    edge_index_list  = []
    edge_weight_list = []

    for rel_name, feat_idx in zip(relation_names, feature_groups):
        print(f"  [图构建] 关系 '{rel_name}'：使用特征维度 {feat_idx[:3]}...，K={k}")
        ei, ew = build_knn_graph(
            features, k=k,
            feature_indices=feat_idx,
            symmetric=True,
        )
        edge_index_list.append(ei)
        edge_weight_list.append(ew)
        print(f"           → 边数：{ei.shape[1]}")

    graph_data = {
        "x":                torch.tensor(features, dtype=torch.float32),
        "y":                torch.tensor(labels,   dtype=torch.long),
        "edge_index_list":  edge_index_list,
        "edge_weight_list": edge_weight_list,
        "num_relations":    len(relation_names),
        "num_nodes":        N,
        "relation_names":   relation_names,
    }

    total_edges = sum(ei.shape[1] for ei in edge_index_list)
    print(f"[INFO] 异质图构建完成：{N} 节点，{len(relation_names)} 关系，共 {total_edges} 条边")
    return graph_data


# ============================================================
# 转换为 PyG 的 HeteroData / 普通 Data 格式
# ============================================================
def to_pyg_data(graph_data: dict, train_mask=None, val_mask=None, test_mask=None):
    """
    将 graph_data 转换为 torch_geometric.data.Data 对象。
    使用 RGCN 的 edge_type 表示多关系。
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("请安装 torch-geometric：pip install torch-geometric")

    x = graph_data["x"]
    y = graph_data["y"]
    N = graph_data["num_nodes"]

    # 合并所有关系的边，用 edge_type 区分
    all_edges   = []
    all_weights = []
    all_types   = []
    for rel_id, (ei, ew) in enumerate(
        zip(graph_data["edge_index_list"], graph_data["edge_weight_list"])
    ):
        all_edges.append(ei)
        all_weights.append(ew)
        all_types.append(torch.full((ei.shape[1],), rel_id, dtype=torch.long))

    edge_index  = torch.cat(all_edges,   dim=1)
    edge_weight = torch.cat(all_weights, dim=0)
    edge_type   = torch.cat(all_types,   dim=0)

    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_type=edge_type,
        num_nodes=N,
        num_relations=graph_data["num_relations"],
    )

    if train_mask is not None:
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    if val_mask is not None:
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    if test_mask is not None:
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

    return data


# ============================================================
# 构建训练/验证/测试掩码
# ============================================================
def build_masks(N: int, train_ratio: float = 0.7, val_ratio: float = 0.1, seed: int = 42):
    """随机划分训练/验证/测试掩码"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_train = int(N * train_ratio)
    n_val   = int(N * val_ratio)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    train_mask = np.zeros(N, dtype=bool)
    val_mask   = np.zeros(N, dtype=bool)
    test_mask  = np.zeros(N, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True
    return train_mask, val_mask, test_mask


# ============================================================
# 保存/加载图数据
# ============================================================
def save_graph(graph_data: dict, path: Path):
    """保存图数据为 .pt 文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph_data, path)
    print(f"[INFO] 图数据已保存：{path}")


def load_graph(path: Path) -> dict:
    """加载图数据"""
    return torch.load(path, weights_only=False)


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    feat_dir = Path(__file__).parent.parent / "data" / "features"
    features = np.load(feat_dir / "features.npy")
    labels   = np.load(feat_dir / "labels.npy")

    print(f"特征矩阵：{features.shape}，标签：{labels.shape}")

    # 加载 PCA 预处理器
    import joblib
    scaler = joblib.load(feat_dir / "preprocessors" / "scaler.pkl")
    pca    = joblib.load(feat_dir / "preprocessors" / "pca.pkl")
    features_pca = pca.transform(scaler.transform(features))
    print(f"PCA 后特征：{features_pca.shape}")

    # 构建异质图
    print("\n构建异质图...")
    graph_data = build_heterogeneous_graph(features_pca, labels, k=10, n_components=20)

    # 构建 PyG Data 对象
    train_mask, val_mask, test_mask = build_masks(len(labels))
    data = to_pyg_data(graph_data, train_mask, val_mask, test_mask)
    print(f"\nPyG Data 对象：{data}")
    print(f"train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    # 保存
    save_graph(graph_data, feat_dir / "graph_data.pt")
    torch.save(data, feat_dir / "pyg_data.pt")
    print("[✓] 图构建完成")
