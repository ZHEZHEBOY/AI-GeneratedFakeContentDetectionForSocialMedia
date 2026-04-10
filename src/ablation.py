"""
消融实验模块
系统地测试各模块对最终性能的贡献：

实验设计：
  A. 无FFT特征       → 仅使用小波特征
  B. 无小波特征      → 仅使用FFT特征
  C. 无MC-Dropout    → 标准Dropout（预测时关闭）
  D. 单关系图        → RGCN使用1种关系（全局相似度）
  E. 无图结构(MLP)   → 直接用MLP分类（无图卷积）
  F. 完整MC-RGCN     → 所有模块

每个实验保持其他超参数不变，只改变对应模块。
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_extraction import (
    batch_extract_features, normalize_and_reduce,
    FFT_FEATURE_NAMES, WAVELET_FEATURE_NAMES
)
from src.graph_construction import build_heterogeneous_graph, build_masks, to_pyg_data
from src.models.mc_rgcn import MCRGCN
from src.models.baselines import compute_metrics, MLPTrainer

ROOT_DIR   = Path(__file__).parent.parent
RESULT_DIR = ROOT_DIR / "results" / "tables"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 标准 RGCN（无 MC Dropout，预测时关闭 Dropout）
# ============================================================
class StandardRGCN(MCRGCN):
    """RGCN without MC Dropout（消融实验 C）"""

    def predict_standard(self, x, edge_index, edge_type):
        """标准预测：eval 模式下关闭 Dropout"""
        self.eval()
        with torch.no_grad():
            logits = nn.Module.__call__(self, x, edge_index, edge_type)
            probs  = F.softmax(logits, dim=-1)
        return probs


# ============================================================
# 训练单个消融配置
# ============================================================
def run_ablation_config(
    config_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    pca_components: int = 20,
    knn_k: int = 10,
    num_relations: int = 3,
    use_mc_dropout: bool = True,
    epochs: int = 100,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """
    运行一个消融实验配置，返回测试集指标。
    """
    print(f"\n[消融] 运行：{config_name}")
    device = torch.device(device)

    # 划分数据
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    X_tr_pca, X_te_pca, _ = normalize_and_reduce(X_tr, X_te, n_components=pca_components)

    # 构建全图（用 PCA 特征）- 每次消融都重新 fit，不复用主流程保存的 scaler/pca
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    scaler = StandardScaler().fit(X_tr)
    pca = PCA(n_components=pca_components, random_state=seed).fit(scaler.transform(X_tr))
    features_pca = pca.transform(scaler.transform(features))

    graph_data = build_heterogeneous_graph(
        features_pca, labels, k=knn_k, n_components=pca_components
    )
    # 限制关系数量（消融D）
    if num_relations < 3:
        graph_data["edge_index_list"]  = graph_data["edge_index_list"][:num_relations]
        graph_data["edge_weight_list"] = graph_data["edge_weight_list"][:num_relations]
        graph_data["num_relations"]    = num_relations
        # 重映射 edge_type
        import torch as th
        ei_list, et_list = [], []
        for r_id, ei in enumerate(graph_data["edge_index_list"]):
            ei_list.append(ei)
            et_list.append(th.full((ei.shape[1],), r_id, dtype=th.long))

    train_mask, val_mask, test_mask = build_masks(len(labels), seed=seed)
    pyg_data = to_pyg_data(graph_data, train_mask, val_mask, test_mask).to(device)

    x          = pyg_data.x
    y          = pyg_data.y
    edge_index = pyg_data.edge_index
    edge_type  = pyg_data.edge_type

    # 初始化模型
    model = MCRGCN(
        in_channels=pca_components,
        hidden_channels=64,
        out_channels=2,
        num_relations=num_relations,
        dropout_rate=0.3 if use_mc_dropout else 0.0,
        use_pyg=False,  # 避免 pyg 依赖问题
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 训练
    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_type)
        loss = criterion(logits[pyg_data.train_mask], y[pyg_data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(x, edge_index, edge_type)
                val_pred = val_logits[pyg_data.val_mask].argmax(dim=-1).cpu().numpy()
            val_acc = compute_metrics(y[pyg_data.val_mask].cpu().numpy(), val_pred)["accuracy"]
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")
            best_val_acc = max(best_val_acc, val_acc)

    # 测试
    if use_mc_dropout:
        mean_probs, _ = model.predict_with_uncertainty(x, edge_index, edge_type, n_forward=30)
        test_pred = mean_probs[pyg_data.test_mask].argmax(dim=-1).cpu().numpy()
        test_prob = mean_probs[pyg_data.test_mask][:, 1].cpu().numpy()
    else:
        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index, edge_type)
            probs  = F.softmax(logits, dim=-1)
        test_pred = probs[pyg_data.test_mask].argmax(dim=-1).cpu().numpy()
        test_prob = probs[pyg_data.test_mask][:, 1].cpu().numpy()

    test_true = y[pyg_data.test_mask].cpu().numpy()
    metrics = compute_metrics(test_true, test_pred, test_prob)
    print(f"  [{config_name}] Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics.get('auc', float('nan')):.4f}")
    return metrics


# ============================================================
# 消融实验主流程
# ============================================================
def run_all_ablations(
    images: np.ndarray,
    labels: np.ndarray,
    epochs: int = 100,
    device: str = "cpu",
) -> dict:
    """运行所有消融实验"""
    print("\n" + "=" * 60)
    print("消融实验开始")
    print("=" * 60)

    # 提取原始特征
    print("[特征] 提取频域特征...")
    features_all = batch_extract_features(images, n_jobs=4)
    n_fft = len(FFT_FEATURE_NAMES)   # 17

    ablation_results = {}

    # --- A. 无FFT特征（仅小波）---
    feat_no_fft = features_all[:, n_fft:]  # 仅小波特征 36维
    ablation_results["无FFT特征"] = run_ablation_config(
        "无FFT特征", feat_no_fft, labels,
        pca_components=min(20, feat_no_fft.shape[1]),
        epochs=epochs, device=device,
    )

    # --- B. 无小波特征（仅FFT）---
    feat_no_wav = features_all[:, :n_fft]  # 仅FFT特征 17维
    ablation_results["无小波特征"] = run_ablation_config(
        "无小波特征", feat_no_wav, labels,
        pca_components=min(15, feat_no_wav.shape[1]),
        epochs=epochs, device=device,
    )

    # --- C. 无MC-Dropout ---
    ablation_results["无MC-Dropout"] = run_ablation_config(
        "无MC-Dropout", features_all, labels,
        use_mc_dropout=False,
        epochs=epochs, device=device,
    )

    # --- D. 单关系图 ---
    ablation_results["单关系图"] = run_ablation_config(
        "单关系图", features_all, labels,
        num_relations=1,
        epochs=epochs, device=device,
    )

    # --- E. 无图结构（MLP）---
    print(f"\n[消融] 运行：无图结构（MLP）")
    X_tr, X_te, y_tr, y_te = train_test_split(
        features_all, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_tr_pca, X_te_pca, _ = normalize_and_reduce(X_tr, X_te, n_components=20)
    mlp = MLPTrainer(in_channels=20, epochs=epochs, batch_size=256)
    mlp.fit(X_tr_pca, y_tr)
    ablation_results["无图结构（MLP）"] = mlp.evaluate(X_te_pca, y_te)
    m = ablation_results["无图结构（MLP）"]
    print(f"  [无图结构（MLP）] Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}")

    # --- F. 完整MC-RGCN ---
    ablation_results["完整MC-RGCN★"] = run_ablation_config(
        "完整MC-RGCN", features_all, labels,
        use_mc_dropout=True, num_relations=3,
        epochs=epochs, device=device,
    )

    # 保存结果
    save_path = RESULT_DIR / "ablation_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, ensure_ascii=False, indent=2)
    print(f"\n[✓] 消融实验结果已保存：{save_path}")

    # 打印汇总表
    print("\n消融实验汇总：")
    print(f"{'配置':<20} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 55)
    for name, m in ablation_results.items():
        print(f"{name:<20} {m.get('accuracy', 0):>10.4f} {m.get('f1', 0):>10.4f} {m.get('auc', float('nan')):>10.4f}")

    return ablation_results


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    from src.data_preprocessing import load_processed
    images, labels = load_processed("dataset")
    print(f"数据：{images.shape}，标签：{labels.shape}")

    ablation_results = run_all_ablations(images, labels, epochs=args.epochs, device=args.device)
