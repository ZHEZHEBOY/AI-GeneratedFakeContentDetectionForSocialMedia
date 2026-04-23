"""

MC-RGCN 训练脚本
包含：
  - 训练循环（含早停）
  - 验证集监控
  - 模型保存/加载
  - 超参数配置
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import load_processed, PROCESSED_DIR, generate_demo_dataset, save_processed, load_images_from_dir
from src.feature_extraction import batch_extract_features, normalize_and_reduce
from src.graph_construction import build_heterogeneous_graph, build_masks, to_pyg_data
from src.models.mc_rgcn import MCRGCN, model_summary
from src.models.baselines import get_sklearn_baselines, MLPTrainer, compute_metrics, print_comparison_table

ROOT_DIR    = Path(__file__).parent.parent
RESULT_DIR  = ROOT_DIR / "results"
MODEL_DIR   = ROOT_DIR / "checkpoints"
FEAT_DIR    = ROOT_DIR / "data" / "features"

for d in [RESULT_DIR / "figures", RESULT_DIR / "tables", MODEL_DIR, FEAT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 超参数配置
# ============================================================
DEFAULT_CONFIG = {
    # 数据
    "max_per_class": 5000,
    "img_size":      64,
    "pca_components": 20,
    "knn_k":         10,

    # MC-RGCN
    "hidden_channels": 64,
    "num_relations":   3,
    "dropout_rate":    0.3,
    "mc_n_forward":    50,

    # 训练
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "epochs":        200,
    "patience":      20,   # 早停耐心值
    "batch_size":    -1,   # -1 = 全图训练（节点分类）

    # 其他
    "seed":    42,
    "device":  "auto",
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 数据准备流程
# ============================================================
def prepare_data(config: dict, use_demo: bool = False):
    """完整数据准备流程：加载→特征提取→降维→图构建"""

    # 1. 加载图像
    if use_demo:
        from src.data_preprocessing import RAW_DIR, generate_demo_dataset
        real_dir, fake_dir = generate_demo_dataset(n_real=500, n_fake=500)
        images, labels = load_images_from_dir(real_dir, fake_dir, max_per_class=500, img_size=(config["img_size"], config["img_size"]))
        save_processed(images, labels, "dataset")
    else:
        images, labels = load_processed("dataset")

    print(f"[数据] 共 {len(labels)} 张图像：真实={(labels==0).sum()}, AI合成={(labels==1).sum()}")

    # 2. 提取频域特征
    feat_path = FEAT_DIR / "features.npy"
    lab_path  = FEAT_DIR / "labels.npy"
    if feat_path.exists() and lab_path.exists():
        print("[特征] 加载已有特征文件...")
        features = np.load(feat_path)
        labels_f = np.load(lab_path)
        assert len(features) == len(labels), "特征与图像数量不匹配，请删除 data/features/*.npy 重新生成"
    else:
        features = batch_extract_features(images, n_jobs=4)
        np.save(feat_path, features)
        np.save(lab_path,  labels)

    # 3. 划分训练/测试集（sklearn模型用）
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, labels, test_size=0.2, random_state=config["seed"], stratify=labels
    )

    # 4. 标准化 + PCA 降维
    prep_dir = FEAT_DIR / "preprocessors"
    X_tr_pca, X_te_pca, evr = normalize_and_reduce(
        X_tr, X_te,
        n_components=config["pca_components"],
        save_dir=prep_dir,
    )
    print(f"[PCA] 降维完成：{config['pca_components']}维，累计方差：{evr.cumsum()[-1]:.4f}")

    # 5. 构建全图（GCN用）
    import joblib
    scaler = joblib.load(prep_dir / "scaler.pkl")
    pca    = joblib.load(prep_dir / "pca.pkl")
    features_pca = pca.transform(scaler.transform(features))

    print(f"\n[图] 构建异质图（K={config['knn_k']}）...")
    graph_data = build_heterogeneous_graph(
        features_pca, labels,
        k=config["knn_k"],
        n_components=config["pca_components"],
    )
    train_mask, val_mask, test_mask = build_masks(
        len(labels), train_ratio=0.7, val_ratio=0.1, seed=config["seed"]
    )
    pyg_data = to_pyg_data(graph_data, train_mask, val_mask, test_mask)

    return {
        "X_train": X_tr_pca, "X_test": X_te_pca,
        "y_train": y_tr,     "y_test": y_te,
        "pyg_data": pyg_data,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "labels": labels,
        "features_pca": features_pca,
    }


# ============================================================
# MC-RGCN 训练
# ============================================================
def train_mc_rgcn(data: dict, config: dict) -> tuple:
    """训练 MC-RGCN 模型"""

    if config["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])
    print(f"\n[MC-RGCN] 设备：{device}")

    pyg_data = data["pyg_data"].to(device)
    x          = pyg_data.x
    y          = pyg_data.y
    edge_index = pyg_data.edge_index
    edge_type  = pyg_data.edge_type
    train_mask = pyg_data.train_mask
    val_mask   = pyg_data.val_mask
    test_mask  = pyg_data.test_mask

    model = MCRGCN(
        in_channels=config["pca_components"],
        hidden_channels=config["hidden_channels"],
        out_channels=2,
        num_relations=config["num_relations"],
        dropout_rate=config["dropout_rate"],
        use_pyg=True,
    ).to(device)
    model_summary(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc  = 0.0
    best_epoch    = 0
    patience_cnt  = 0
    history       = {"train_loss": [], "val_acc": [], "val_f1": []}

    print(f"\n[MC-RGCN] 开始训练（最多 {config['epochs']} Epoch，早停耐心 {config['patience']}）...")
    t0 = time.time()

    for epoch in range(1, config["epochs"] + 1):
        # --- 训练 ---
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_type)
        loss   = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # --- 验证 ---
        model.eval()
        with torch.no_grad():
            val_logits = model(x, edge_index, edge_type)
            val_pred   = val_logits[val_mask].argmax(dim=-1).cpu().numpy()
        val_true = y[val_mask].cpu().numpy()
        val_metrics = compute_metrics(val_true, val_pred)
        val_acc = val_metrics["accuracy"]
        val_f1  = val_metrics["f1"]

        scheduler.step(val_acc)
        history["train_loss"].append(loss.item())
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | {elapsed:.1f}s")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), MODEL_DIR / "mc_rgcn_best.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= config["patience"]:
                print(f"\n[早停] Epoch {epoch}，最优 Epoch={best_epoch}，Val Acc={best_val_acc:.4f}")
                break

    # 加载最优模型
    model.load_state_dict(torch.load(MODEL_DIR / "mc_rgcn_best.pth", map_location=device))
    print(f"\n[MC-RGCN] 训练完成，最优模型来自 Epoch {best_epoch}，Val Acc={best_val_acc:.4f}")

    # --- 测试集评估（MC预测）---
    print(f"[MC-RGCN] 使用 MC Dropout ({config['mc_n_forward']} 次采样) 进行测试集预测...")
    mean_probs, uncertainty = model.predict_with_uncertainty(
        x, edge_index, edge_type, n_forward=config["mc_n_forward"]
    )
    test_pred = mean_probs[test_mask].argmax(dim=-1).cpu().numpy()
    test_prob = mean_probs[test_mask][:, 1].cpu().numpy()
    test_true = y[test_mask].cpu().numpy()
    test_metrics = compute_metrics(test_true, test_pred, test_prob)
    test_metrics["mc_uncertainty_mean"] = float(uncertainty[test_mask].mean().item())

    print(f"\n[MC-RGCN] 测试集结果：")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return model, test_metrics, history


# ============================================================
# 对比模型训练
# ============================================================
def train_baselines(data: dict, config: dict) -> dict:
    """训练所有对比模型并返回结果"""
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    all_results = {}
    print("\n" + "=" * 60)
    print("对比模型训练")
    print("=" * 60)

    # sklearn 模型
    for baseline in get_sklearn_baselines():
        baseline.fit(X_train, y_train)
        metrics = baseline.evaluate(X_test, y_test)
        all_results[baseline.name] = metrics
        baseline.save(MODEL_DIR / f"{baseline.name.split('（')[0]}.pkl")
        print(f"[{baseline.name}] Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics.get('auc', float('nan')):.4f}")

    # MLP
    mlp = MLPTrainer(
        in_channels=config["pca_components"],
        hidden_dims=[128, 64, 32],
        dropout_rate=0.3,
        lr=1e-3,
        epochs=100,
        batch_size=256,
    )
    mlp.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    mlp_metrics = mlp.evaluate(X_test, y_test)
    all_results[mlp.name] = mlp_metrics
    mlp.save(MODEL_DIR / "mlp.pth")
    print(f"[{mlp.name}] Acc={mlp_metrics['accuracy']:.4f}, F1={mlp_metrics['f1']:.4f}")

    return all_results


# ============================================================
# 保存结果
# ============================================================
def save_results(rgcn_metrics: dict, baseline_results: dict, history: dict, config: dict):
    all_results = {"MC-RGCN（主模型）": rgcn_metrics}
    all_results.update(baseline_results)

    # 打印对比表
    print_comparison_table(all_results)

    # 保存为 JSON
    result_path = RESULT_DIR / "tables" / "model_comparison.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 保存训练历史
    hist_path = RESULT_DIR / "tables" / "training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 保存为 CSV
    import pandas as pd
    rows = []
    for model_name, metrics in all_results.items():
        row = {"模型": model_name}
        for k, v in metrics.items():
            if k != "confusion_matrix":
                row[k] = round(v, 4) if isinstance(v, float) else v
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "tables" / "model_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\n[✓] 结果已保存至 results/tables/")

    return all_results


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MC-RGCN 训练脚本")
    parser.add_argument("--demo",          action="store_true", help="使用演示数据集")
    parser.add_argument("--skip-baselines",action="store_true", help="跳过对比模型训练")
    parser.add_argument("--epochs",        type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr",            type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--hidden",        type=int,   default=DEFAULT_CONFIG["hidden_channels"])
    parser.add_argument("--knn-k",         type=int,   default=DEFAULT_CONFIG["knn_k"])
    parser.add_argument("--pca",           type=int,   default=DEFAULT_CONFIG["pca_components"])
    parser.add_argument("--mc-n",          type=int,   default=DEFAULT_CONFIG["mc_n_forward"])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        "epochs":          args.epochs,
        "lr":              args.lr,
        "hidden_channels": args.hidden,
        "knn_k":           args.knn_k,
        "pca_components":  args.pca,
        "mc_n_forward":    args.mc_n,
    })

    set_seed(config["seed"])
    print("=" * 60)
    print("MC-RGCN 训练启动")
    print("=" * 60)
    print(f"配置：{json.dumps(config, indent=2, ensure_ascii=False)}")

    # 准备数据
    data = prepare_data(config, use_demo=args.demo)

    # 训练 MC-RGCN
    model, rgcn_metrics, history = train_mc_rgcn(data, config)

    # 训练对比模型
    baseline_results = {} if args.skip_baselines else train_baselines(data, config)

    # 保存结果
    all_results = save_results(rgcn_metrics, baseline_results, history, config)

    print("\n[✓] 所有训练完成！")
    print("下一步：运行 python src/visualize.py 生成可视化图表")
