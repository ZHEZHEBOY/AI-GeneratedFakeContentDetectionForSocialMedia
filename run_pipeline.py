"""
一键运行完整实验流程
Usage:
  python run_pipeline.py              # 使用已下载的真实数据集
  python run_pipeline.py --demo       # 使用自动生成的演示数据集（无需下载）
  python run_pipeline.py --ablation   # 同时运行消融实验
  python run_pipeline.py --demo --ablation  # 演示模式 + 消融实验
"""

import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from src.data_preprocessing import (
    load_processed, save_processed, generate_demo_dataset,
    load_images_from_dir, auto_detect_dataset, RAW_DIR
)
from src.feature_extraction import batch_extract_features, normalize_and_reduce, ALL_FEATURE_NAMES
from src.graph_construction import build_heterogeneous_graph, build_masks, to_pyg_data
from src.models.mc_rgcn import MCRGCN, model_summary
from src.models.baselines import get_sklearn_baselines, MLPTrainer, compute_metrics, print_comparison_table
from src.visualize import (
    generate_all_figures, plot_confusion_matrix,
    plot_roc_curves, plot_mc_uncertainty, plot_ablation_study
)
from sklearn.model_selection import train_test_split

FEAT_DIR   = ROOT_DIR / "data" / "features"
MODEL_DIR  = ROOT_DIR / "checkpoints"
RESULT_DIR = ROOT_DIR / "results"
for d in [FEAT_DIR, MODEL_DIR, RESULT_DIR / "tables", RESULT_DIR / "figures"]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
CONFIG = {
    "max_per_class":   5000,
    "img_size":        64,
    "pca_components":  20,
    "knn_k":           10,
    "hidden_channels": 64,
    "num_relations":   3,
    "dropout_rate":    0.3,
    "mc_n_forward":    50,
    "lr":              1e-3,
    "weight_decay":    1e-4,
    "epochs":          200,
    "patience":        20,
    "seed":            42,
    "device":          "auto",
}


def banner(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Step 1: 数据加载
# ============================================================
def step1_load_data(use_demo: bool) -> tuple:
    banner("Step 1 / 6  数据集加载")
    if use_demo:
        print("[DEMO] 生成演示数据集（500张/类）...")
        real_dir, fake_dir = generate_demo_dataset(n_real=500, n_fake=500)
        images, labels = load_images_from_dir(real_dir, fake_dir, max_per_class=500,
                                               img_size=(CONFIG["img_size"], CONFIG["img_size"]))
        save_processed(images, labels, "dataset")
    else:
        try:
            images, labels = load_processed("dataset")
            print(f"[加载] 使用已处理数据：{images.shape}")
        except FileNotFoundError:
            print("[检测] 未找到处理后数据，开始从原始图像加载...")
            real_dir, fake_dir = auto_detect_dataset(RAW_DIR)
            images, labels = load_images_from_dir(
                real_dir, fake_dir,
                max_per_class=CONFIG["max_per_class"],
                img_size=(CONFIG["img_size"], CONFIG["img_size"]),
            )
            save_processed(images, labels, "dataset")

    print(f"[✓] 数据集：{len(labels)} 张 | 真实={(labels==0).sum()} | AI合成={(labels==1).sum()}")
    return images, labels


# ============================================================
# Step 2: 特征提取
# ============================================================
def step2_extract_features(images: np.ndarray, labels: np.ndarray) -> tuple:
    banner("Step 2 / 6  频域特征提取（FFT + 小波）")
    feat_path = FEAT_DIR / "features.npy"
    lab_path  = FEAT_DIR / "labels.npy"
    if feat_path.exists() and lab_path.exists() and np.load(lab_path).shape[0] == len(labels):
        print("[加载] 使用已有特征文件...")
        features = np.load(feat_path)
        labels   = np.load(lab_path)
    else:
        features = batch_extract_features(images, n_jobs=4)
        np.save(feat_path, features)
        np.save(lab_path,  labels)
    print(f"[✓] 特征矩阵：{features.shape}（{len(ALL_FEATURE_NAMES)} 维）")
    return features, labels


# ============================================================
# Step 3: 降维 & 图构建
# ============================================================
def step3_build_graph(features: np.ndarray, labels: np.ndarray) -> dict:
    banner("Step 3 / 6  PCA 降维 + 异质图构建")
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, labels, test_size=0.2,
        random_state=CONFIG["seed"], stratify=labels,
    )
    prep_dir = FEAT_DIR / "preprocessors"
    X_tr_pca, X_te_pca, evr = normalize_and_reduce(
        X_tr, X_te, n_components=CONFIG["pca_components"], save_dir=prep_dir
    )
    import joblib
    scaler = joblib.load(prep_dir / "scaler.pkl")
    pca    = joblib.load(prep_dir / "pca.pkl")
    features_pca = pca.transform(scaler.transform(features))

    graph_data = build_heterogeneous_graph(
        features_pca, labels, k=CONFIG["knn_k"], n_components=CONFIG["pca_components"]
    )
    train_mask, val_mask, test_mask = build_masks(
        len(labels), train_ratio=0.7, val_ratio=0.1, seed=CONFIG["seed"]
    )
    pyg_data = to_pyg_data(graph_data, train_mask, val_mask, test_mask)
    print(f"[✓] 图构建完成：{graph_data['num_nodes']} 节点，{graph_data['num_relations']} 关系")

    return {
        "X_train": X_tr_pca, "X_test": X_te_pca,
        "y_train": y_tr,     "y_test": y_te,
        "pyg_data": pyg_data,
        "features_pca": features_pca,
        "labels": labels,
    }


# ============================================================
# Step 4: 训练主模型 MC-RGCN
# ============================================================
def step4_train_mcrgcn(data: dict) -> tuple:
    banner("Step 4 / 6  MC-RGCN 主模型训练")
    if CONFIG["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(CONFIG["device"])
    print(f"[设备] {device}")

    pyg   = data["pyg_data"].to(device)
    x, y  = pyg.x, pyg.y
    ei, et = pyg.edge_index, pyg.edge_type

    model = MCRGCN(
        in_channels=CONFIG["pca_components"],
        hidden_channels=CONFIG["hidden_channels"],
        out_channels=2,
        num_relations=CONFIG["num_relations"],
        dropout_rate=CONFIG["dropout_rate"],
        use_pyg=True,
    ).to(device)
    model_summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc, patience_cnt, best_epoch = 0.0, 0, 0
    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    t0 = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, ei, et)
        loss   = criterion(logits[pyg.train_mask], y[pyg.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vl = model(x, ei, et)
            vp = vl[pyg.val_mask].argmax(-1).cpu().numpy()
        vm = compute_metrics(y[pyg.val_mask].cpu().numpy(), vp)
        scheduler.step(vm["accuracy"])

        history["train_loss"].append(loss.item())
        history["val_acc"].append(vm["accuracy"])
        history["val_f1"].append(vm["f1"])

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | Loss {loss.item():.4f} | Val Acc {vm['accuracy']:.4f} | {time.time()-t0:.0f}s")

        if vm["accuracy"] > best_val_acc:
            best_val_acc = vm["accuracy"]
            best_epoch   = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), MODEL_DIR / "mc_rgcn_best.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["patience"]:
                print(f"\n[早停] Epoch={epoch}，最优={best_epoch}，Val Acc={best_val_acc:.4f}")
                break

    model.load_state_dict(torch.load(MODEL_DIR / "mc_rgcn_best.pth", map_location=device))
    print(f"\n[MC] 使用 {CONFIG['mc_n_forward']} 次 MC 采样进行测试...")
    mean_probs, uncertainty = model.predict_with_uncertainty(x, ei, et, n_forward=CONFIG["mc_n_forward"])

    test_pred = mean_probs[pyg.test_mask].argmax(-1).cpu().numpy()
    test_prob = mean_probs[pyg.test_mask][:, 1].cpu().numpy()
    test_true = y[pyg.test_mask].cpu().numpy()
    metrics   = compute_metrics(test_true, test_pred, test_prob)
    metrics["mc_uncertainty_mean"] = float(uncertainty[pyg.test_mask].mean().item())

    print("\n[MC-RGCN] 测试结果：")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return model, metrics, history, uncertainty.cpu().numpy(), data["labels"]


# ============================================================
# Step 5: 训练对比模型
# ============================================================
def step5_train_baselines(data: dict) -> dict:
    banner("Step 5 / 6  对比模型训练")
    X_tr, X_te = data["X_train"], data["X_test"]
    y_tr, y_te = data["y_train"], data["y_test"]
    results = {}
    for bl in get_sklearn_baselines():
        bl.fit(X_tr, y_tr)
        m = bl.evaluate(X_te, y_te)
        results[bl.name] = m
        bl.save(MODEL_DIR / f"{bl.name.split('（')[0]}.pkl")
        print(f"[{bl.name}] Acc={m['accuracy']:.4f} F1={m['f1']:.4f} AUC={m.get('auc',float('nan')):.4f}")

    mlp = MLPTrainer(in_channels=CONFIG["pca_components"], epochs=100, batch_size=256)
    mlp.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
    m = mlp.evaluate(X_te, y_te)
    results[mlp.name] = m
    mlp.save(MODEL_DIR / "mlp.pth")
    print(f"[{mlp.name}] Acc={m['accuracy']:.4f} F1={m['f1']:.4f}")
    return results


# ============================================================
# Step 6: 保存结果 & 生成图表
# ============================================================
def step6_save_and_visualize(
    rgcn_metrics, baseline_results, history,
    uncertainty, labels_full,
    images, features, labels,
):
    banner("Step 6 / 6  结果保存 & 可视化")
    all_results = {"MC-RGCN（主模型）": rgcn_metrics}
    all_results.update(baseline_results)
    print_comparison_table(all_results)

    # 保存 JSON
    with open(RESULT_DIR / "tables" / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(RESULT_DIR / "tables" / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 保存 CSV
    import pandas as pd
    rows = [{"模型": k, **{k2: round(v2, 4) for k2, v2 in v.items() if k2 != "confusion_matrix"}}
            for k, v in all_results.items()]
    pd.DataFrame(rows).to_csv(RESULT_DIR / "tables" / "model_comparison.csv",
                               index=False, encoding="utf-8-sig")

    # 可视化
    generate_all_figures(
        images=images[:min(500, len(images))],
        labels_raw=labels[:min(500, len(labels))],
        features=features, labels=labels,
        history=history, results=all_results,
    )

    # 混淆矩阵
    if "confusion_matrix" in rgcn_metrics:
        plot_confusion_matrix(rgcn_metrics["confusion_matrix"], "MC-RGCN")

    # ROC 曲线（需要概率值，此处用指标中 auc 代替展示）
    print("[✓] 结果已保存至 results/")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI合成虚假内容检测 - 完整实验流程")
    parser.add_argument("--demo",      action="store_true", help="使用演示数据集（无需下载）")
    parser.add_argument("--ablation",  action="store_true", help="运行消融实验")
    parser.add_argument("--skip-viz",  action="store_true", help="跳过可视化")
    parser.add_argument("--epochs",    type=int, default=CONFIG["epochs"])
    parser.add_argument("--device",    type=str, default=CONFIG["device"])
    args = parser.parse_args()

    CONFIG["epochs"] = args.epochs
    CONFIG["device"] = args.device
    set_seed(CONFIG["seed"])

    t_start = time.time()

    # 运行流程
    images, labels                          = step1_load_data(args.demo)
    features, labels                        = step2_extract_features(images, labels)
    data                                    = step3_build_graph(features, labels)
    model, rgcn_m, history, uncert, labs_f  = step4_train_mcrgcn(data)
    baseline_results                        = step5_train_baselines(data)

    if not args.skip_viz:
        step6_save_and_visualize(
            rgcn_m, baseline_results, history,
            uncert, labs_f, images, features, labels,
        )

    # 消融实验（可选）
    if args.ablation:
        banner("消融实验（额外步骤）")
        from src.ablation import run_all_ablations
        ablation_results = run_all_ablations(images, labels, epochs=min(100, args.epochs), device=args.device)
        if not args.skip_viz:
            plot_ablation_study(ablation_results)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  全流程完成！总用时：{total_time/60:.1f} 分钟")
    print(f"  结果文件：results/tables/model_comparison.csv")
    print(f"  可视化图表：results/figures/")
    print(f"{'='*60}")
