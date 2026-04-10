"""
对比模型（Baseline Models）
与原论文一一对应，供评委直观比较：

1. 逻辑回归（LR）      —— 传统统计模型
2. 支持向量机（SVM）   —— 传统机器学习
3. 决策树（DT）        —— 传统机器学习
4. 随机森林（RF）      —— 集成学习
5. MLP 多层感知机      —— 基础深度学习

所有模型使用相同的 PCA 降维特征作为输入，确保公平对比。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
import joblib
from pathlib import Path


# ============================================================
# 统一评估指标计算
# ============================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """计算分类评估指标"""
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, average="binary"),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="binary", zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["auc"] = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


# ============================================================
# sklearn 系列模型
# ============================================================
class SklearnBaseline:
    """
    封装 sklearn 模型，统一训练/预测/评估接口
    """
    def __init__(self, name: str, model):
        self.name  = name
        self.model = model
        self.is_trained = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"[{self.name}] 训练中...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"[{self.name}] 训练完成")
        return self

    def predict(self, X: np.ndarray) -> tuple:
        y_pred = self.model.predict(X)
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, "decision_function"):
            df = self.model.decision_function(X)
            # 归一化到 [0,1]
            y_prob = (df - df.min()) / (df.max() - df.min() + 1e-10)
        else:
            y_prob = y_pred.astype(float)
        return y_pred, y_prob

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        y_pred, y_prob = self.predict(X)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        return metrics

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)
        self.is_trained = True


def get_sklearn_baselines() -> list:
    """返回所有 sklearn 对比模型"""
    return [
        SklearnBaseline("LR（逻辑回归）", LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, n_jobs=-1
        )),
        SklearnBaseline("SVM（支持向量机）", SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=42
        )),
        SklearnBaseline("DT（决策树）", DecisionTreeClassifier(
            max_depth=10, random_state=42
        )),
        SklearnBaseline("RF（随机森林）", RandomForestClassifier(
            n_estimators=200, max_depth=15,
            random_state=42, n_jobs=-1
        )),
    ]


# ============================================================
# MLP 多层感知机（PyTorch 实现）
# ============================================================
class MLPClassifier(nn.Module):
    """
    MLP 多层感知机，基础深度学习基线模型

    架构：
      输入 → FC(128) → BN → ReLU → Dropout(0.3)
           → FC(64)  → BN → ReLU → Dropout(0.3)
           → FC(32)  → ReLU
           → FC(2)   → 输出（logits）
    """
    def __init__(
        self,
        in_channels:  int = 20,
        hidden_dims:  list = None,
        out_channels: int = 2,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = in_channels
        for i, hd in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hd

        layers.append(nn.Linear(prev_dim, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPTrainer:
    """MLP 训练器，封装 PyTorch 训练流程"""

    def __init__(
        self,
        in_channels: int = 20,
        hidden_dims: list = None,
        dropout_rate: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 256,
        device: str = "auto",
    ):
        self.epochs     = epochs
        self.batch_size = batch_size
        self.name       = "MLP（多层感知机）"

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = MLPClassifier(
            in_channels=in_channels,
            hidden_dims=hidden_dims or [128, 64, 32],
            out_channels=2,
            dropout_rate=dropout_rate,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_accs = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        print(f"[MLP] 开始训练，设备：{self.device}，Epochs：{self.epochs}")

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss   = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
            avg_loss = total_loss / len(loader)
            self.train_losses.append(avg_loss)

            if epoch % 20 == 0 or epoch == 1:
                if X_val is not None:
                    val_metrics = self.evaluate(X_val, y_val)
                    self.val_accs.append(val_metrics["accuracy"])
                    print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
                else:
                    print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

        print(f"[MLP] 训练完成")
        return self

    def predict(self, X: np.ndarray) -> tuple:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
        y_pred = probs.argmax(axis=1)
        y_prob = probs[:, 1]
        return y_pred, y_prob

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        y_pred, y_prob = self.predict(X)
        return compute_metrics(y_true, y_pred, y_prob)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


# ============================================================
# 打印对比结果表格
# ============================================================
def print_comparison_table(results: dict):
    """打印所有模型的结果对比表格"""
    print("\n" + "=" * 80)
    print("模型性能对比表")
    print("=" * 80)
    header = f"{'模型':<20} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'AUC':>10}"
    print(header)
    print("-" * 80)
    for model_name, metrics in results.items():
        row = (
            f"{model_name:<20} "
            f"{metrics.get('accuracy', float('nan')):>10.4f} "
            f"{metrics.get('f1', float('nan')):>10.4f} "
            f"{metrics.get('precision', float('nan')):>10.4f} "
            f"{metrics.get('recall', float('nan')):>10.4f} "
            f"{metrics.get('auc', float('nan')):>10.4f}"
        )
        print(row)
    print("=" * 80 + "\n")


# ============================================================
# 单元测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("对比模型单元测试")
    print("=" * 60)

    N_train, N_test, D = 1000, 200, 20
    X_train = np.random.randn(N_train, D).astype(np.float32)
    y_train = np.random.randint(0, 2, N_train)
    X_test  = np.random.randn(N_test, D).astype(np.float32)
    y_test  = np.random.randint(0, 2, N_test)

    all_results = {}

    # sklearn 模型
    for baseline in get_sklearn_baselines():
        baseline.fit(X_train, y_train)
        metrics = baseline.evaluate(X_test, y_test)
        all_results[baseline.name] = metrics
        print(f"[{baseline.name}] Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    # MLP
    mlp = MLPTrainer(in_channels=D, epochs=30, batch_size=128)
    mlp.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    metrics = mlp.evaluate(X_test, y_test)
    all_results[mlp.name] = metrics
    print(f"[MLP] Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    print_comparison_table(all_results)
