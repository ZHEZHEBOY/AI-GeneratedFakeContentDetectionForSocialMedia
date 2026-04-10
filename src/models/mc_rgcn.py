"""
MC-RGCN：蒙特卡洛 Dropout 优化的异质图卷积网络
主模型，用于 AI 合成图像检测

架构：
  输入特征 → RGCN Layer 1 → ReLU → MC-Dropout
  → RGCN Layer 2 → ReLU → MC-Dropout
  → 全连接输出层（2分类）

特点：
  - RGCN 支持多关系边（3种相似度关系）
  - 蒙特卡洛 Dropout：训练和预测时均保持 Dropout 激活
    → 多次前向传播取均值作为预测，方差估计不确定性
  - 节点级二分类（真实 vs AI合成）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from torch_geometric.nn import RGCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[WARN] torch-geometric 未安装，将使用手动实现的 RGCN")


# ============================================================
# 手动实现 RGCN（当 torch-geometric 不可用时）
# ============================================================
class ManualRGCNConv(nn.Module):
    """
    简化版 RGCN 卷积层（手动实现，无需 torch-geometric）
    对每种关系分别聚合邻居特征，再加权求和

    h_i' = W_0 * h_i + Σ_r ( (1/|N_r(i)|) Σ_{j∈N_r(i)} W_r * h_j )
    """
    def __init__(self, in_channels: int, out_channels: int, num_relations: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        # 自环权重
        self.W_self = nn.Linear(in_channels, out_channels, bias=False)
        # 每种关系的权重矩阵
        self.W_rel  = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False)
            for _ in range(num_relations)
        ])
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_type, num_nodes=None):
        """
        x:          [N, in_channels]
        edge_index: [2, E]
        edge_type:  [E] 每条边的关系类型
        """
        if num_nodes is None:
            num_nodes = x.size(0)

        out = self.W_self(x)  # 自环聚合

        for r in range(self.num_relations):
            mask   = edge_type == r
            if mask.sum() == 0:
                continue
            ei_r   = edge_index[:, mask]  # [2, E_r]
            src, dst = ei_r[0], ei_r[1]

            # 消息：W_r * h_src
            msg = self.W_rel[r](x[src])  # [E_r, out]

            # 聚合到 dst（均值聚合）
            agg = torch.zeros(num_nodes, self.out_channels, device=x.device)
            cnt = torch.zeros(num_nodes, 1, device=x.device)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
            cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(len(dst), 1, device=x.device))
            cnt = cnt.clamp(min=1.0)
            agg = agg / cnt

            out = out + agg

        return out + self.bias


# ============================================================
# MC-RGCN 主模型
# ============================================================
class MCRGCN(nn.Module):
    """
    蒙特卡洛 Dropout 优化的异质图卷积网络

    Args:
        in_channels:   输入特征维度（= PCA维度）
        hidden_channels: 隐层维度
        out_channels:  输出类别数（=2）
        num_relations: 关系数量（=3）
        dropout_rate:  Dropout 概率
        use_pyg:       是否使用 torch-geometric 的 RGCNConv
    """
    def __init__(
        self,
        in_channels:     int = 20,
        hidden_channels: int = 64,
        out_channels:    int = 2,
        num_relations:   int = 3,
        dropout_rate:    float = 0.3,
        use_pyg:         bool = True,
    ):
        super().__init__()
        self.dropout_rate  = dropout_rate
        self.num_relations = num_relations
        self.use_pyg       = use_pyg and HAS_PYG

        # 第一层 RGCN
        if self.use_pyg:
            self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)
            self.conv2 = RGCNConv(hidden_channels, hidden_channels // 2, num_relations=num_relations)
        else:
            self.conv1 = ManualRGCNConv(in_channels, hidden_channels, num_relations)
            self.conv2 = ManualRGCNConv(hidden_channels, hidden_channels // 2, num_relations)

        # 批归一化（稳定训练）
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels // 2)

        # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels // 2, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, out_channels),
        )

    def _mc_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """蒙特卡洛 Dropout：预测阶段也保持 dropout 激活"""
        return F.dropout(x, p=self.dropout_rate, training=True)  # 注意：始终 training=True

    def forward(self, x, edge_index, edge_type):
        """
        x:          [N, in_channels]
        edge_index: [2, E]
        edge_type:  [E]
        """
        # 第一层
        h = self.conv1(x, edge_index, edge_type)
        h = self.bn1(h)
        h = F.relu(h)
        h = self._mc_dropout(h)  # MC Dropout

        # 第二层
        h = self.conv2(h, edge_index, edge_type)
        h = self.bn2(h)
        h = F.relu(h)
        h = self._mc_dropout(h)  # MC Dropout

        # 分类输出
        out = self.classifier(h)
        return out  # [N, 2] logits

    def predict_with_uncertainty(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_type:  torch.Tensor,
        n_forward:  int = 50,
    ) -> tuple:
        """
        蒙特卡洛预测：多次前向传播，估计预测均值和不确定性

        Args:
            n_forward: MC 采样次数（通常 50~100）

        Returns:
            mean_probs:  [N, 2]  预测概率均值
            uncertainty: [N]     预测不确定性（熵或方差）
        """
        self.train()  # 保持 dropout 激活
        all_probs = []
        with torch.no_grad():
            for _ in range(n_forward):
                logits = self.forward(x, edge_index, edge_type)
                probs  = F.softmax(logits, dim=-1)
                all_probs.append(probs.unsqueeze(0))

        all_probs = torch.cat(all_probs, dim=0)  # [n_forward, N, 2]
        mean_probs = all_probs.mean(dim=0)        # [N, 2]

        # 预测熵作为不确定性指标
        eps = 1e-10
        uncertainty = -(mean_probs * (mean_probs + eps).log()).sum(dim=-1)  # [N]

        return mean_probs, uncertainty


# ============================================================
# 模型摘要工具
# ============================================================
def model_summary(model: MCRGCN):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 50}")
    print(f"MC-RGCN 模型参数摘要")
    print(f"{'=' * 50}")
    print(f"  总参数量：{total:,}")
    print(f"  可训练参数：{trainable:,}")
    for name, param in model.named_parameters():
        print(f"  {name}: {tuple(param.shape)}")
    print(f"{'=' * 50}\n")


# ============================================================
# 单元测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MC-RGCN 模型单元测试")
    print("=" * 60)

    # 构造随机数据
    N, D = 200, 20
    x          = torch.randn(N, D)
    y          = torch.randint(0, 2, (N,))
    edge_index = torch.randint(0, N, (2, 1000))
    edge_type  = torch.randint(0, 3, (1000,))

    # 初始化模型（不依赖 torch-geometric）
    model = MCRGCN(
        in_channels=D,
        hidden_channels=64,
        out_channels=2,
        num_relations=3,
        dropout_rate=0.3,
        use_pyg=False,  # 使用手动实现
    )
    model_summary(model)

    # 前向传播
    logits = model(x, edge_index, edge_type)
    print(f"前向传播输出：{logits.shape}，值域：[{logits.min():.3f}, {logits.max():.3f}]")

    # MC 不确定性估计
    mean_probs, uncertainty = model.predict_with_uncertainty(x, edge_index, edge_type, n_forward=20)
    print(f"MC 概率均值：{mean_probs.shape}，不确定性：{uncertainty.shape}")
    print(f"不确定性均值：{uncertainty.mean():.4f}，最大：{uncertainty.max():.4f}")

    # 计算 loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y)
    print(f"\nCross-Entropy Loss：{loss.item():.4f}")
    loss.backward()
    print("[✓] 反向传播正常")
