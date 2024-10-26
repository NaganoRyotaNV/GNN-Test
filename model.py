# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torchvision.models import resnet18

# ResNetを用いた初期特徴抽出モジュール

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = resnet18(weights="IMAGENET1K_V1")  # 注意: "pretrained"ではなく"weights"を使用
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # 最終全結合層を除く
        self.fc = nn.Linear(512, 256)  # 出力を256次元に圧縮

    def forward(self, x):
        if x.dim() == 3:  # バッチ次元がない場合は追加
            x = x.unsqueeze(0)
        with torch.no_grad():
            x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# View GCNの手法を参考にしたエッジ接続の計算
def compute_initial_edges(node_features, threshold=0.5):
    edge_index = []
    num_nodes = node_features.size(0)
    
    # すべてのノードペア (i, j) に対してエッジ重みを計算
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            vi, vj = node_features[i], node_features[j]
            gij = torch.cat([vi, vj, vi - vj, torch.norm(vi - vj).view(1)])
            Sij = torch.sigmoid(torch.dot(gij, torch.ones_like(gij)))  # θsに相当するパラメータを学習可能に設定

            # エッジを閾値でフィルタ
            if Sij > threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])  # 無向グラフ

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

# GATを用いた視線方向分類モデル
class GATViewDirectionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GATViewDirectionClassifier, self).__init__()
        self.resnet = ResNetFeatureExtractor()
        self.gat1 = GATConv(256, 128, heads=4, concat=True)
        self.gat2 = GATConv(128 * 4, 64, heads=4, concat=True)
        self.fc = nn.Linear(64 * 4, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 各ノードの特徴をResNetで抽出
        x = torch.cat([self.resnet(img.unsqueeze(0)) for img in x], dim=0)  # torch.stackからtorch.catに変更
        
        # GAT層でメッセージパッシング
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))

        # グローバルプーリング (平均プーリング)
        x = torch.mean(x, dim=0)

        # 最終分類
        x = self.fc(x)
        return F.log_softmax(x, dim=0)