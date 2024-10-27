# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torchvision.models import resnet18

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet = resnet18(weights="IMAGENET1K_V1")
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # グレースケール対応
        self.resnet.fc = nn.Identity()  # 最終層をIdentityに変更して特徴を取得

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet(x)
        return x

class GATViewDirectionModel(nn.Module):
    def __init__(self, num_nodes=5, in_channels=512, hidden_channels=256, out_channels=128):
        super(GATViewDirectionModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.gat2 = GATConv(hidden_channels * 4, out_channels, heads=4, concat=False)
        self.fc = nn.Linear(out_channels, num_nodes)  # 視線方向のクラス数

    def forward(self, data):
        # 各ノードの特徴をResNetで抽出
        x = torch.stack([self.feature_extractor(img.unsqueeze(0)) for img in data.x])  # 4Dテンソルに変換
        x = x.squeeze(1)  # 不要な次元を削除
        
        # Attentionを用いたメッセージパッシング
        x = F.relu(self.gat1(x, data.edge_index))
        x = F.relu(self.gat2(x, data.edge_index))
        
        # グローバルプーリングによる特徴集約
        x = global_mean_pool(x, data.batch)
        
        return x

