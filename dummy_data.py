# data_preparation.py

import torch
from torch_geometric.data import Data
from model import ResNetFeatureExtractor, compute_initial_edges

# ダミーデータでノードとエッジを持つグラフを作成
def create_graph_data_with_edges(num_nodes=5):
    # 各ノードに対応するダミー画像データ
    node_images = [torch.randn(3, 224, 224) for _ in range(num_nodes)]

    # ResNetで初期特徴ベクトルを抽出
    resnet_extractor = ResNetFeatureExtractor()
    node_features = torch.stack([resnet_extractor(img.unsqueeze(0)) for img in node_images]).squeeze(1)

    # View GCN風にエッジを計算
    edge_index = compute_initial_edges(node_features, threshold=0.5)

    # グラフデータを作成
    data = Data(x=node_images, edge_index=edge_index)
    return data
