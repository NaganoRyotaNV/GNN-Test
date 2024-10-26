# train.py

import torch
from model import GATViewDirectionClassifier
from dummy_data import create_graph_data_with_edges

# パラメータ設定
num_classes = 5  # front, front side, side, back side, back

# モデルの初期化
model = GATViewDirectionClassifier(num_classes)

# ダミーデータを使用してモデルを動作させる
data = create_graph_data_with_edges()
output = model(data)

print("Output:", output)
