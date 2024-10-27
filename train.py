# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_data
from model import GATViewDirectionModel
import sys
import numpy as np

# デバッグテストをコマンドライン引数から取得
debug_test = sys.argv[1] if len(sys.argv) > 1 else None

# エッジの初期構成: 視線方向の関係性を考慮
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 0, 4],  # ソースノード
    [1, 0, 2, 1, 3, 2, 4, 3, 4, 0]   # ターゲットノード
], dtype=torch.long)

# データのロード
root_dir = r"C:\VOCdevkit\JPEGset\check\crop\car"
train_dataset, _ = load_data(root_dir)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

# モデルと最適化の設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GATViewDirectionModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# コサイン類似度計算用
def calculate_cosine_similarity(features):
    if len(features) < 2:
        return 0.0
    sims = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            sim = cosine_similarity(
                features[i].detach().cpu().numpy().reshape(1, -1),
                features[j].detach().cpu().numpy().reshape(1, -1)
            )[0][0]
            sims.append(sim)
    return sum(sims) / len(sims) if sims else 0.0


# デバッグテストの関数
def initil_test():
    print("Running initial feature extraction test...")
    feature_extractor = model.feature_extractor  # モデル内の特徴抽出器
    features = {}
    labels = {}

    # 特徴抽出と同じラベル内・異なるラベル間の類似度計算
    for images, lbls in train_loader:
        images = images.to(device)
        batch_features = feature_extractor(images).cpu().numpy()  # バッチ内の全画像特徴を取得

        for i, label in enumerate(lbls):
            label = label.item()  # 各ラベルを整数に変換
            feature = batch_features[i].flatten()  # 特徴をフラット化

            if label not in features:
                features[label] = []
            features[label].append(feature)

    # 同じ視線方向内のコサイン類似度
    print("Same viewpoint cosine similarity:")
    for label, feats in features.items():
        if len(feats) < 2:
            continue
        sims = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                sim = cosine_similarity([feats[i]], [feats[j]])[0][0]
                sims.append(sim)
        print(f"Label {label} (same viewpoint) average cosine similarity: {np.mean(sims):.4f}")

    # 異なる視線方向間のコサイン類似度
    print("\nDifferent viewpoint cosine similarity:")
    different_viewpoint_sims = []
    viewpoint_labels = list(features.keys())
    for i in range(len(viewpoint_labels)):
        for j in range(i + 1, len(viewpoint_labels)):
            label_i, label_j = viewpoint_labels[i], viewpoint_labels[j]
            feats_i, feats_j = features[label_i], features[label_j]
            for feat_i in feats_i:
                for feat_j in feats_j:
                    sim = cosine_similarity([feat_i], [feat_j])[0][0]
                    different_viewpoint_sims.append(sim)
            print(f"Average cosine similarity between Label {label_i} and Label {label_j}: {np.mean(different_viewpoint_sims):.4f}")

def second_test():
    print("Running initial graph structure test...")
    for images, labels in train_loader:
        images = images.to(device)
        data = Data(x=images, edge_index=edge_index)
        print("Graph nodes (features):", data.x)
        print("Edge index:", data.edge_index)
        break

# train.pyの中にあるthird_test関数の修正
def third_test():
    print("Running 1st epoch cosine similarity test...")
    model.train()
    
    # edge_indexをデバイスに移動
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 0, 4],
        [1, 0, 2, 1, 3, 2, 4, 3, 4, 0]
    ], dtype=torch.long).to(device)
    
    for images, _ in train_loader:
        images = images.to(device)
        data = Data(x=images, edge_index=edge_index)
        output = model(data)
        
        # 同じ視線方向と異なる視線方向のコサイン類似度を計算
        same_view_sim = calculate_cosine_similarity(output[output == output[0]])
        diff_view_sim = calculate_cosine_similarity(output[output != output[0]])
        
        print(f"Same viewpoint cosine similarity: {same_view_sim:.4f}")
        print(f"Different viewpoint cosine similarity: {diff_view_sim:.4f}")
        break


def forth_test():
    print("Running 2nd epoch start node and edge verification...")
    for images, _ in train_loader:
        images = images.to(device)
        data = Data(x=images, edge_index=edge_index)
        print("Nodes at epoch 2 start:", data.x)
        print("Edges at epoch 2 start:", data.edge_index)
        break

# デバッグテストの実行
if debug_test == "initil_test":
    initil_test()
elif debug_test == "second_test":
    second_test()
elif debug_test == "third_test":
    third_test()
elif debug_test == "forth_test":
    forth_test()
else:
    # 学習ループ
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            data = Data(x=images, edge_index=edge_index)
            optimizer.zero_grad()
            output = model(data)
            cos_sim = calculate_cosine_similarity(output)
            loss = 1 - torch.tensor(cos_sim, requires_grad=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
