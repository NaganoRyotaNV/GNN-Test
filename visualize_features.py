import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_data
from torchvision.models import resnet18

# 特徴抽出器
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet = resnet18(weights="IMAGENET1K_V1")
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # グレースケール対応
        self.resnet.fc = torch.nn.Identity()  # 512次元の特徴を取得

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet(x)
        return x

# データ準備
root_dir = r"C:\VOCdevkit\JPEGset\check\crop\car"  # carデータセットを使用
train_dataset, _ = load_data(root_dir)  # 訓練データのみを使用
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# 特徴抽出
feature_extractor = FeatureExtractor()
features = {}
labels = {}

for images, lbls in train_loader:
    label = lbls.item()
    feature = feature_extractor(images).cpu().numpy().flatten()
    if label not in features:
        features[label] = []
    features[label].append(feature)

# 同じ視線方向内のコサイン類似度を計算
print("Same viewpoint cosine similarity:")
for label, feats in features.items():
    if len(feats) < 2:
        continue  # 比較対象がなければスキップ
    sims = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            sim = cosine_similarity([feats[i]], [feats[j]])[0][0]
            sims.append(sim)
    print(f"Label {label} (same viewpoint) average cosine similarity: {np.mean(sims):.4f}")

# 異なる視線方向間のコサイン類似度を計算
print("\nDifferent viewpoint cosine similarity:")
different_viewpoint_sims = []
viewpoint_labels = list(features.keys())
for i in range(len(viewpoint_labels)):
    for j in range(i + 1, len(viewpoint_labels)):
        label_i, label_j = viewpoint_labels[i], viewpoint_labels[j]
        feats_i, feats_j = features[label_i], features[label_j]
        # 異なる視線方向間の類似度を計算
        for feat_i in feats_i:
            for feat_j in feats_j:
                sim = cosine_similarity([feat_i], [feat_j])[0][0]
                different_viewpoint_sims.append(sim)
        print(f"Average cosine similarity between Label {label_i} and Label {label_j}: {np.mean(different_viewpoint_sims):.4f}")
