# data_loader.py

import os
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ViewpointDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # グレースケールで読み込む
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(root_dir, test_size=0.2):
    viewpoints = ['back', 'backside', 'front', 'frontside', 'side']
    image_paths = []
    labels = []

    for label, viewpoint in enumerate(viewpoints):
        # jpg, jpeg, png, JPEGに対応
        path_jpg = os.path.join(root_dir, viewpoint, "*.jpg")
        path_jpeg = os.path.join(root_dir, viewpoint, "*.jpeg")
        path_JPEG = os.path.join(root_dir, viewpoint, "*.JPEG")
        path_png = os.path.join(root_dir, viewpoint, "*.png")
        
        images = glob(path_jpg) + glob(path_jpeg) + glob(path_JPEG) + glob(path_png)
        
        if not images:  # 画像が存在しない場合の警告
            print(f"No images found in {viewpoint} directory")
        image_paths.extend(images)
        labels.extend([label] * len(images))

    print(f"Total images found: {len(image_paths)}")  # 画像数を出力

    if len(image_paths) == 0:
        raise ValueError("No images found in the specified directories.")

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # グレースケール用の正規化
    ])

    train_dataset = ViewpointDataset(train_paths, train_labels, transform=transform)
    test_dataset = ViewpointDataset(test_paths, test_labels, transform=transform)

    return train_dataset, test_dataset
