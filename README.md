# 1. Conda 環境を新規作成

conda create -n view_direction_classification python=3.8 -y

# 2. 環境をアクティベート

conda activate view_direction_classification

# 3. PyTorch のインストール (CUDA 対応の場合はバージョン指定を調整)

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. PyTorch Geometric と依存ライブラリのインストール

conda install -y -c conda-forge numpy scipy tqdm
conda install -y -c conda-forge scikit-learn

pip install torch-geometric

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.**version**)").html

# 4.1 　確認

python -c "import torch_geometric; print('torch_geometric installed successfully')"

# 5. その他の必要なライブラリをインストール

pip install numpy matplotlib
