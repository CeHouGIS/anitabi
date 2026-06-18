"""
用 Place Pulse 2.0 数据集训练街景感知质量回归模型。

输入：  data/raw/place_pulse/（images/ + qscores.tsv + studies.tsv）
输出：  models/perception_net.pth   （模型权重）
        models/score_stats.json     （各维度归一化参数）

6 个感知维度：safety / lively / beautiful / wealthy / boring / depressing
对本研究最有用的是 lively 和 beautiful。
"""
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

warnings.filterwarnings('ignore')

# ── 路径 ──────────────────────────────────────────────────
DATA_DIR  = Path('data/raw/place_pulse')
MODEL_DIR = Path('models');  MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / 'perception_net.pth'
STATS_PATH = MODEL_DIR / 'score_stats.json'

# ── 超参数 ────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 128   # 更大 batch → 更少 GPU 调度次数，更快
EPOCHS     = 10    # 10 epoch 对此数据量已足够收敛
LR         = 1e-4
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

STUDY_MAP = {
    '50a68a51fdc9f05596000002': 'safety',
    '50f62c41a84ea7c5fdd2e454': 'lively',
    '50f62c68a84ea7c5fdd2e456': 'boring',
    '50f62cb7a84ea7c5fdd2e458': 'wealthy',
    '50f62ccfa84ea7c5fdd2e459': 'depressing',
    '5217c351ad93a7d3e7b07a64': 'beautiful',
}
ATTRS = ['safety', 'lively', 'boring', 'wealthy', 'depressing', 'beautiful']


# ── 数据准备 ──────────────────────────────────────────────
def load_scores():
    """读取 qscores.tsv，pivot 成 (location_id → 6维分数) 宽表。"""
    print("读取评分表…")
    q = pd.read_csv(DATA_DIR / 'qscores.tsv', sep='\t',
                    usecols=['location_id', 'study_id', 'trueskill.score'])
    q['attr'] = q['study_id'].map(STUDY_MAP)
    q = q.dropna(subset=['attr'])

    wide = q.pivot_table(index='location_id', columns='attr',
                         values='trueskill.score', aggfunc='mean')
    wide = wide.reindex(columns=ATTRS).dropna()
    print(f"  有完整6维评分的图片: {len(wide)}")
    return wide


def build_img_index():
    """建立 location_id → 图片路径 的索引。"""
    img_dir = DATA_DIR / 'images'
    index = {}
    for p in img_dir.iterdir():
        if p.suffix.upper() in ('.JPG', '.JPEG', '.PNG'):
            # 文件名格式: lat_lon_location_id_city.JPG
            parts = p.stem.split('_')
            if len(parts) >= 3:
                loc_id = parts[2]
                index[loc_id] = p
    print(f"  找到图片: {len(index)}")
    return index


class ListSampler(torch.utils.data.Sampler):
    """纯 Python 随机采样，避免 torch.randperm().numpy() 在 NumPy 2.x 下崩溃。"""
    def __init__(self, n, shuffle=True):
        self.n = n
        self.shuffle = shuffle

    def __iter__(self):
        import random
        idx = list(range(self.n))
        if self.shuffle:
            random.shuffle(idx)
        return iter(idx)

    def __len__(self):
        return self.n


class PlacePulseDataset(Dataset):
    def __init__(self, loc_ids, scores_df, img_index, transform):
        self.ids       = loc_ids
        self.scores    = scores_df
        self.img_index = img_index
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        lid = self.ids[i]
        path = self.img_index[lid]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        score = torch.tensor(list(self.scores.loc[lid, ATTRS]), dtype=torch.float32)
        return img, score


# ── 模型 ──────────────────────────────────────────────────
class PerceptionNet(nn.Module):
    def __init__(self, num_attrs=6):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 冻结前两个 stage，微调后三个
        for name, param in backbone.named_parameters():
            if name.startswith('layer1') or name.startswith('layer2') \
               or name.startswith('conv1') or name.startswith('bn1'):
                param.requires_grad = False
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_attrs),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ── 训练 ──────────────────────────────────────────────────
def normalize_scores(wide):
    """将各维度 TrueSkill 分归一化到 0-10。"""
    stats = {}
    wide_n = wide.copy()
    for col in ATTRS:
        mn, mx = wide[col].min(), wide[col].max()
        wide_n[col] = (wide[col] - mn) / (mx - mn + 1e-8) * 10
        stats[col] = {'min': float(mn), 'max': float(mx)}
    return wide_n, stats


def train():
    print(f"设备: {DEVICE}\n")

    # 1. 数据
    wide   = load_scores()
    index  = build_img_index()
    common = [lid for lid in wide.index if lid in index]
    print(f"  可用样本（有图+有评分）: {len(common)}")

    wide_n, stats = normalize_scores(wide.loc[common])
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    import random; random.seed(42); random.shuffle(common)
    split = int(len(common) * 0.8)
    train_ids, val_ids = common[:split], common[split:]
    print(f"  训练: {len(train_ids)}  验证: {len(val_ids)}\n")

    # 2. DataLoader
    # PILToTensor 不经过 numpy，避免 NumPy 2.x 兼容问题
    to_tensor = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_tf = T.Compose([
        T.Resize(256), T.RandomCrop(IMG_SIZE),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        to_tensor,
    ])
    val_tf = T.Compose([
        T.Resize(256), T.CenterCrop(IMG_SIZE),
        to_tensor,
    ])

    train_ds = PlacePulseDataset(train_ids, wide_n, index, train_tf)
    val_ds   = PlacePulseDataset(val_ids,   wide_n, index, val_tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=ListSampler(len(train_ds), shuffle=True),
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          sampler=ListSampler(len(val_ds), shuffle=False),
                          num_workers=0, pin_memory=False)

    # 3. 模型 & 优化器
    model = PerceptionNet().to(DEVICE)
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    criterion = nn.HuberLoss(delta=1.0)

    best_val = float('inf')
    for ep in range(1, EPOCHS + 1):
        # train
        model.train()
        t0 = time.time()
        tr_loss = 0.0
        for imgs, scores in train_dl:
            imgs, scores = imgs.to(DEVICE), scores.to(DEVICE)
            optim.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, scores)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * len(imgs)
        tr_loss /= len(train_ds)

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, scores in val_dl:
                imgs, scores = imgs.to(DEVICE), scores.to(DEVICE)
                pred = model(imgs)
                val_loss += criterion(pred, scores).item() * len(imgs)
        val_loss /= len(val_ds)
        scheduler.step()

        mark = ' ★' if val_loss < best_val else ''
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"Epoch {ep:2d}/{EPOCHS}  "
              f"train={tr_loss:.4f}  val={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"t={time.time()-t0:.0f}s{mark}")

    print(f"\n训练完成，最佳验证损失: {best_val:.4f}")
    print(f"模型保存: {MODEL_PATH}")
    print(f"归一化参数: {STATS_PATH}")


if __name__ == '__main__':
    train()
