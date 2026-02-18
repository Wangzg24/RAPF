import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import MiniImageNetFewShot
from model import ProtoNet
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os
import random

seed = 1234
random.seed(seed)  # python seed
os.environ['PYTHONHASHSEED'] = str(
    seed)  # 设置python哈希种子，for certain hash-based operations (e.g., the item order in a set or a dict）。seed为0的时候表示不用这个feature，也可以设置为整数。 有时候需要在终端执行，到脚本实行可能就迟了。
np.random.seed(
    seed)  # If you or any of the libraries you are using rely on NumPy, 比如Sampling，或者一些augmentation。 哪些是例外可以看https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(seed)  # 为当前CPU设置随机种子。 pytorch官网倒是说(both CPU and CUDA)
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed) # 使用多块GPU时，均设置随机种子
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # 设置为True时，cuDNN使用非确定性算法寻找最高效算法
torch.backends.cudnn.enabled = True  # pytorch使用CUDANN加速，即使用GPU加速



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置任务
N = 3
K = 1
Q = 1
B = 2
EPOCHS = 30

train_set = MiniImageNetFewShot('./data/mini_imagenet', 'train', N, K, Q, 2000)
val_set   = MiniImageNetFewShot('./data/mini_imagenet', 'val',   N, K, Q, 1000)
test_set  = MiniImageNetFewShot('./data/mini_imagenet', 'test',  N, K, Q, 1000)

train_loader = DataLoader(train_set, batch_size=B, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=B)
test_loader  = DataLoader(test_set, batch_size=B)

model = ProtoNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate(loader):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for support, query, labels in loader:
            support = support.to(device)
            query = query.to(device)
            labels = labels.to(device)

            logits = model(support, query, N, K, Q)
            logits = logits.view(-1, N)
            labels = labels.view(-1)

            pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean()
            total_acc += acc.item()

    return total_acc / len(loader)

best_val = 0.0
for epoch in range(EPOCHS):
    model.train()
    for support, query, labels in train_loader:
        support = support.to(device)
        query = query.to(device)
        labels = labels.to(device)

        logits = model(support, query, N, K, Q)
        logits = logits.view(-1, N)
        labels = labels.view(-1)

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_acc = evaluate(val_loader)
    print(f"Epoch {epoch+1}, Val Acc: {val_acc:.4f}")
    # 保存最佳模型
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "best_model"+str(N)+".pth")
        print("save best checkpoint")

# =========================
# 测试
# =========================
print("\nLoading best model...")
name = "best_model"+str(N)+".pth"
model.load_state_dict(torch.load(name))

test_acc = evaluate(test_loader)
print("Final Test Acc:", test_acc)
