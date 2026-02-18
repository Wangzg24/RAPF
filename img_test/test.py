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

# fixed seed or random seed
random.seed(seed)  # python seed
os.environ['PYTHONHASHSEED'] = str(
    seed)  
np.random.seed(seed) 
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True 
torch.backends.cudnn.enabled = True 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tasks setting
N = 9
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

# =========================
# Test
# =========================
print("\nLoading best model...")
name = "best_model"+str(N)+".pth"
model.load_state_dict(torch.load(name))

test_acc = evaluate(test_loader)
print("Final Test Acc:", test_acc)
