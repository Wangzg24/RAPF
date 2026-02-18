import torch
import torch.nn as nn
from dyarea import BayesianDynamicBoundaryPrototypicalNetwork

# =========================
# 1. Conv4 Backbone (Fast Version)
# =========================
class ConvEncoder(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.encoder = nn.Sequential(
            block(3, hidden),
            block(hidden, hidden),
            block(hidden, hidden),
            block(hidden, hidden),
        )

        # 🔥 关键：把 5x5 变成 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)              # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 64)
        return x


# =========================
# 2. Fast Euclidean Distance
# =========================
def euclidean_dist(x, y):
    """
    x: (B, num_query, D)
    y: (B, N, D)
    return: (B, num_query, N)
    """
    B, num_query, D = x.size()
    N = y.size(1)

    x = x.unsqueeze(2)  # (B, num_query, 1, D)
    y = y.unsqueeze(1)  # (B, 1, N, D)

    return ((x - y) ** 2).sum(dim=3)



class ProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder()
        # self.dyarea = BayesianDynamicBoundaryPrototypicalNetwork(
        #                                         encoder_dim=64,
        #                                         num_classes=7,
        #                                         prior_strength=1.0
        #                                     )

    def forward(self, support, query, N, K, Q):
        B, _, _, C, H, W = support.shape

        support = support.view(B*N*K, C, H, W)
        query = query.view(B*N*Q, C, H, W)

        support_emb = self.encoder(support)
        query_emb = self.encoder(query)

        D = support_emb.size(-1)

        support_emb = support_emb.view(B, N, K, D)
        query_emb = query_emb.view(B, N*Q, D)
        # print()
        prototypes = support_emb.mean(dim=2)

        # support_emb = support_emb.view(B, N, D)
        # logits = self.dyarea(support_emb, query_emb)
        
        logits = -euclidean_dist(query_emb, prototypes)

        return logits
