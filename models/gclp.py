import os
import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import json

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy

import networkx as nx


class GCLP(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, dot=False, relation_encoder=None, N=5, Q=1):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.dot = dot
        self.fc1 = nn.Linear(768, 768 * 2)
        self.fc2 = nn.Linear(768 * 2, 3)

        self.relation_encoder = relation_encoder
        self.hidden_size = 768

      

        self.fc = nn.Linear(self.hidden_size, self.hidden_size * 2)





    def __dist__(self, x, y, dim):
        self.dot = False
        if self.dot:
            # print("debug!")
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
        # return self.__dist__(S, Q.unsqueeze(2), 3)

    

    def compute_circumcenter(self, v1, v2, v3, eps = float(1e-8)):
        """
        计算三角形的外心（到三个顶点等距的点）
        输入形状: [B, N, D] 的三个顶点张量
        输出形状: [B, N, D] 的外心张量
        """
        # 计算边向量
        e1 = v2 - v1  # [B, N, D]
        e2 = v3 - v1  # [B, N, D]

        # 构造正交基
        e1_norm = torch.norm(e1, dim=-1, keepdim=True) + eps  # [B, N, 1]
        u = e1 / e1_norm  # [B, N, D]

        # 计算 e2 在 u 上的投影
        proj_e2_u = torch.sum(e2 * u, dim=-1, keepdim=True)  # [B, N, 1]
        v_ortho = e2 - proj_e2_u * u  # [B, N, D]
        v_norm = torch.norm(v_ortho, dim=-1, keepdim=True) + eps  # [B, N, 1]
        v = v_ortho / v_norm  # [B, N, D]

        # 投影到二维平面
        a = e1_norm.squeeze(-1)  # v2 在 u 方向的距离 [B, N]
        b = proj_e2_u.squeeze(-1)  # v3 在 u 方向的投影 [B, N]
        c = torch.sum(e2 * v, dim=-1)  # v3 在 v 方向的投影 [B, N]

        # 计算二维外心坐标
        c_x = a / 2  # [B, N]
        numerator = b ** 2 + c ** 2 - a * b  # [B, N]

        denominator = 2 * c + eps  # 避免除以零
        c_y = numerator / denominator  # [B, N]
        
        # 求面积
        # area = 0.5 * a * torch.abs(c)  # [B, N]

        # 逆投影回原空间
        circumcenter = v1 + c_x.unsqueeze(-1) * u + c_y.unsqueeze(-1) * v  # [B, N, D]
        return circumcenter


    # 人为添加噪声
    def shuffle_n_dimension(self, x):
        """
        打乱张量的 N 维度（保持 B 和 D 不变）

        参数:
            x: 输入张量，形状为 [B, N, D]

        返回:
            shuffled_x: 打乱后的张量，形状仍为 [B, N, D]
        """
        B, N, D = x.shape

        # 为每个样本生成独立的随机排列索引
#         idx = torch.argsort(torch.rand(B, N), dim=1)  # [B, N]

#         # 扩展索引维度以匹配张量
#         idx = idx.unsqueeze(-1).expand(-1, -1, D)     # [B, N, D]

#         # 按索引重新排列张量
#         return x.gather(1, idx.cuda())
        
        # 均匀分布（范围 [0,1)）
        tensor_uniform = torch.rand(B, N, D)
        
        return tensor_uniform.cuda()


    def forward(self, support, query, rel_txt, N, K, total_Q, label):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        label: label of the query [BN]
        '''

        ##get relation
        # if self.relation_encoder:
        #    rel_gol, rel_loc = self.relation_encoder(rel_txt)
        # else:
        rel_gol, rel_loc = self.sentence_encoder(rel_txt, cat=False)  # # rel_gol [B*N, D]

        # support,  s_loc = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        # query,  q_loc = self.sentence_encoder(query) # (B * total_Q, D)

        support_h, support_t, s_loc, s_gol = self.sentence_encoder(
            support)  # (B * N * K, D), where D is the hidden size
        query_h, query_t, q_loc, q_gol = self.sentence_encoder(query)  # (B * total_Q, D)

        support = torch.cat((support_h, support_t), -1)
        query = torch.cat((query_h, query_t), -1)

#         # 拼接实体需要使用双倍隐藏层
        support = support.view(-1, N, K, self.hidden_size * 2)  # (B, N, K, D)
        # print(query.size())
        query = query.view(-1, total_Q, self.hidden_size * 2)  # (B, total_Q, D)
        
        # support = s_gol
        # query = q_gol


        Q = int(query.size(1) / N)
        NQ = total_Q
        B = support.size(0)
        D = support.size(-1)

        support_mean = torch.mean(support, 2)
        support_proto_ins = support_mean #[B, N, D]

        rel_loc = torch.mean(rel_loc, 1)  # [B*N, D]
        rel_rep = torch.cat((rel_gol, rel_loc), -1)
        rel_rep = rel_rep.view(B, N, -1)

        s_gol = self.fc1(s_gol).view(B, N, D) # [B, N, 768 * 2]
        
        
        # 人为打乱，增加噪声
        # s_gol = self.shuffle_n_dimension(s_gol)
        


        query = query.view(-1, NQ, D)

        # 计算外心（几何中心）
        support_proto = self.compute_circumcenter(support_proto_ins, s_gol, rel_rep)
        
        
        # support_proto = support_proto_ins + s_gol + rel_rep
        
        # support_proto = support_proto_ins + rel_rep





        triplet_loss = 0.0

        logits = self.__batch_dist__(support_proto, query)  # (B, total_Q, N)


        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        # return logits, pred, dist_all, dist_banjing
        return logits, pred, triplet_loss