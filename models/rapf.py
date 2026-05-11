import json
import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import itertools
from itertools import combinations
import random
import math
from collections import Counter
import numpy as np
from models.dyarea import BayesianDynamicBoundaryPrototypicalNetwork
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class RAPF(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, dot=False, relation_encoder=None, N=5, Q=1):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.dot = dot
        self.fc1 = nn.Linear(768, 768 * 2)

        # self.relation_encoder = relation_encoder
        self.hidden_size = 768

        self.dyarea = BayesianDynamicBoundaryPrototypicalNetwork(
                                                encoder_dim=768 * 2,
                                                num_classes=3,
                                                prior_strength=1.0
                                            )

    def __dist__(self, x, y, dim):
        # If False, use Euclidean distance.
        self.dot = False
        if self.dot:
            print("debug!")
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
        # return self.__dist__(S, Q.unsqueeze(2), 3)
    
    # Adding noise artificially
    def shuffle_n_dimension(self, x):
        """
        Shuffle the N-dimensional tensor (keeping B and D unchanged).

        parameter:
            x: Input tensor with shape [B, N, D]

        return:
            shuffled_x: The shuffled tensor, still in the shape [B, N, D]
        """
        B, N, D = x.shape

        idx = torch.argsort(torch.rand(B, N), dim=1)  # [B, N] Randomly shuffle features
        # idx = torch.tensor([[0,1,2,3,3],[0,1,2,3,3]]) # noise 20% shuffle features
        # idx = torch.tensor([[0,1,2,4,3],[0,1,2,4,3]]) # noise 40%
        # idx = torch.tensor([[0,1,3,4,2],[0,1,3,4,2]]) # noise 60%
        # idx = torch.tensor([[0,3,1,4,2],[0,3,1,4,2]]) # noise 80%
        # idx = torch.tensor([[3,0,1,4,2],[3,0,1,4,2]]) # noise 100%

        # Expand index dimensions to match tensors
        idx = idx.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]
        
    def forward(self, support, query, rel_txt, support_head_entity, support_till_entity, query_head_entity, query_till_entity, N, K, total_Q, label, flag):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''

        rel_gol, rel_loc = self.sentence_encoder(rel_txt, cat=False)  # # rel_gol [B*N, D]

        support_h, support_t, s_loc, s_gol = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query_h, query_t, q_loc, q_gol = self.sentence_encoder(query)  # (B * total_Q, D)

        support = torch.cat((support_h, support_t), -1) # [B*N*K, 2D]
        query = torch.cat((query_h, query_t), -1)
        # support = s_gol
        # query = q_gol

        # Double the hidden layer is required to stitch together solids.
        # support = support.view(-1, N, K, self.hidden_size)  # (B, N, K, D)
        # query = query.view(-1, total_Q, self.hidden_size)  # (B, total_Q, D)
        # support = torch.mean(support, 2)  # Calculate prototype for each class
        support = support.view(-1, N, 768 * 2)
        query = query.view(-1, total_Q, 768 * 2)
        # support = support.view(-1, N, 768)
        # query = query.view(-1, total_Q, 768)

        B = support.size(0)
        Q = int(query.size(1) / N)
        NQ = total_Q
        B = support.size(0)
        D = support.size(-1)

        rel_loc = torch.mean(rel_loc, 1)  # [B*N, D]
        rel_rep = torch.cat((rel_gol, rel_loc), -1)
        rel_rep = rel_rep.view(B, N, -1)
        
        support = support + rel_rep # [B, N, D]
        logits = self.dyarea(support, query)
        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred
