import torch
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class BayesianDynamicBoundaryPrototypicalNetwork(nn.Module):
    def __init__(self, encoder_dim, num_classes, prior_strength=1.0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_classes = num_classes
        self.prior_strength = prior_strength
        # self.boundary_margin = nn.Parameter(torch.tensor([1.0]))

        # ====== 初始化先验 ======
        self.prior_means = nn.Parameter(torch.zeros(num_classes, encoder_dim))
        self.prior_cov_factor = nn.Parameter(torch.eye(encoder_dim).unsqueeze(0).repeat(num_classes, 1, 1) * 0.1)

        # ====== 基于现有样本生成样本观测方差 ======
        self.cov_adapter = nn.Sequential(
            nn.Linear(encoder_dim, 768),
            nn.ReLU(),
            nn.Linear(768, encoder_dim),
            nn.Softplus()
        )
        # self.cov_adapter = nn.Sequential(
        #     nn.Linear(encoder_dim, 384),
        #     nn.ReLU(),
        #     nn.Linear(384, encoder_dim),
        #     nn.Softplus()
        # )

        # ====== 注册常量缓冲区 ======
        self.register_buffer('eye_matrix', torch.eye(encoder_dim))

    # ===========================================================
    # 🔹 稳定求逆函数：优先 Cholesky，异常时回退到伪逆
    # ===========================================================
    def robust_inverse(self, sigma, eps_chol=1e-4, eps_pinv=1e-3):
        try:
            L = torch.linalg.cholesky(sigma)
            sigma_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            sigma_inv = torch.linalg.pinv(sigma)
        return sigma_inv

    # ===========================================================
    # 🔹 确保正定性（轻微扰动）
    # ===========================================================
    def ensure_positive_definite(self, matrix, epsilon=1e-6):
        eye = self.eye_matrix.to(matrix.device)
        return matrix + epsilon * eye

    # ===========================================================
    # 🔹 贝叶斯协方差计算（完全向量化）
    # ===========================================================
    def compute_bayesian_covariance(self, support_embeddings):
        """
        Args:
            support_embeddings: [B, N, D]
        Returns:
            class_means: [B, N, D]
            class_covs:  [B, N, D, D]
        """
        B, N, D = support_embeddings.shape

        prior_means = self.prior_means.unsqueeze(0).expand(B, -1, -1)
        prior_cov = torch.bmm(
            self.prior_cov_factor,
            self.prior_cov_factor.transpose(1, 2)
        ).unsqueeze(0).expand(B, -1, -1, -1)

        # 样本统计量（K=1）
        sample_covs = self.eye_matrix.to(support_embeddings.device) * 0.01  # [D, D] 初始化较小的方差
        sample_covs = sample_covs.unsqueeze(0).unsqueeze(0).expand(B, N, D, D)

        # 通过MLP生成样本观测方差
        # sample_covs = self.cov_adapter(support_embeddings.view(-1, support_embeddings.size(-1))).view(B, N, -1) # [B, N, D]
        # sample_covs = torch.diag_embed(sample_covs, dim1=-2, dim2=-1)  # [B, N, D, D]
        posterior_means = (self.prior_strength * prior_means + support_embeddings) / (self.prior_strength + 1)
        posterior_covs = (self.prior_strength * prior_cov + sample_covs) / (self.prior_strength + 1)
        # posterior_means = support_embeddings
        # posterior_covs = sample_covs
        posterior_covs = self.ensure_positive_definite(posterior_covs)

        return posterior_means, posterior_covs

    # ===========================================================
    # 🔹 Bhattacharyya 距离（使用 robust_inverse）
    # ===========================================================
    def bhattacharyya_distance(self, mean1, cov1, mean2, cov2, eps=1e-6):
        """
        mean1, mean2: [..., D]
        cov1, cov2:   [..., D, D]
        return:       [...,]
        """

        # cov1 = self.ensure_positive_definite(cov1)
        # cov2 = self.ensure_positive_definite(cov2)

        sigma = (cov1 + cov2) / 2.0
        diff = mean1 - mean2

        # 使用稳定逆
        sigma_inv = self.robust_inverse(sigma)

        diff_unsq = diff.unsqueeze(-1)
        term1 = 0.125 * torch.matmul(
            diff_unsq.transpose(-2, -1),
            torch.matmul(sigma_inv, diff_unsq)
        ).squeeze(-1).squeeze(-1)

        # determinant ratio term
        det_sigma = torch.logdet(sigma)
        det_cov1 = torch.logdet(cov1)
        det_cov2 = torch.logdet(cov2)
        term2 = 0.5 * (det_sigma - 0.5 * (det_cov1 + det_cov2))

        res = term1 + term2
        return res

        # 在线标准化：基于当前批次的距离分布
        # dist_mean = res.mean().detach()
        # dist_std = res.std().detach() + 1e-8
        # normalized_dist = (res - dist_mean) / dist_std
        #
        # return normalized_dist


        # return term1 + term2
        # return term1

    # ===========================================================
    # 🔹 前向传播（完全向量化 + 稳定）
    # ===========================================================
    def forward(self, support_embeddings, query_embeddings):
        """
        support_embeddings: [B, N, D]
        query_embeddings:   [B, NQ, D]
        """
        B, N, D = support_embeddings.shape
        NQ = query_embeddings.shape[1]

        # Step 1: 贝叶斯更新
        class_means, class_covs = self.compute_bayesian_covariance(support_embeddings)

        # Step 2: 批量展开
        query_means = query_embeddings.unsqueeze(2).expand(B, NQ, N, D)

        # query_covs = self.eye_matrix.to(query_embeddings.device).unsqueeze(0).unsqueeze(0) * 1e-6
        query_covs = self.cov_adapter(query_embeddings.view(-1, query_embeddings.size(-1))).view(B, NQ, D)
        query_covs = torch.diag_embed(query_covs, dim1=-2, dim2=-1)  # [B, NQ, D, D]

        query_covs = query_covs.unsqueeze(2).expand(B, NQ, N, D, D)
        class_means_exp = class_means.unsqueeze(1).expand(B, NQ, N, D)
        class_covs_exp = class_covs.unsqueeze(1).expand(B, NQ, N, D, D)

        # Step 3: 批量 Bhattacharyya 距离
        distances = self.bhattacharyya_distance(query_means, query_covs, class_means_exp, class_covs_exp)
        logits = -distances  # smaller distance → higher logit
        return logits, 0.0
