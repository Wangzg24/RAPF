from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel
from dyarea import BayesianDynamicBoundaryPrototypicalNetwork
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



class CLIPGCLPModel(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", topk_patches: int = 8, use_adapter: bool = True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.topk_patches = topk_patches
        self.use_adapter = use_adapter

        hidden_dim = self.clip.config.projection_dim
        if use_adapter:
            self.global_adapter = nn.Linear(hidden_dim, hidden_dim)
            self.local_adapter = nn.Linear(hidden_dim, hidden_dim)
            self.text_adapter = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.global_adapter = nn.Identity()
            self.local_adapter = nn.Identity()
            self.text_adapter = nn.Identity()


        self.dyarea = BayesianDynamicBoundaryPrototypicalNetwork(
            encoder_dim=512,
            num_classes=5,
            prior_strength=1.0
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_images(self, images: List):
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        global_feat = self.clip.get_image_features(pixel_values=pixel_values)
        global_feat = self.global_adapter(global_feat)
        # global_feat = F.normalize(global_feat, dim=-1)
        global_feat = global_feat

        vision_outputs = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
        patch_tokens = self.clip.visual_projection(patch_tokens)

        patch_scores = patch_tokens.norm(dim=-1)
        k = min(self.topk_patches, patch_tokens.size(1))
        topk_idx = torch.topk(patch_scores, k=k, dim=-1).indices
        batch_idx = torch.arange(patch_tokens.size(0), device=self.device).unsqueeze(-1)
        topk_patches = patch_tokens[batch_idx, topk_idx]
        local_feat = topk_patches.mean(dim=1)
        local_feat = self.local_adapter(local_feat)
        # local_feat = F.normalize(local_feat, dim=-1)
        local_feat = local_feat

        return global_feat, local_feat

    def encode_texts(self, texts: List[str]):
        prompts = [f"a photo of a {t}" for t in texts]
        text_inputs = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_feat = self.clip.get_text_features(**text_inputs)
        text_feat = self.text_adapter(text_feat)
        # text_feat = F.normalize(text_feat, dim=-1)
        text_feat = text_feat
        return text_feat

    @staticmethod
    def circumcenter_fusion(v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor, eps: float = 1e-6):
        """
        Calculate the circumcenter of the triangle (the point equidistant from the three vertices).
        Input shape: a three-vertex tensor of shape [B, N, D]
        Output circumcentered tensor of shape [B, N, D]
        """
        # Calculate edge vectors
        e1 = v2 - v1  # [B, N, D]
        e2 = v3 - v1  # [B, N, D]

        # Constructing orthogonal bases
        e1_norm = torch.norm(e1, dim=-1, keepdim=True)  # [B, N, 1]
        u = e1 / e1_norm  # [B, N, D]

        # Calculate the projection of e2 onto u.
        proj_e2_u = torch.sum(e2 * u, dim=-1, keepdim=True)  # [B, N, 1]
        v_ortho = e2 - proj_e2_u * u  # [B, N, D]
        v_norm = torch.norm(v_ortho, dim=-1, keepdim=True)  # [B, N, 1]
        v = v_ortho / v_norm  # [B, N, D]

        # Projected onto a two-dimensional plane
        a = e1_norm.squeeze(-1)  # The distance of v2 in the u direction [B, N]
        b = proj_e2_u.squeeze(-1)  # Projection of v3 in the u direction [B, N]
        c = torch.sum(e2 * v, dim=-1)  # Projection of v3 in the v direction [B, N]

        # Calculate the two-dimensional circumcenter coordinates
        c_x = a / 2  # [B, N]
        numerator = b ** 2 + c ** 2 - a * b  # [B, N]

        denominator = 2 * c
        c_y = numerator / denominator  # [B, N]

        # Reverse projection back to the original space
        circumcenter = v1 + c_x.unsqueeze(-1) * u + c_y.unsqueeze(-1) * v  # [B, N, D]
        return circumcenter, c

    def compute_episode_outputs(self, episode, mode: str = "gclp") -> Dict[str, torch.Tensor]:
        N, K = episode.support_labels.shape
        _, Q = episode.query_labels.shape

        s_global, s_local = self.encode_images(episode.support_images)
        q_global, q_local = self.encode_images(episode.query_images)
        r_text = self.encode_texts(episode.support_texts)

        D = s_global.size(-1)
        s_global = s_global.view(N, K, D)
        s_local = s_local.view(N, K, D)
        q_global = q_global.view(N, Q, D)
        q_local = q_local.view(N, Q, D)

        # s_global_mean = F.normalize(s_global.mean(dim=1), dim=-1)
        # s_local_mean = F.normalize(s_local.mean(dim=1), dim=-1)
        # r_text = F.normalize(r_text, dim=-1)
        s_global_mean = s_global.mean(dim=1)
        s_local_mean = s_local.mean(dim=1)
        r_text = r_text

        proto_plain = s_local_mean
        # proto_add = F.normalize(s_global_mean + s_local_mean + r_text, dim=-1)
        proto_add = s_global_mean + s_local_mean + r_text
        proto_gclp, c = self.circumcenter_fusion(s_local_mean, s_global_mean, r_text)

        if mode == "plain":
            proto = proto_plain
        elif mode == "add":
            proto = proto_add
        elif mode == "gclp":
            if torch.any(c < 1e-6):
                print('debug')
                proto = proto_add / 3
            else:
                proto = proto_gclp
        else:
            raise ValueError(f"Unknown mode: {mode}")

        q_local_flat = q_local.reshape(N * Q, D)
        logits, _ = self.dyarea(proto.unsqueeze(0).expand(1, -1, -1), q_local_flat.unsqueeze(0).expand(1, -1, -1))
        logits = logits.squeeze(0)

        # logits = -torch.cdist(q_local_flat, proto, p=2)
        targets = episode.query_labels.reshape(-1).to(self.device)
        pred = torch.argmax(logits, dim=-1)
        acc = (pred == targets).float().mean()
        loss = F.cross_entropy(logits, targets)

        return {
            "loss": loss,
            "acc": acc,
            "targets": targets,
            "pred": pred,
            "logits": logits,
            "support_global": s_global,
            "support_local": s_local,
            "support_text": r_text,
            "query_global": q_global,
            "query_local": q_local,
            "proto_plain": proto_plain,
            "proto_add": proto_add,
            "proto_gclp": proto_gclp,
            "proto_used": proto,
        }
