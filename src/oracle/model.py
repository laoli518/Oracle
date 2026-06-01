"""Trainable projection-head model over frozen CLIP features."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class CustomCLIPContrastiveTrainer(nn.Module):
    """
    两个可训练投影头；CLIP 本体冻结。
    图像和视频特征均为 512 维，image_projection 输入/输出均为 512。
    """

    def __init__(
        self,
        feature_dim: int = 512,
        temperature: float = 0.1,
        negative_weight: float = 0.3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = max(temperature, 0.01)
        self.negative_weight = negative_weight

        self.image_projection = self._create_projection_head(feature_dim, feature_dim)
        self.text_projection  = self._create_projection_head(feature_dim, feature_dim)
        self._initialize_weights()

        logger.info(
            f"投影头初始化: dim={feature_dim}→{feature_dim}, "
            f"T={temperature}, neg_w={negative_weight}"
        )

    def _create_projection_head(self, in_dim: int, out_dim: int):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def _initialize_weights(self):
        for module in [self.image_projection, self.text_projection]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

    def _safe_normalize(self, features):
        norms = torch.norm(features, p=2, dim=-1, keepdim=True)
        noise_mask = norms < 1e-8
        if noise_mask.any():
            features = features + noise_mask.float() * torch.randn_like(features) * 1e-6
            norms = torch.norm(features, p=2, dim=-1, keepdim=True)
        return features / (norms + 1e-8)

    def forward(self, image_feats: torch.Tensor, text_feats: torch.Tensor):
        img_out = self.image_projection(image_feats)
        txt_out = self.text_projection(text_feats)
        if torch.isnan(img_out).any() or torch.isinf(img_out).any():
            logger.error("投影后图像特征包含 NaN/Inf")
            return None, None
        if torch.isnan(txt_out).any() or torch.isinf(txt_out).any():
            logger.error("投影后文本特征包含 NaN/Inf")
            return None, None
        return self._safe_normalize(img_out), self._safe_normalize(txt_out)

    def compute_contrastive_loss(self, image_features, text_features, is_positive_pairs):
        try:
            device = image_features.device
            sim = torch.matmul(image_features, text_features.T) / self.temperature
            sim = torch.clamp(sim, -20, 20)

            pos_idx = torch.where(is_positive_pairs == 1)[0]
            if len(pos_idx) > 0:
                pos_sim = sim[pos_idx][:, pos_idx]
                pos_labels = torch.arange(len(pos_idx), device=device)
                positive_loss = F.cross_entropy(pos_sim, pos_labels)
            else:
                positive_loss = torch.tensor(0.0, device=device)

            neg_idx = torch.where(is_positive_pairs == 0)[0]
            if len(neg_idx) > 0:
                neg_diag = torch.diagonal(sim[neg_idx][:, neg_idx])
                negative_loss = torch.mean(torch.relu(neg_diag + 0.5))
            else:
                negative_loss = torch.tensor(0.0, device=device)

            total_loss = positive_loss + self.negative_weight * negative_loss
            total_loss = torch.clamp(total_loss, max=10.0)
            return total_loss, positive_loss, negative_loss

        except Exception as e:
            logger.error(f"损失计算异常: {e}")
            dev = image_features.device
            return (
                torch.tensor(float("inf"), device=dev),
                torch.tensor(0.0, device=dev),
                torch.tensor(0.0, device=dev),
            )
