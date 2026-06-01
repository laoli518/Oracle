"""Frozen-CLIP media and text feature caching."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import clip
import torch
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

from .utils import _extract_frames_evenly, _safe_ensure_dir

logger = logging.getLogger(__name__)

class CLIPFeatureCache:
    """
    预计算并缓存所有图像/视频和文本的 CLIP 特征。

    图像与视频特征维度统一为 D（512）：
      - 图像：直接 encode_image，归一化，维度 D。
      - 视频：均匀抽取帧，计算帧特征的 mean 和 std，
              然后融合为 mean + motion_alpha * std，归一化，维度 D。
              motion_alpha=0 退化为纯均值（与图像等价）；
              motion_alpha>0 保留运动幅度信息（对 Fight 识别有帮助）。

    统一维度彻底消除 batch stack 时的维度不一致问题。
    """

    def __init__(
        self,
        clip_model,
        clip_preprocess,
        device,
        cache_file: Optional[str] = None,
        num_frames_per_video: int = 8,
        motion_alpha: float = 0.5,
    ):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device
        self.cache_file = cache_file
        self.num_frames_per_video = num_frames_per_video
        self.motion_alpha = motion_alpha          # std 融合权重，0 = 纯 mean

        self.image_features: Dict[str, torch.Tensor] = {}
        self.text_features: Dict[str, torch.Tensor] = {}

        if cache_file and os.path.exists(cache_file):
            self._load(cache_file)

    def _load(self, path: str):
        logger.info(f"加载特征缓存: {path}")
        data = torch.load(path, map_location="cpu")
        self.image_features = data.get("image_features", {})
        self.text_features = data.get("text_features", {})
        logger.info(
            f"  图像/视频特征: {len(self.image_features)} 条  "
            f"文本特征: {len(self.text_features)} 条"
        )

    def save(self, path: Optional[str] = None):
        path = path or self.cache_file
        if path is None:
            return
        _safe_ensure_dir(os.path.dirname(path) or ".")
        torch.save(
            {"image_features": self.image_features, "text_features": self.text_features},
            path,
        )
        logger.info(
            f"特征缓存已保存: {path}  "
            f"(图像/视频={len(self.image_features)}, 文本={len(self.text_features)})"
        )

    def precompute_images(self, image_paths: List[str], batch_size: int = 256):
        """图像特征：encode → L2 归一化，维度 D（512）。"""
        to_compute = [p for p in image_paths if p not in self.image_features]
        if not to_compute:
            logger.info("图像特征已全部缓存，跳过。")
            return
        logger.info(f"预计算图像特征: {len(to_compute)} 张")

        self.clip_model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, len(to_compute), batch_size), desc="图像特征预计算"):
                batch_paths = to_compute[start: start + batch_size]
                tensors, valid_paths = [], []
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        tensors.append(self.clip_preprocess(img))
                        valid_paths.append(p)
                    except Exception as e:
                        logger.warning(f"图像读取失败 {p}: {e}")
                if not tensors:
                    continue
                batch_t = torch.stack(tensors).to(self.device)
                feats = self.clip_model.encode_image(batch_t).float().cpu()
                # 图像：直接 L2 归一化，维度 D
                feats = F.normalize(feats, p=2, dim=-1)
                for path, feat in zip(valid_paths, feats):
                    self.image_features[path] = feat   # shape: (512,)

    def precompute_videos(self, video_paths: List[str]):
        """
        视频特征：均匀抽帧 → 计算 mean 和 std → mean + motion_alpha*std → L2 归一化。
        输出维度 D（512），与图像特征完全一致，不再有维度不匹配问题。
        """
        to_compute = [p for p in video_paths if p not in self.image_features]
        if not to_compute:
            logger.info("视频特征已全部缓存，跳过。")
            return
        logger.info(
            f"预计算视频特征: {len(to_compute)} 个，"
            f"每个抽取 {self.num_frames_per_video} 帧，"
            f"融合方式=mean + {self.motion_alpha}*std → 512维"
        )

        self.clip_model.eval()
        for vpath in tqdm(to_compute, desc="视频特征预计算"):
            frames = _extract_frames_evenly(
                Path(vpath), self.num_frames_per_video, self.clip_preprocess
            )
            if not frames:
                logger.warning(f"视频无法提取任何帧，跳过: {vpath}")
                continue
            with torch.no_grad():
                batch_t = torch.stack(frames).to(self.device)
                feats = self.clip_model.encode_image(batch_t).float()
                feats = F.normalize(feats, p=2, dim=-1)   # (T, 512)

                mean_feat = feats.mean(dim=0)              # (512,)

                if self.motion_alpha > 0.0 and feats.shape[0] > 1:
                    std_feat = feats.std(dim=0)            # (512,)
                    video_feat = mean_feat + self.motion_alpha * std_feat
                else:
                    video_feat = mean_feat

                video_feat = F.normalize(video_feat, p=2, dim=-1).cpu()

            self.image_features[vpath] = video_feat       # shape: (512,)

    def precompute_texts(self, texts: List[str], batch_size: int = 512):
        to_compute = list(set(t for t in texts if t not in self.text_features))
        if not to_compute:
            logger.info("文本特征已全部缓存，跳过。")
            return
        logger.info(f"预计算文本特征: {len(to_compute)} 条")

        self.clip_model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, len(to_compute), batch_size), desc="文本特征预计算"):
                batch_texts = to_compute[start: start + batch_size]
                try:
                    tokens = clip.tokenize(batch_texts).to(self.device)
                    feats = self.clip_model.encode_text(tokens).float().cpu()
                    feats = F.normalize(feats, p=2, dim=-1)
                    for text, feat in zip(batch_texts, feats):
                        self.text_features[text] = feat
                except Exception as e:
                    logger.warning(f"文本批次编码失败: {e}")

    def get_image_feat(self, path: str) -> Optional[torch.Tensor]:
        return self.image_features.get(path)

    def get_text_feat(self, text: str) -> Optional[torch.Tensor]:
        return self.text_features.get(text)
