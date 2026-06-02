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
    Precompute and cache CLIP features for all images/videos and text.

    Image and video feature dimensions are unified as D = 512:
    - Image: directly encoded with encode_image, normalized, and represented as D-dimensional features.
    - Video: uniformly sample frames, compute the mean and std of frame-level features,
            then fuse them as mean + motion_alpha * std, normalize, and represent them as D-dimensional features.
            motion_alpha = 0 reduces the video feature to the pure mean representation, equivalent to image features.
            motion_alpha > 0 preserves motion-amplitude information, which is useful for Fight recognition.

    The unified dimensionality fully resolves dimension inconsistency during batch stacking.
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
        self.motion_alpha = motion_alpha          # std fusion weight; 0 means pure mean

        self.image_features: Dict[str, torch.Tensor] = {}
        self.text_features: Dict[str, torch.Tensor] = {}

        if cache_file and os.path.exists(cache_file):
            self._load(cache_file)

    def _load(self, path: str):
        logger.info(f"Loading feature cache.: {path}")
        data = torch.load(path, map_location="cpu")
        self.image_features = data.get("image_features", {})
        self.text_features = data.get("text_features", {})
        logger.info(
            f"  image/video features: {len(self.image_features)}   "
            f"text features: {len(self.text_features)} "
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
            f"Feature cache has been saved.: {path}  "
            f"(Image/Video={len(self.image_features)}, Text={len(self.text_features)})"
        )

    def precompute_images(self, image_paths: List[str], batch_size: int = 256):
        """Image features: encode -> L2-normalize, with dimensionality D = 512."""
        to_compute = [p for p in image_paths if p not in self.image_features]
        if not to_compute:
            logger.info("All image features are already cached. Skipping.")
            return
        logger.info(f"Precomputing image features: {len(to_compute)} ")

        self.clip_model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, len(to_compute), batch_size), desc="Precomputing image features"):
                batch_paths = to_compute[start: start + batch_size]
                tensors, valid_paths = [], []
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        tensors.append(self.clip_preprocess(img))
                        valid_paths.append(p)
                    except Exception as e:
                        logger.warning(f"Failed to read image. {p}: {e}")
                if not tensors:
                    continue
                batch_t = torch.stack(tensors).to(self.device)
                feats = self.clip_model.encode_image(batch_t).float().cpu()
                # Image: directly L2-normalize, with dimensionality D.
                feats = F.normalize(feats, p=2, dim=-1)
                for path, feat in zip(valid_paths, feats):
                    self.image_features[path] = feat   # shape: (512,)

    def precompute_videos(self, video_paths: List[str]):
        """
        Video features: uniformly sample frames → compute mean and std → mean + motion_alpha * std → L2 normalization. 
        The output dimension is D (512), exactly matching the image feature dimension, with no further dimension mismatch problems.
        """
        to_compute = [p for p in video_paths if p not in self.image_features]
        if not to_compute:
            logger.info("All video features are already cached. Skipping.")
            return
        logger.info(
            f"Precomputing video features: {len(to_compute)} videos,"
            f"sampling {self.num_frames_per_video} frames per video,"
            f"fusion method = mean + {self.motion_alpha} * std -> 512-dimensional"
        )

        self.clip_model.eval()
        for vpath in tqdm(to_compute, desc="Precomputing video features"):
            frames = _extract_frames_evenly(
                Path(vpath), self.num_frames_per_video, self.clip_preprocess
            )
            if not frames:
                logger.warning(f"No frames could be extracted from the video. Skipping.: {vpath}")
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
            logger.info("All text features are already cached. Skipping.")
            return
        logger.info(f"Precomputing text features: {len(to_compute)} entries")

        self.clip_model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, len(to_compute), batch_size), desc="Precomputing text features"):
                batch_texts = to_compute[start: start + batch_size]
                try:
                    tokens = clip.tokenize(batch_texts).to(self.device)
                    feats = self.clip_model.encode_text(tokens).float().cpu()
                    feats = F.normalize(feats, p=2, dim=-1)
                    for text, feat in zip(batch_texts, feats):
                        self.text_features[text] = feat
                except Exception as e:
                    logger.warning(f"Failed to encode text batch: {e}")

    def get_image_feat(self, path: str) -> Optional[torch.Tensor]:
        return self.image_features.get(path)

    def get_text_feat(self, text: str) -> Optional[torch.Tensor]:
        return self.text_features.get(text)
