"""Dataset construction for image/video-text contrastive pairs."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from .features import CLIPFeatureCache
from .prompts import DirectLabelMapper
from .utils import _safe_ensure_dir

logger = logging.getLogger(__name__)

class PigBehaviorDirectDataset(Dataset):
    """
    猪行为对比学习数据集（v5）。
    ─ 图像和视频特征维度统一为 512，batch stack 不再报错。
    ─ 支持训练模式（随机采样一条描述）和评估模式（展开全部正/负描述）。
      - set_train_mode(): 固定种子随机展开，供训练使用。
      - set_eval_mode():  对每个样本展开全部正/负描述，供评估使用。
    """

    def __init__(
        self,
        data_path: str,
        clip_preprocess,
        clip_model,
        device,
        feature_cache_file: Optional[str] = None,
        sample_cache_file: Optional[str] = None,
        num_frames_per_video: int = 8,
        max_samples_per_label: Optional[int] = None,
        negative_sampling_ratio: float = 0.5,
        motion_alpha: float = 0.5,
    ):
        self.data_path = data_path
        self.clip_preprocess = clip_preprocess
        self.negative_sampling_ratio = negative_sampling_ratio
        self.num_frames_per_video = num_frames_per_video
        self.motion_alpha = motion_alpha

        self.label_mapper = DirectLabelMapper()
        all_labels = self.label_mapper.get_all_labels()
        self.label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

        with open(data_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        if isinstance(loaded_data, dict):
            self.training_data = loaded_data.get("training_data", [])
        elif isinstance(loaded_data, list):
            self.training_data = loaded_data
        else:
            raise ValueError(f"不支持的数据格式: {type(loaded_data)}")

        if sample_cache_file and os.path.exists(sample_cache_file):
            logger.info(f"从缓存加载样本列表: {sample_cache_file}")
            with open(sample_cache_file, "r", encoding="utf-8") as f:
                self.label_samples = json.load(f)
            logger.info(f"样本列表加载完成: {len(self.label_samples)} 个标签")
        else:
            logger.info("整理样本（每个视频/图像=1个样本）...")
            self.label_samples = self._organize_by_label(max_samples_per_label)
            if sample_cache_file:
                _safe_ensure_dir(os.path.dirname(sample_cache_file) or ".")
                with open(sample_cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.label_samples, f, ensure_ascii=False, indent=2)
                logger.info(f"样本列表已缓存: {sample_cache_file}")

        self.feature_cache = CLIPFeatureCache(
            clip_model, clip_preprocess, device,
            cache_file=feature_cache_file,
            num_frames_per_video=num_frames_per_video,
            motion_alpha=motion_alpha,
        )
        self._precompute_all_features(clip_model, device)
        if feature_cache_file:
            self.feature_cache.save(feature_cache_file)

        # 特征维度始终为 512
        self.feat_dim = 512
        self.text_feat_dim = 512

        # 默认训练模式（随机采样描述）
        self._build_flat_samples_train()

        logger.info(
            f"数据集就绪: {len(self)} 条（含正/负）, {len(self.label_samples)} 个标签  "
            f"feat_dim={self.feat_dim}  motion_alpha={self.motion_alpha}"
        )
        self._print_dataset_stats()

    # ── 模式切换 ──────────────────────────────────────────────────────────────

    def set_train_mode(self, seed: int = 42):
        """
        训练模式：每个样本随机选取一条正描述，以 negative_sampling_ratio
        概率随机选取一条负描述。传入 seed 可实现可复现的逐 epoch 重采样。
        """
        self._build_flat_samples_train(seed=seed)
        logger.info(f"[Dataset] 切换为训练模式，共 {len(self._flat)} 条，seed={seed}")

    def set_eval_mode(self):
        """
        评估模式：每个样本展开全部正描述 + 全部负描述，
        确保对每个图像/视频的所有描述都进行评估。
        """
        self._build_flat_samples_eval()
        logger.info(f"[Dataset] 切换为评估模式，共 {len(self._flat)} 条（全量正/负）")

    # ── 内部构建方法 ──────────────────────────────────────────────────────────

    def _build_flat_samples_train(self, seed: int = 42):
        """训练模式：按指定随机种子展开正/负样本列表。"""
        rng = random.Random(seed)
        self._flat: List[dict] = []

        for label in sorted(self.label_samples.keys()):
            samples = self.label_samples[label]
            pos_descs = self.label_mapper.get_positive_descriptions(label)
            neg_descs = self.label_mapper.get_negative_descriptions(label)

            for sample in samples:
                # 每个样本随机选一条正描述
                self._flat.append({
                    "sample": sample,
                    "label": label,
                    "is_positive_pair": 1,
                    "text": rng.choice(pos_descs),
                })
                # 以 negative_sampling_ratio 概率随机选一条负描述
                if rng.random() < self.negative_sampling_ratio:
                    self._flat.append({
                        "sample": sample,
                        "label": label,
                        "is_positive_pair": 0,
                        "text": rng.choice(neg_descs),
                    })

        logger.info(f"_build_flat_samples_train 完成: {len(self._flat)} 条（含正/负）")

    def _build_flat_samples_eval(self):
        """
        评估模式：每个样本展开全部正描述 + 全部负描述。
        例如每个标签有 6 条正描述和 6 条负描述，则每个样本产生 12 条记录。
        """
        self._flat = []

        for label in sorted(self.label_samples.keys()):
            samples = self.label_samples[label]
            pos_descs = self.label_mapper.get_positive_descriptions(label)
            neg_descs = self.label_mapper.get_negative_descriptions(label)

            for sample in samples:
                # 展开全部正描述
                for text in pos_descs:
                    self._flat.append({
                        "sample": sample,
                        "label": label,
                        "is_positive_pair": 1,
                        "text": text,
                    })
                # 展开全部负描述
                for text in neg_descs:
                    self._flat.append({
                        "sample": sample,
                        "label": label,
                        "is_positive_pair": 0,
                        "text": text,
                    })

        if self.label_samples:
            first_label = next(iter(self.label_samples))
            n_desc = (
                len(self.label_mapper.get_positive_descriptions(first_label))
                + len(self.label_mapper.get_negative_descriptions(first_label))
            )
            logger.info(
                f"_build_flat_samples_eval 完成: {len(self._flat)} 条"
                f"（每样本 × {n_desc} 描述，全量正/负）"
            )
        else:
            logger.info("_build_flat_samples_eval 完成: 0 条（无样本）")

    # ── 组织样本 ──────────────────────────────────────────────────────────────

    def _organize_by_label(
        self, max_samples_per_label: Optional[int]
    ) -> Dict[str, List[dict]]:
        from collections import defaultdict
        tmp: Dict[str, List[dict]] = defaultdict(list)
        skipped = 0
        unrecognized: set = set()

        for sample in tqdm(self.training_data, desc="整理样本"):
            raw_label = sample.get("original_label", "")
            input_text = sample.get("input", "")

            original_label = self.label_mapper.resolve_label(raw_label)
            if original_label is None:
                unrecognized.add(raw_label)
                skipped += 1
                continue

            if not input_text:
                skipped += 1
                continue

            if max_samples_per_label and len(tmp[original_label]) >= max_samples_per_label:
                continue

            media_path: Optional[Path] = None
            if ":" in input_text:
                parts = input_text.split(":", 1)
                if len(parts) == 2:
                    media_path = Path(parts[1].strip())

            if not media_path or not media_path.exists():
                skipped += 1
                continue

            suffix = media_path.suffix.lower()
            if suffix not in IMAGE_EXTENSIONS and suffix not in VIDEO_EXTENSIONS:
                skipped += 1
                continue

            is_video = suffix in VIDEO_EXTENSIONS
            entry = sample.copy()
            entry["media_path"] = str(media_path)
            entry["original_label"] = original_label
            entry["is_video"] = is_video
            tmp[original_label].append(entry)

        if unrecognized:
            logger.warning(
                f"以下 label 无法识别（共 {len(unrecognized)} 种）: {sorted(unrecognized)}"
            )
        if skipped:
            logger.info(f"跳过了 {skipped} 个无效/未识别样本")

        for label in tmp:
            tmp[label].sort(key=lambda x: x["media_path"])

        return dict(tmp)

    def _precompute_all_features(self, clip_model, device):
        image_paths, video_paths = [], []
        for samples in self.label_samples.values():
            for s in samples:
                if s.get("is_video", False):
                    video_paths.append(s["media_path"])
                else:
                    image_paths.append(s["media_path"])

        if image_paths:
            self.feature_cache.precompute_images(list(set(image_paths)))
        if video_paths:
            self.feature_cache.precompute_videos(list(set(video_paths)))

        all_texts = []
        for label in self.label_mapper.get_all_labels():
            all_texts.extend(self.label_mapper.get_positive_descriptions(label))
            all_texts.extend(self.label_mapper.get_negative_descriptions(label))
        self.feature_cache.precompute_texts(all_texts)

    # ── Dataset 接口 ──────────────────────────────────────────────────────────

    def get_all_positive_descriptions(self) -> Dict[str, List[str]]:
        return {label: self.label_mapper.get_positive_descriptions(label)
                for label in self.label_mapper.get_all_labels()}

    def get_all_negative_descriptions(self) -> Dict[str, List[str]]:
        return {label: self.label_mapper.get_negative_descriptions(label)
                for label in self.label_mapper.get_all_labels()}

    def __len__(self):
        return len(self._flat)

    def __getitem__(self, idx):
        entry  = self._flat[idx]
        sample = entry["sample"]
        label  = entry["label"]
        text   = entry["text"]
        is_positive_pair = entry["is_positive_pair"]

        img_feat = self.feature_cache.get_image_feat(sample["media_path"])
        txt_feat = self.feature_cache.get_text_feat(text)

        if img_feat is None:
            img_feat = torch.zeros(self.feat_dim)
        if txt_feat is None:
            txt_feat = torch.zeros(self.text_feat_dim)

        return {
            "image_feat":       img_feat,
            "text_feat":        txt_feat,
            "text":             text,
            "label":            label,
            "label_idx":        self.label_to_idx.get(label, 0),
            "original_label":   sample.get("original_label", ""),
            "media_path":       sample["media_path"],
            "is_positive_pair": is_positive_pair,
        }

    def _print_dataset_stats(self):
        logger.info("数据集统计:")
        for label, samples in sorted(self.label_samples.items()):
            n_vid = sum(1 for s in samples if s.get("is_video", False))
            n_img = len(samples) - n_vid
            pos_n = len(self.label_mapper.get_positive_descriptions(label))
            neg_n = len(self.label_mapper.get_negative_descriptions(label))
            logger.info(
                f"  {label}: {len(samples)} 原始样本 (视频={n_vid}, 图像={n_img})"
                f"  正描述={pos_n}条  负描述={neg_n}条"
            )
        logger.info(f"  展开后总条数（训练模式/含正负）: {len(self._flat)}")
        logger.info(f"  每视频帧数: {self.num_frames_per_video}")
        logger.info(f"  负样本采样比例（训练模式）: {self.negative_sampling_ratio}")
        logger.info(f"  视频运动融合权重 motion_alpha: {self.motion_alpha}")
