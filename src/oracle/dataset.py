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
    Pig behavior contrastive learning dataset (v5).
    - Image and video feature dimensions are unified to 512, so batch stacking no longer raises dimension mismatch errors.
    - Supports training mode, which randomly samples one description, and evaluation mode, which expands all positive and negative descriptions.
    - set_train_mode(): randomly expands descriptions with a fixed seed for training.
    - set_eval_mode(): expands all positive and negative descriptions for each sample for evaluation.
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
            raise ValueError(f"Unsupported data format: {type(loaded_data)}")

        if sample_cache_file and os.path.exists(sample_cache_file):
            logger.info(f"Loading the sample list from cache: {sample_cache_file}")
            with open(sample_cache_file, "r", encoding="utf-8") as f:
                self.label_samples = json.load(f)
            logger.info(f"Sample list loaded: {len(self.label_samples)} labels.")
        else:
            logger.info("Organizing samples, where each video/image is treated as one sample...")
            self.label_samples = self._organize_by_label(max_samples_per_label)
            if sample_cache_file:
                _safe_ensure_dir(os.path.dirname(sample_cache_file) or ".")
                with open(sample_cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.label_samples, f, ensure_ascii=False, indent=2)
                logger.info(f"Sample list has been cached: {sample_cache_file}")

        self.feature_cache = CLIPFeatureCache(
            clip_model, clip_preprocess, device,
            cache_file=feature_cache_file,
            num_frames_per_video=num_frames_per_video,
            motion_alpha=motion_alpha,
        )
        self._precompute_all_features(clip_model, device)
        if feature_cache_file:
            self.feature_cache.save(feature_cache_file)

        # Feature dimension: always 512.
        self.feat_dim = 512
        self.text_feat_dim = 512

        # Default training mode: randomly sample descriptions
        self._build_flat_samples_train()

        logger.info(
            f"Dataset ready: {len(self)} entries including positive and negative samples, {len(self.label_samples)} labels."
            f"feat_dim={self.feat_dim}  motion_alpha={self.motion_alpha}"
        )
        self._print_dataset_stats()

    # ── Mode switching ──────────────────────────────────────────────────────────────

    def set_train_mode(self, seed: int = 42):
        """
        Training mode: for each sample, one positive description is randomly selected, and
        one negative description is randomly selected with probability negative_sampling_ratio.
        Passing a seed enables reproducible per-epoch resampling.
        """
        self._build_flat_samples_train(seed=seed)
        logger.info(f"[Dataset] Switched to training mode: {len(self._flat)} entries, seed={seed}")

    def set_eval_mode(self):
        """
        Evaluation mode: for each sample, all positive descriptions and all negative
        descriptions are expanded, ensuring that every description associated with
        each image/video is evaluated.
        """
        self._build_flat_samples_eval()
        logger.info(f"[Dataset] Switched to eval mode: {len(self._flat)} entries (full positive/negative)")

    # ── Internal construction method ──────────────────────────────────────────────────────────

    def _build_flat_samples_train(self, seed: int = 42):
        """Training mode: expand the positive/negative sample list using the specified random seed."""
        rng = random.Random(seed)
        self._flat: List[dict] = []

        for label in sorted(self.label_samples.keys()):
            samples = self.label_samples[label]
            pos_descs = self.label_mapper.get_positive_descriptions(label)
            neg_descs = self.label_mapper.get_negative_descriptions(label)

            for sample in samples:
                # Randomly select one positive description for each sample.
                self._flat.append({
                    "sample": sample,
                    "label": label,
                    "is_positive_pair": 1,
                    "text": rng.choice(pos_descs),
                })
                # Randomly select one negative description with probability negative_sampling_ratio.
                if rng.random() < self.negative_sampling_ratio:
                    self._flat.append({
                        "sample": sample,
                        "label": label,
                        "is_positive_pair": 0,
                        "text": rng.choice(neg_descs),
                    })

        logger.info(f"_build_flat_samples_train completed: {len(self._flat)} entries including positive and negative samples.")

    def _build_flat_samples_eval(self):
        """
        Evaluation mode: for each sample, all positive descriptions and all negative
        descriptions are expanded.

        For example, if each label has 6 positive descriptions and 6 negative
        descriptions, each sample produces 12 records.
        """
        self._flat = []

        for label in sorted(self.label_samples.keys()):
            samples = self.label_samples[label]
            pos_descs = self.label_mapper.get_positive_descriptions(label)
            neg_descs = self.label_mapper.get_negative_descriptions(label)

            for sample in samples:
                # Expand all positive descriptions.
                for text in pos_descs:
                    self._flat.append({
                        "sample": sample,
                        "label": label,
                        "is_positive_pair": 1,
                        "text": text,
                    })
                # Expand all negative descriptions.
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
                f"build_flat_samples_eval completed: {len(self._flat)} entries."
                f"Per sample × {n_desc} descriptions, full positive/negative"
            )
        else:
            logger.info("_build_flat_samples_eval completed: 0 entries (no samples).")

    # ── Organizing samples ──────────────────────────────────────────────────────────────

    def _organize_by_label(
        self, max_samples_per_label: Optional[int]
    ) -> Dict[str, List[dict]]:
        from collections import defaultdict
        tmp: Dict[str, List[dict]] = defaultdict(list)
        skipped = 0
        unrecognized: set = set()

        for sample in tqdm(self.training_data, desc="Organizing samples"):
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
                f"The following labels could not be recognized: {len(unrecognized)} unique labels: {sorted(unrecognized)}"
            )
        if skipped:
            logger.info(f"Skipped {skipped} invalid or unrecognized samples.")

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

    # ── Dataset interface ──────────────────────────────────────────────────────────

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
        logger.info("Dataset statistics:")
        for label, samples in sorted(self.label_samples.items()):
            n_vid = sum(1 for s in samples if s.get("is_video", False))
            n_img = len(samples) - n_vid
            pos_n = len(self.label_mapper.get_positive_descriptions(label))
            neg_n = len(self.label_mapper.get_negative_descriptions(label))
            logger.info(
                f"  {label}: {len(samples)} original sample (video={n_vid}, image={n_img})"
                f"  Positive descriptions: {pos_n} entries  Negative descriptions: {neg_n} entries"
            )
        logger.info(f"  Total expanded entries (training mode, including positive/negative): {len(self._flat)}")
        logger.info(f"  Frames per video: {self.num_frames_per_video}")
        logger.info(f"  Negative sampling ratio (training mode): {self.negative_sampling_ratio}")
        logger.info(f"  Video motion fusion weight motion_alpha: {self.motion_alpha}")
