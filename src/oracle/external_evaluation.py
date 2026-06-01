"""Independent test-set evaluation with interchangeable positive/negative descriptions."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from .constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from .model import CustomCLIPContrastiveTrainer

logger = logging.getLogger(__name__)


def normalize_label(label: str) -> str:
    """Normalize label spelling for case-insensitive matching."""
    return re.sub(r"\s+", " ", label.strip()).lower()


class DescriptionManager:
    """Read either training prompts or externally supplied zero-shot prompts.

    Accepted entries use ``positive``/``negative`` or
    ``positive_zs``/``negative_zs`` arrays.
    """

    def __init__(self, descriptions: Dict[str, dict]):
        self._descriptions = descriptions
        self._normalized_to_key = {
            normalize_label(key): key for key in descriptions
        }

    @classmethod
    def from_file(cls, path: str) -> "DescriptionManager":
        with open(path, "r", encoding="utf-8") as file:
            descriptions = json.load(file)
        if not isinstance(descriptions, dict):
            raise ValueError("描述文件顶层必须为 label 到描述字典的映射。")
        logger.info("从描述文件加载 %d 个标签: %s", len(descriptions), path)
        return cls(descriptions)

    @classmethod
    def from_checkpoint(
        cls,
        positive: Dict[str, List[str]],
        negative: Dict[str, List[str]],
    ) -> "DescriptionManager":
        labels = sorted(set(positive) | set(negative))
        descriptions = {
            label: {
                "positive": positive.get(label, []),
                "negative": negative.get(label, []),
            }
            for label in labels
        }
        return cls(descriptions)

    def resolve(self, label: str) -> Optional[str]:
        return self._normalized_to_key.get(normalize_label(label))

    def get_positive(self, label: str) -> List[str]:
        key = self.resolve(label) or label
        entry = self._descriptions.get(key, {})
        values = entry.get("positive_zs") or entry.get("positive") or []
        if not values:
            raise ValueError(f"标签 {label!r} 缺少正描述。")
        return values

    def get_negative(self, label: str) -> List[str]:
        key = self.resolve(label) or label
        entry = self._descriptions.get(key, {})
        values = entry.get("negative_zs") or entry.get("negative") or []
        if not values:
            raise ValueError(f"标签 {label!r} 缺少负描述。")
        return values

    def all_labels(self) -> List[str]:
        return list(self._descriptions.keys())

    def all_texts(self) -> List[str]:
        texts: List[str] = []
        for label in self.all_labels():
            texts.extend(self.get_positive(label))
            texts.extend(self.get_negative(label))
        return texts


def load_test_samples(
    test_json: str,
    description_manager: DescriptionManager,
) -> Dict[str, List[dict]]:
    """Load evaluable media samples grouped by labels in the description set."""
    with open(test_json, "r", encoding="utf-8") as file:
        raw_data = json.load(file)
    samples = raw_data if isinstance(raw_data, list) else raw_data.get("training_data", [])

    grouped: Dict[str, List[dict]] = {}
    skipped = 0
    unmatched = set()
    for item in tqdm(samples, desc="加载测试样本"):
        raw_label = item.get("original_label", "")
        label = description_manager.resolve(raw_label)
        if label is None:
            unmatched.add(raw_label)
            skipped += 1
            continue
        input_text = item.get("input", "")
        if ":" not in input_text:
            skipped += 1
            continue
        media_path = Path(input_text.split(":", 1)[1].strip())
        suffix = media_path.suffix.lower()
        if not media_path.exists() or suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            skipped += 1
            continue
        grouped.setdefault(label, []).append(
            {"media_path": str(media_path), "is_video": suffix in VIDEO_EXTENSIONS}
        )
    if unmatched:
        logger.warning("描述集中无法匹配的测试标签: %s", sorted(unmatched))
    logger.info(
        "测试样本加载完成: %d 条，%d 类；跳过 %d 条。",
        sum(len(values) for values in grouped.values()), len(grouped), skipped,
    )
    return grouped


def _project_texts(
    model: CustomCLIPContrastiveTrainer,
    texts: List[str],
    text_features: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    missing = [text for text in texts if text not in text_features]
    if missing:
        raise RuntimeError(f"缺少 {len(missing)} 条文本特征，请先运行文本特征预计算。")
    encoded = torch.stack([text_features[text] for text in texts]).to(device)
    return F.normalize(model.text_projection(encoded), p=2, dim=-1)


def _valid_media_features(samples: List[dict], media_features: Dict[str, torch.Tensor]):
    valid = [sample for sample in samples if sample["media_path"] in media_features]
    missing = len(samples) - len(valid)
    return valid, missing


def evaluate_pos_neg_pair_accuracy(
    model: CustomCLIPContrastiveTrainer,
    label_samples: Dict[str, List[dict]],
    descriptions: DescriptionManager,
    media_features: Dict[str, torch.Tensor],
    text_features: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """Classify true-label positive/negative text pairs using zero threshold."""
    model.eval()
    per_class: Dict[str, dict] = {}
    total_correct = total_pairs = skipped_media = 0
    with torch.no_grad():
        for label, samples in tqdm(label_samples.items(), desc="正/负对准确率"):
            valid, missing = _valid_media_features(samples, media_features)
            skipped_media += missing
            pos_text = _project_texts(model, descriptions.get_positive(label), text_features, device)
            neg_text = _project_texts(model, descriptions.get_negative(label), text_features, device)
            pos_correct = pos_total = neg_correct = neg_total = 0
            for start in range(0, len(valid), batch_size):
                paths = [sample["media_path"] for sample in valid[start:start + batch_size]]
                images = torch.stack([media_features[path] for path in paths]).to(device)
                projected = F.normalize(model.image_projection(images), p=2, dim=-1)
                pos_sims = projected @ pos_text.T
                neg_sims = projected @ neg_text.T
                pos_correct += int((pos_sims > 0).sum().item())
                pos_total += pos_sims.numel()
                neg_correct += int((neg_sims < 0).sum().item())
                neg_total += neg_sims.numel()
            per_class[label] = {
                "positive_acc": 100.0 * pos_correct / pos_total if pos_total else 0.0,
                "negative_acc": 100.0 * neg_correct / neg_total if neg_total else 0.0,
                "positive_correct": pos_correct,
                "positive_total": pos_total,
                "negative_correct": neg_correct,
                "negative_total": neg_total,
            }
            total_correct += pos_correct + neg_correct
            total_pairs += pos_total + neg_total
    return {
        "overall_acc": 100.0 * total_correct / total_pairs if total_pairs else 0.0,
        "evaluated_pairs": total_pairs,
        "skipped_media": skipped_media,
        "per_class": per_class,
    }


def evaluate_full_description_margin(
    model: CustomCLIPContrastiveTrainer,
    label_samples: Dict[str, List[dict]],
    descriptions: DescriptionManager,
    media_features: Dict[str, torch.Tensor],
    text_features: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """Compute per-class mean positive similarity, negative similarity, and margin."""
    model.eval()
    per_class: Dict[str, dict] = {}
    skipped_media = 0
    with torch.no_grad():
        for label, samples in tqdm(label_samples.items(), desc="全描述 margin"):
            valid, missing = _valid_media_features(samples, media_features)
            skipped_media += missing
            pos_text = _project_texts(model, descriptions.get_positive(label), text_features, device)
            neg_text = _project_texts(model, descriptions.get_negative(label), text_features, device)
            pos_values: List[float] = []
            neg_values: List[float] = []
            for start in range(0, len(valid), batch_size):
                paths = [sample["media_path"] for sample in valid[start:start + batch_size]]
                images = torch.stack([media_features[path] for path in paths]).to(device)
                projected = F.normalize(model.image_projection(images), p=2, dim=-1)
                pos_values.extend((projected @ pos_text.T).mean(dim=1).cpu().tolist())
                neg_values.extend((projected @ neg_text.T).mean(dim=1).cpu().tolist())
            pos_array = np.asarray(pos_values, dtype=float)
            neg_array = np.asarray(neg_values, dtype=float)
            per_class[label] = {
                "pos_sim_mean": float(pos_array.mean()) if pos_array.size else 0.0,
                "pos_sim_std": float(pos_array.std()) if pos_array.size else 0.0,
                "neg_sim_mean": float(neg_array.mean()) if neg_array.size else 0.0,
                "neg_sim_std": float(neg_array.std()) if neg_array.size else 0.0,
                "margin": float((pos_array - neg_array).mean()) if pos_array.size else 0.0,
                "sample_count": int(pos_array.size),
            }
    margins = [entry["margin"] for entry in per_class.values() if entry["sample_count"] > 0]
    return {
        "avg_margin": float(np.mean(margins)) if margins else 0.0,
        "skipped_media": skipped_media,
        "per_class": per_class,
    }


def evaluate_pos_neg_margin_accuracy(
    model: CustomCLIPContrastiveTrainer,
    label_samples: Dict[str, List[dict]],
    descriptions: DescriptionManager,
    media_features: Dict[str, torch.Tensor],
    text_features: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """Evaluate whether each sample is closer to its positive than negative prompts."""
    model.eval()
    per_class: Dict[str, dict] = {}
    errors: List[dict] = []
    correct = total = skipped_media = 0
    with torch.no_grad():
        for label, samples in tqdm(label_samples.items(), desc="逐样本 margin 准确率"):
            valid, missing = _valid_media_features(samples, media_features)
            skipped_media += missing
            pos_text = _project_texts(model, descriptions.get_positive(label), text_features, device)
            neg_text = _project_texts(model, descriptions.get_negative(label), text_features, device)
            per_class.setdefault(label, {"correct": 0, "total": 0})
            for start in range(0, len(valid), batch_size):
                batch = valid[start:start + batch_size]
                images = torch.stack([media_features[sample["media_path"]] for sample in batch]).to(device)
                projected = F.normalize(model.image_projection(images), p=2, dim=-1)
                mean_pos = (projected @ pos_text.T).mean(dim=1)
                mean_neg = (projected @ neg_text.T).mean(dim=1)
                decision = mean_pos > mean_neg
                for index, sample in enumerate(batch):
                    total += 1
                    per_class[label]["total"] += 1
                    if bool(decision[index].item()):
                        correct += 1
                        per_class[label]["correct"] += 1
                    else:
                        errors.append({
                            "media_path": sample["media_path"],
                            "true_label": label,
                            "mean_pos_sim": round(float(mean_pos[index].item()), 6),
                            "mean_neg_sim": round(float(mean_neg[index].item()), 6),
                            "margin": round(float((mean_pos[index] - mean_neg[index]).item()), 6),
                        })
    return {
        "overall_accuracy": 100.0 * correct / total if total else 0.0,
        "total_samples": total,
        "skipped_media": skipped_media,
        "per_class_accuracy": {
            label: 100.0 * values["correct"] / values["total"]
            for label, values in per_class.items() if values["total"]
        },
        "per_class_detail": per_class,
        "errors": errors,
    }


def evaluate_zero_shot_margin_classification(
    model: CustomCLIPContrastiveTrainer,
    label_samples: Dict[str, List[dict]],
    descriptions: DescriptionManager,
    media_features: Dict[str, torch.Tensor],
    text_features: Dict[str, torch.Tensor],
    device: torch.device,
) -> dict:
    """Select the label with the greatest positive-minus-negative prompt margin."""
    model.eval()
    labels = descriptions.all_labels()
    label_to_index = {label: index for index, label in enumerate(labels)}
    pos_text = {label: _project_texts(model, descriptions.get_positive(label), text_features, device) for label in labels}
    neg_text = {label: _project_texts(model, descriptions.get_negative(label), text_features, device) for label in labels}
    true_values: List[int] = []
    predicted_values: List[int] = []
    per_class: Dict[str, dict] = {}
    errors: List[dict] = []
    skipped_media = 0
    with torch.no_grad():
        for true_label, samples in tqdm(label_samples.items(), desc="零样本分类"):
            per_class.setdefault(true_label, {"correct": 0, "total": 0})
            for sample in samples:
                media_feature = media_features.get(sample["media_path"])
                if media_feature is None:
                    skipped_media += 1
                    continue
                projected = F.normalize(model.image_projection(media_feature.unsqueeze(0).to(device)), p=2, dim=-1)
                scores = {
                    candidate: float(((projected @ pos_text[candidate].T).mean() - (projected @ neg_text[candidate].T).mean()).item())
                    for candidate in labels
                }
                predicted_label = max(scores, key=scores.get)
                true_values.append(label_to_index[true_label])
                predicted_values.append(label_to_index[predicted_label])
                per_class[true_label]["total"] += 1
                if predicted_label == true_label:
                    per_class[true_label]["correct"] += 1
                else:
                    errors.append({
                        "media_path": sample["media_path"],
                        "true_label": true_label,
                        "pred_label": predicted_label,
                        "pred_score": round(scores[predicted_label], 6),
                        "true_score": round(scores[true_label], 6),
                    })
    return {
        "overall_accuracy": 100.0 * accuracy_score(true_values, predicted_values) if predicted_values else 0.0,
        "total_samples": len(predicted_values),
        "skipped_media": skipped_media,
        "per_class_accuracy": {
            label: 100.0 * values["correct"] / values["total"]
            for label, values in per_class.items() if values["total"]
        },
        "per_class_detail": per_class,
        "errors": errors,
    }
