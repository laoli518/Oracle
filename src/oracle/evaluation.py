"""Model evaluation routines."""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PigBehaviorDirectDataset
from .model import CustomCLIPContrastiveTrainer
from .utils import _safe_ensure_dir, _save_json

logger = logging.getLogger(__name__)

def collate_precomputed_batch(batch, device):
    image_feats = torch.stack([b["image_feat"] for b in batch]).to(device)
    text_feats  = torch.stack([b["text_feat"]  for b in batch]).to(device)
    label_indices = torch.tensor(
        [b["label_idx"] for b in batch], dtype=torch.long
    ).to(device)
    is_positive_pairs = torch.tensor(
        [b["is_positive_pair"] for b in batch], dtype=torch.long
    ).to(device)
    labels      = [b["label"]      for b in batch]
    media_paths = [b["media_path"] for b in batch]
    return image_feats, text_feats, label_indices, is_positive_pairs, labels, media_paths


# ── 评估 ──────────────────────────────────────────────────────────────────────

def evaluate_direct_model(
    model: CustomCLIPContrastiveTrainer,
    data_loader: DataLoader,
    device,
    save_similarity_values: bool = False,
    epoch_idx: Optional[int] = None,
    split_name: Optional[str] = None,
    save_dir: Optional[str] = None,
):
    """
    评估函数。
    调用前请确保 dataset 已切换为 eval 模式（set_eval_mode()），
    以便对每个样本的全部正/负描述进行评估。
    """
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    correct = 0
    total = 0
    per_class_counters: Dict[str, Dict[str, int]] = {}
    similarity_values: List[dict] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"评估[{split_name}]"):
            image_feats, text_feats, label_indices, is_positive_pairs, labels, media_paths = \
                collate_precomputed_batch(batch, device)

            proj_img, proj_txt = model(image_feats, text_feats)
            if proj_img is None:
                continue

            loss, _, _ = model.compute_contrastive_loss(proj_img, proj_txt, is_positive_pairs)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            total_loss += loss.item()
            valid_batches += 1

            sims = torch.diagonal(torch.matmul(proj_img, proj_txt.T))

            for i in range(len(is_positive_pairs)):
                lbl    = labels[i]
                is_pos = int(is_positive_pairs[i].item())
                sim_val = float(sims[i].item())
                is_correct = (sim_val > 0.0 and is_pos == 1) or (
                    sim_val < 0.0 and is_pos == 0
                )

                if lbl not in per_class_counters:
                    per_class_counters[lbl] = {
                        "pos_correct": 0, "pos_total": 0,
                        "neg_correct": 0, "neg_total": 0,
                    }
                if is_pos == 1:
                    per_class_counters[lbl]["pos_total"] += 1
                    if is_correct:
                        per_class_counters[lbl]["pos_correct"] += 1
                else:
                    per_class_counters[lbl]["neg_total"] += 1
                    if is_correct:
                        per_class_counters[lbl]["neg_correct"] += 1

                correct += int(is_correct)
                total   += 1
                similarity_values.append({
                    "media_path":       media_paths[i],
                    "similarity":       sim_val,
                    "correct":          bool(is_correct),
                    "label":            lbl,
                    "is_positive_pair": is_pos,
                })

    avg_loss    = total_loss / max(1, valid_batches)
    overall_acc = correct / max(1, total) * 100.0

    per_class_summary: Dict[str, dict] = {}
    for lbl, cnt in per_class_counters.items():
        per_class_summary[lbl] = {
            "positive_acc":     cnt["pos_correct"] / max(1, cnt["pos_total"]) * 100.0,
            "negative_acc":     cnt["neg_correct"] / max(1, cnt["neg_total"]) * 100.0,
            "positive_correct": cnt["pos_correct"],
            "positive_total":   cnt["pos_total"],
            "negative_correct": cnt["neg_correct"],
            "negative_total":   cnt["neg_total"],
        }

    if save_similarity_values and epoch_idx is not None and split_name and save_dir:
        out_path = os.path.join(
            save_dir, "similarity",
            f"{split_name}_similarity_values_epoch_{epoch_idx}.json",
        )
        _save_json(similarity_values, out_path)

    return overall_acc, avg_loss, per_class_summary, similarity_values


# ── 全描述验证评估 ────────────────────────────────────────────────────────────

def evaluate_validation_full_descriptions(
    model: CustomCLIPContrastiveTrainer,
    dataset: PigBehaviorDirectDataset,
    device,
    save_dir: Optional[str] = None,
    epoch_idx: Optional[int] = None,
) -> Dict[str, dict]:
    model.eval()
    feature_cache = dataset.feature_cache
    label_mapper  = dataset.label_mapper
    all_labels    = label_mapper.get_all_labels()

    def _get_proj_text_feats(descriptions: List[str]) -> torch.Tensor:
        feats = []
        for d in descriptions:
            f = feature_cache.get_text_feat(d)
            feats.append(f if f is not None else torch.zeros(model.feature_dim))
        feats_t = torch.stack(feats).to(device)
        with torch.no_grad():
            proj = model.text_projection(feats_t)
            proj = F.normalize(proj, p=2, dim=-1)
        return proj

    label_proj_pos: Dict[str, torch.Tensor] = {}
    label_proj_neg: Dict[str, torch.Tensor] = {}
    for lbl in all_labels:
        label_proj_pos[lbl] = _get_proj_text_feats(label_mapper.get_positive_descriptions(lbl))
        label_proj_neg[lbl] = _get_proj_text_feats(label_mapper.get_negative_descriptions(lbl))

    per_class_data: Dict[str, Dict[str, List[float]]] = {}

    with torch.no_grad():
        for lbl, samples in tqdm(dataset.label_samples.items(), desc="全描述验证评估"):
            if lbl not in label_proj_pos:
                continue
            paths = [s["media_path"] for s in samples]
            if lbl not in per_class_data:
                per_class_data[lbl] = {"pos_sims": [], "neg_sims": []}

            batch_size = 256
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start: start + batch_size]
                img_feat_list = []
                for p in batch_paths:
                    f = feature_cache.get_image_feat(p)
                    img_feat_list.append(
                        f if f is not None else torch.zeros(dataset.feat_dim)
                    )

                img_feats = torch.stack(img_feat_list).to(device)
                proj_img  = F.normalize(model.image_projection(img_feats), p=2, dim=-1)

                pos_sim = torch.matmul(proj_img, label_proj_pos[lbl].T)
                neg_sim = torch.matmul(proj_img, label_proj_neg[lbl].T)

                per_class_data[lbl]["pos_sims"].extend(pos_sim.mean(dim=1).cpu().tolist())
                per_class_data[lbl]["neg_sims"].extend(neg_sim.mean(dim=1).cpu().tolist())

    per_class_full_eval: Dict[str, dict] = {}
    for lbl, data in per_class_data.items():
        ps = np.array(data["pos_sims"])
        ns = np.array(data["neg_sims"])
        per_class_full_eval[lbl] = {
            "pos_sim_mean": float(ps.mean()) if len(ps) else 0.0,
            "pos_sim_std":  float(ps.std())  if len(ps) else 0.0,
            "neg_sim_mean": float(ns.mean()) if len(ns) else 0.0,
            "neg_sim_std":  float(ns.std())  if len(ns) else 0.0,
            "margin":       float(ps.mean() - ns.mean()) if len(ps) and len(ns) else 0.0,
            "sample_count": len(ps),
        }

    logger.info("── 全描述验证评估结果 ──────────────────────────")
    for lbl, stats in sorted(per_class_full_eval.items()):
        logger.info(
            f"  {lbl:30s}  pos_mean={stats['pos_sim_mean']:+.4f}  "
            f"neg_mean={stats['neg_sim_mean']:+.4f}  "
            f"margin={stats['margin']:+.4f}  n={stats['sample_count']}"
        )

    if save_dir is not None:
        tag = f"_epoch{epoch_idx}" if epoch_idx is not None else ""
        out_path = os.path.join(save_dir, f"val_full_desc_eval{tag}.json")
        _save_json(per_class_full_eval, out_path)
        logger.info(f"全描述验证评估已保存: {out_path}")

    return per_class_full_eval

def evaluate_zero_shot_classification_direct(
    model: CustomCLIPContrastiveTrainer,
    dataset: PigBehaviorDirectDataset,
    device,
    save_dir: Optional[str] = None,
):
    """
    零样本多分类评估。
    打分策略：score(label) = mean(pos_sim) - mean(neg_sim)
    此函数直接遍历 label_samples，与 _flat 模式无关。
    """
    model.eval()
    feature_cache = dataset.feature_cache
    label_mapper  = dataset.label_mapper
    all_labels    = label_mapper.get_all_labels()

    label_proj_pos: Dict[str, torch.Tensor] = {}
    label_proj_neg: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for lbl in all_labels:
            def _proj(descs: List[str]) -> torch.Tensor:
                feats = []
                for d in descs:
                    f = feature_cache.get_text_feat(d)
                    feats.append(f if f is not None else torch.zeros(model.feature_dim))
                t = torch.stack(feats).to(device)
                return F.normalize(model.text_projection(t), p=2, dim=-1)

            label_proj_pos[lbl] = _proj(label_mapper.get_positive_descriptions(lbl))
            label_proj_neg[lbl] = _proj(label_mapper.get_negative_descriptions(lbl))

    label_to_idx = dataset.label_to_idx
    all_predictions, all_true_labels = [], []
    detailed_results: Dict[str, Dict[str, int]] = {}
    error_records: List[dict] = []

    with torch.no_grad():
        for lbl, samples in tqdm(dataset.label_samples.items(), desc="零样本评估"):
            true_idx = label_to_idx.get(lbl, -1)
            for sample in samples:
                img_feat = feature_cache.get_image_feat(sample["media_path"])
                if img_feat is None:
                    continue
                img_t    = img_feat.unsqueeze(0).to(device)
                proj_img = F.normalize(model.image_projection(img_t), p=2, dim=-1)

                best_score = -float("inf")
                best_lbl   = None
                for cand_lbl in all_labels:
                    pos_sim = torch.matmul(proj_img, label_proj_pos[cand_lbl].T).mean().item()
                    neg_sim = torch.matmul(proj_img, label_proj_neg[cand_lbl].T).mean().item()
                    score   = pos_sim - neg_sim
                    if score > best_score:
                        best_score = score
                        best_lbl   = cand_lbl

                pred_idx = label_to_idx.get(best_lbl, -1)
                all_predictions.append(pred_idx)
                all_true_labels.append(true_idx)

                if lbl not in detailed_results:
                    detailed_results[lbl] = {"correct": 0, "total": 0}
                detailed_results[lbl]["total"] += 1
                if pred_idx == true_idx:
                    detailed_results[lbl]["correct"] += 1
                else:
                    error_records.append({
                        "media_path": sample["media_path"],
                        "true_label": lbl,
                        "pred_label": best_lbl,
                        "pred_score": round(best_score, 6),
                    })

    if not all_predictions:
        return {"overall_accuracy": 0.0, "label_accuracies": {}, "error_count": 0}

    overall_accuracy = accuracy_score(all_true_labels, all_predictions) * 100
    label_accuracies = {
        lbl: stats["correct"] / stats["total"] * 100
        for lbl, stats in detailed_results.items() if stats["total"] > 0
    }

    logger.info("── 零样本各类准确率 ──────────────────────────")
    for lbl in sorted(label_accuracies):
        acc   = label_accuracies[lbl]
        total = detailed_results[lbl]["total"]
        logger.info(f"  {lbl:30s}  {acc:6.2f}%  ({detailed_results[lbl]['correct']}/{total})")
    logger.info(f"  整体准确率: {overall_accuracy:.2f}%")

    if save_dir is not None:
        _safe_ensure_dir(save_dir)
        _save_json(error_records, os.path.join(save_dir, "zero_shot_errors.json"))
        logger.info(
            f"零样本错误记录已保存: {os.path.join(save_dir, 'zero_shot_errors.json')}"
            f"  (共 {len(error_records)} 条)"
        )

    return {
        "overall_accuracy": overall_accuracy,
        "label_accuracies": label_accuracies,
        "detailed_results": detailed_results,
        "error_count":      len(error_records),
    }
