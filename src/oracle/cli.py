"""Command-line entry point for training and evaluation."""

import argparse
import logging
import os
import random

import clip
import numpy as np
import torch

from .dataset import PigBehaviorDirectDataset
from .evaluation import (
    evaluate_validation_full_descriptions,
    evaluate_zero_shot_classification_direct,
)
from .model import CustomCLIPContrastiveTrainer
from .trainer import train_direct_model
from .utils import _safe_ensure_dir, _save_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="ORACLE Training entry point for pig behavior positive/negative description contrastive learning"
    )
    parser.add_argument("--train-data",          default="data/train-mp4.json")
    parser.add_argument("--val-data", "--test-data", dest="val_data", default="data/val.json", help="Validation set JSON used for model selection during training; --test-data is kept as a backward-compatible alias for older scripts.")
    parser.add_argument("--output-dir",          default="outputs/training")
    parser.add_argument("--feature-cache-train", default="outputs/training/train_features.pt")
    parser.add_argument("--feature-cache-val", "--feature-cache-test", dest="feature_cache_val", default="outputs/training/val_features.pt")
    parser.add_argument("--sample-cache-train",  default="outputs/training/train_samples.json")
    parser.add_argument("--sample-cache-val", "--sample-cache-test", dest="sample_cache_val", default="outputs/training/val_samples.json")
    parser.add_argument("--num-frames",    type=int,   default=25,
                        help="Number of uniformly sampled frames per video")
    parser.add_argument("--motion-alpha",  type=float, default=0.5,
                        help="Video motion information fusion weight：feat = mean + alpha*std (0=纯mean)")
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch-size",    type=int,   default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature",   type=float, default=0.07)
    parser.add_argument("--negative-weight", type=float, default=0.5)
    parser.add_argument("--negative-ratio",  type=float, default=0.5)
    parser.add_argument("--max-samples",   type=int,   default=None)
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    logger.info(f"Number of frames per video: {args.num_frames}，motion_alpha={args.motion_alpha}")

    logger.info("Loading CLIP ViT-B/32 ...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    logger.info("Building the training dataset...")
    train_dataset = PigBehaviorDirectDataset(
        args.train_data, clip_preprocess, clip_model, device,
        feature_cache_file=args.feature_cache_train,
        sample_cache_file=args.sample_cache_train,
        num_frames_per_video=args.num_frames,
        max_samples_per_label=args.max_samples,
        negative_sampling_ratio=args.negative_ratio,
        motion_alpha=args.motion_alpha,
    )
    logger.info("Building the testing dataset...")
    val_dataset = PigBehaviorDirectDataset(
        args.val_data, clip_preprocess, clip_model, device,
        feature_cache_file=args.feature_cache_val,
        sample_cache_file=args.sample_cache_val,
        num_frames_per_video=args.num_frames,
        negative_sampling_ratio=0.5,
        motion_alpha=args.motion_alpha,
    )

    # Image and video features are unified to 512 dimensions
    model = CustomCLIPContrastiveTrainer(
        feature_dim=512,
        temperature=args.temperature,
        negative_weight=args.negative_weight,
    ).to(device)

    history = train_direct_model(
        model, train_dataset, val_dataset, device,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        save_dir=args.output_dir,
    )

    # ── Final evaluation after training uses eval mode ───────────────────────────────────
    logger.info("Validation-set zero-shot classification evaluation using margin scoring...")
    # Zero-shot evaluation directly iterates over label_samples, does not depend on _flat, and does not require mode switching.
    zero_shot = evaluate_zero_shot_classification_direct(
        model, val_dataset, device, save_dir=args.output_dir
    )

    logger.info("Validation-set evaluation with all descriptions...")
    # evaluate_validation_full_descriptions 也直接遍历 label_samples，不依赖 _flat
    final_full_eval = evaluate_validation_full_descriptions(
        model, val_dataset, device, save_dir=args.output_dir
    )

    logger.info("=" * 60)
    logger.info(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
    logger.info(f"Zero-shot classification accuracy: {zero_shot['overall_accuracy']:.2f}%")

    _safe_ensure_dir(args.output_dir)
    results = {
        "training_history":      history,
        "validation_zero_shot_evaluation":  zero_shot,
        "validation_full_desc_eval":  final_full_eval,
        "model_config":          vars(args),
        "positive_descriptions": train_dataset.get_all_positive_descriptions(),
        "negative_descriptions": train_dataset.get_all_negative_descriptions(),
    }
    _save_json(results, os.path.join(args.output_dir, "direct_contrastive_results.json"))

    if history["best_val_acc"] > 0:
        model_path = os.path.join(args.output_dir, "best_direct_contrastive_model.pth")
        torch.save({
            "model_state_dict":      model.state_dict(),
            "feature_dim":           512,
            "motion_alpha":          args.motion_alpha,
            "num_frames_per_video":  args.num_frames,
            "training_history":      history,
            "positive_descriptions": train_dataset.get_all_positive_descriptions(),
            "negative_descriptions": train_dataset.get_all_negative_descriptions(),
        }, model_path)
        logger.info(f"model saved: {model_path}")

    logger.info(f"complete！Result directory: {args.output_dir}")
