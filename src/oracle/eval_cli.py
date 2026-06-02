"""Command-line entry points for ORACLE description-based test evaluation."""

import argparse
import logging
import os
import random
from typing import Optional

import clip
import numpy as np
import torch

from .external_evaluation import (
    DescriptionManager,
    evaluate_full_description_margin,
    evaluate_pos_neg_margin_accuracy,
    evaluate_pos_neg_pair_accuracy,
    load_test_samples,
)
from .features import CLIPFeatureCache
from .model import CustomCLIPContrastiveTrainer
from .utils import _safe_ensure_dir, _save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _build_parser(protocol: str) -> argparse.ArgumentParser:
    if protocol == "test1":
        description = "ORACLE test1：Evaluate independent test samples using training-seen descriptions."
        default_output = "outputs/test1_seen_descriptions"
    else:
        description = "ORACLE test2：Evaluate independent test samples using training-unseen descriptions."
        default_output = "outputs/test2_unseen_descriptions"

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-path", required=True, help="Path to the .pth weights saved during training")
    parser.add_argument("--test-data", default="data/test.json", help="Independent test sample JSON file")
    if protocol == "test2":
        parser.add_argument(
            "--desc-file",
            default="data/descriptions/test2_unseen_descriptions_example.json",
            help="Description JSON not used during training, with fields positive_zs/negative_zs.",
        )
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--feature-cache", default=None, help="Optional media/text feature cache .pt file")
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--motion-alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def _run(protocol: str, argv: Optional[list[str]] = None) -> None:
    args = _build_parser(protocol).parse_args(argv)
    _safe_ensure_dir(args.output_dir)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluation protocol: %s", protocol)
    logger.info("device: %s", device)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    feature_dim = int(checkpoint.get("feature_dim", 512))
    num_frames = int(checkpoint.get("num_frames_per_video", args.num_frames))
    motion_alpha = float(checkpoint.get("motion_alpha", args.motion_alpha))

    model = CustomCLIPContrastiveTrainer(feature_dim=feature_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if protocol == "test1":
        positive = checkpoint.get("positive_descriptions", {})
        negative = checkpoint.get("negative_descriptions", {})
        if not positive:
            raise ValueError("The training descriptions were not saved in the model file, so test1 cannot be performed.")
        descriptions = DescriptionManager.from_checkpoint(positive, negative)
        description_source = "descriptions_saved_in_checkpoint"
    else:
        descriptions = DescriptionManager.from_file(args.desc_file)
        description_source = args.desc_file

    label_samples = load_test_samples(args.test_data, descriptions)

    logger.info("Loading the frozen CLIP encoder and computing test features...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)

    cache = CLIPFeatureCache(
        clip_model,
        preprocess,
        device,
        cache_file=args.feature_cache,
        num_frames_per_video=num_frames,
        motion_alpha=motion_alpha,
    )
    video_paths = [sample["media_path"] for values in label_samples.values() for sample in values if sample["is_video"]]
    image_paths = [sample["media_path"] for values in label_samples.values() for sample in values if not sample["is_video"]]
    cache.precompute_images(sorted(set(image_paths)), batch_size=args.batch_size)
    cache.precompute_videos(sorted(set(video_paths)))
    cache.precompute_texts(descriptions.all_texts())
    cache.save(args.feature_cache)

    pair_result = evaluate_pos_neg_pair_accuracy(
        model, label_samples, descriptions, cache.image_features, cache.text_features, device, args.batch_size
    )
    margin_result = evaluate_full_description_margin(
        model, label_samples, descriptions, cache.image_features, cache.text_features, device, args.batch_size
    )
    sample_margin_result = evaluate_pos_neg_margin_accuracy(
        model, label_samples, descriptions, cache.image_features, cache.text_features, device, args.batch_size
    )
    _save_json(sample_margin_result["errors"], os.path.join(args.output_dir, f"{protocol}_pos_neg_margin_errors.json"))

    results = {
        "evaluation_protocol": protocol,
        "description_protocol": "seen_during_training" if protocol == "test1" else "unseen_during_training",
        "description_source": description_source,
        "config": vars(args),
        "checkpoint_feature_dim": feature_dim,
        "effective_num_frames": num_frames,
        "effective_motion_alpha": motion_alpha,
        "pos_neg_accuracy": pair_result,
        "full_desc_margin": margin_result,
        "pos_neg_margin_accuracy": {key: value for key, value in sample_margin_result.items() if key != "errors"},
    }
    results_path = os.path.join(args.output_dir, f"{protocol}_results.json")
    _save_json(results, results_path)
    logger.info("%s Done. Result files:: %s", protocol, results_path)
    logger.info("Positive/negative pair accuracy: %.2f%%", pair_result["overall_acc"])
    logger.info("Per-sample accuracy Pos/Neg margin accuracy: %.2f%%", sample_margin_result["overall_accuracy"])


def main_test1() -> None:
    _run("test1")


def main_test2() -> None:
    _run("test2")


if __name__ == "__main__":
    main_test1()
