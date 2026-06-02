"""I/O helpers and media-frame extraction utilities."""

import json
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def _safe_ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_json(obj, path: str):
    _safe_ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _save_per_epoch_metrics(save_dir: str, epoch_idx: int, metrics: dict):
    _safe_ensure_dir(save_dir)
    out_path = os.path.join(save_dir, f"per_class_acc_epoch_{epoch_idx}.json")
    _save_json(metrics, out_path)
    logger.info(f"Per-class positive/negative sample accuracy has been saved:: {out_path}")


# ── 视频帧提取工具 ────────────────────────────────────────────────────────────

def _extract_frames_evenly(
    video_path: Path,
    num_frames: int,
    clip_preprocess,
) -> List:
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        n = min(num_frames, total)
        indices = np.linspace(0, total - 1, n, dtype=int)

        tensors = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            tensors.append(clip_preprocess(img))
        cap.release()
        return tensors
    except Exception as e:
        logger.warning(f"Failed to extract video frames {video_path}: {e}")
        return []
