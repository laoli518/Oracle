#!/usr/bin/env python3
"""
Fight / No-Fight Binary Classification - Deep-Frozen + Residual Adapter Fine-tuning v12

Changes from v11:
  1. Few-shot selection strategy changed to nested/hierarchical:
       - First select shots_max (default 10) per class via k-Center Greedy
       - For each subsequent shots count (9→8→7→6→5), remove the most
         redundant sample per class from the previous set, so every smaller
         set is a strict subset of the larger one.
  2. Added `kcenter_find_redundant()` — removes the point closest to any
     other selected point (inverse of k-center greedy).
  3. Added `reduce_train_items()` — wraps redundant-removal per class.
  4. Outer loop now runs shots_max → shots_min (descending) instead of
     ascending, building each set from the previous one.
"""

import os, argparse, random, itertools, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import clip
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from itertools import combinations
from sklearn.metrics import (cohen_kappa_score, accuracy_score,
                             precision_recall_fscore_support)
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Global seed
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────────────────────────────────────────
# Labels and text descriptions
# ─────────────────────────────────────────────────────────────────────────────
LABEL_DESCRIPTIONS = {
    "Fight": {
        "positive": [
            "pigs fighting aggressively with each other",
            "pigs in aggressive physical conflict",
            "pigs biting and head-knocking in fight",
            "pigs pushing and charging aggressively",
            "pigs engaged in agonistic interaction",
            "pigs showing violent confrontation",
        ]
    },
    "No Fight": {
        "positive": [
            "pigs showing no visible aggression",
            "pigs in calm peaceful interaction",
            "pigs coexisting without fighting",
            "pigs displaying non-aggressive behavior",
            "pigs in harmonious social behavior",
            "pigs interacting without conflict",
        ]
    },
}

ALL_LABELS = ["Fight", "No Fight"]
LABEL_IDX  = {"Fight": 0, "No Fight": 1}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_annotations(path: str):
    ext = os.path.splitext(path)[1].lower()
    df  = pd.read_excel(path) if ext in {".xlsx", ".xls"} else pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"Annotation file columns: {list(df.columns)}")

    rename_map = {}
    for c in df.columns:
        if c == "\u6587\u4ef6\u540d":
            rename_map[c] = "filename"
        elif c == "\u4e00\u81f4\u6027":
            rename_map[c] = "consistency"
        elif "\u6807\u6ce8" in c:
            new = c.replace(" \u6807\u6ce8", "_label").replace("\u6807\u6ce8", "_label") \
                   .strip().replace(" ", "_")
            rename_map[c] = new
    df = df.rename(columns=rename_map)

    if "consistency" in df.columns:
        df["consistency"] = df["consistency"].replace(
            {"\u4e00\u81f4": "consistent", "\u4e0d\u4e00\u81f4": "inconsistent"}
        )

    FILE_ALIASES = {"filename", "file_name", "FileName", "filepath", "FilePath", "path"}
    file_col = next((c for c in df.columns if c in FILE_ALIASES), None)
    if file_col and file_col != "filename":
        df = df.rename(columns={file_col: "filename"})
    if "filename" not in df.columns:
        raise ValueError(f"Cannot find filename column. Columns: {list(df.columns)}")

    LABEL_ALIASES  = {"label", "Label", "class", "Class", "category", "Category"}
    label_col      = next((c for c in df.columns if c in LABEL_ALIASES), None)
    annotator_cols = [c for c in df.columns if c.endswith("_label")]

    if label_col is None:
        if not annotator_cols:
            raise ValueError(
                f"Cannot find label column or annotator columns.\n"
                f"Columns: {list(df.columns)}"
            )
        print(f"Annotator columns: {annotator_cols} → majority vote")

        def majority_vote(row):
            votes = [v for v in row if isinstance(v, str) and v.strip() in ALL_LABELS]
            return max(set(votes), key=votes.count) if votes else None

        df["label"] = df[annotator_cols].apply(majority_vote, axis=1)
        n_dropped   = df["label"].isna().sum()
        if n_dropped:
            print(f"  {n_dropped} rows dropped (no majority)")
        df = df.dropna(subset=["label"])
    elif label_col != "label":
        df = df.rename(columns={label_col: "label"})

    df = df[df["label"].isin(ALL_LABELS)].reset_index(drop=True)
    print(f"Valid samples: {len(df)} | "
          + " | ".join(f"{l}: {(df['label']==l).sum()}" for l in ALL_LABELS))
    return df, annotator_cols


def build_path_index(data_root: str):
    idx, valid_ext = {}, IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    for root, _, files in os.walk(data_root):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in valid_ext:
                idx[f] = os.path.join(root, f)
    return idx

# ─────────────────────────────────────────────────────────────────────────────
# Media loading
# ─────────────────────────────────────────────────────────────────────────────
def load_img(path, preprocess_fn, device):
    return preprocess_fn(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def sample_video_frames(path: str, n_frames: int = 3, seed: int = GLOBAL_SEED):
    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read frame count: {path}")
    rng     = random.Random(seed)
    indices = sorted(rng.choices(range(total), k=n_frames))
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read: {path}")
    return frames


def load_media(path, preprocess_fn, device, n_frames=3, frame_seed=GLOBAL_SEED):
    if os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS:
        tensors = [preprocess_fn(f).unsqueeze(0)
                   for f in sample_video_frames(path, n_frames, frame_seed)]
        return torch.cat(tensors, dim=0).to(device)
    return load_img(path, preprocess_fn, device)

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class ProjCLIPWithAdapter(nn.Module):
    def __init__(self, clip_model, dropout=0.1, ratio=0.2):
        super().__init__()
        self.clip_model = clip_model
        self.ratio      = ratio
        d               = clip_model.visual.output_dim

        self.image_projection = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d)
        ).float()
        self.text_projection = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d)
        ).float()
        hidden_d = d // 4
        self.img_adapter = nn.Sequential(
            nn.Linear(d, hidden_d, bias=False), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_d, d, bias=False)
        ).float()
        self.txt_adapter = nn.Sequential(
            nn.Linear(d, hidden_d, bias=False), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_d, d, bias=False)
        ).float()

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        self.image_projection.eval()
        self.text_projection.eval()
        return self

    def _norm(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    def encode_image(self, x):
        with torch.no_grad():
            raw       = self.clip_model.encode_image(x).float()
            base_feat = self.image_projection(raw)
        feat = base_feat + self.ratio * self.img_adapter(base_feat)
        return self._norm(feat).mean(dim=0, keepdim=True)

    def encode_text(self, toks):
        with torch.no_grad():
            raw       = self.clip_model.encode_text(toks).float()
            base_feat = self.text_projection(raw)
        return self._norm(base_feat + self.ratio * self.txt_adapter(base_feat))


def fresh_adapter_model(clip_model, base_state, device, dropout, ratio):
    m = ProjCLIPWithAdapter(clip_model, dropout=dropout, ratio=ratio).to(device)
    m.load_state_dict(base_state, strict=False)
    return m

# ─────────────────────────────────────────────────────────────────────────────
# Scheduler / optimizer
# ─────────────────────────────────────────────────────────────────────────────
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer(name, params, lr, weight_decay=1e-2):
    name = name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True,
                         weight_decay=weight_decay)
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay)

# ─────────────────────────────────────────────────────────────────────────────
# Inter-annotator kappa
# ─────────────────────────────────────────────────────────────────────────────
def compute_inter_annotator_kappa(ann_df, annotator_cols, exclude_filenames):
    held = ann_df[~ann_df["filename"].isin(exclude_filenames)].copy()
    print(f"\nInter-annotator kappa: {len(held)} held-out samples")

    pair_kappas = {}
    for ann_a, ann_b in combinations(annotator_cols, 2):
        valid = held[[ann_a, ann_b]].dropna()
        valid = valid[valid[ann_a].isin(ALL_LABELS) & valid[ann_b].isin(ALL_LABELS)]
        k     = cohen_kappa_score(valid[ann_a], valid[ann_b], labels=ALL_LABELS)
        pair_kappas[(ann_a, ann_b)] = k

    mean_kappa = float(np.mean(list(pair_kappas.values())))
    return pair_kappas, mean_kappa

# ─────────────────────────────────────────────────────────────────────────────
# Model evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_text_anchors(model, device):
    f_pos = model.encode_text(
        clip.tokenize(LABEL_DESCRIPTIONS["Fight"]["positive"]).to(device))
    n_pos = model.encode_text(
        clip.tokenize(LABEL_DESCRIPTIONS["No Fight"]["positive"]).to(device))
    return f_pos, n_pos


def softmax_predict(img_feat, f_pos, n_pos, temp=1.0):
    s_F = (img_feat @ f_pos.T).squeeze(0).mean().item()
    s_N = (img_feat @ n_pos.T).squeeze(0).mean().item()
    return ("Fight" if s_F / temp >= s_N / temp else "No Fight"), s_F, s_N


@torch.no_grad()
def evaluate(model, ann_df, path_index, preprocess, device,
             temp=1.0, exclude_paths=None, n_frames=3):
    """Returns (metrics_dict, per_sample_records_list)."""
    model.eval()
    exclude      = set(exclude_paths) if exclude_paths else set()
    f_pos, n_pos = get_text_anchors(model, device)

    preds_all,  gts_all  = [], []
    preds_cons, gts_cons = [], []
    records = []

    for _, row in ann_df.iterrows():
        fname = row["filename"]
        p     = path_index.get(fname)
        if p is None or p in exclude:
            continue
        try:
            media            = load_media(p, preprocess, device,
                                          n_frames=n_frames, frame_seed=GLOBAL_SEED)
            feat             = model.encode_image(media)
            pred, s_F, s_N   = softmax_predict(feat, f_pos, n_pos, temp)
            gt               = row["label"]
            is_consistent    = str(row.get("consistency", "")).strip().lower() == "consistent"

            preds_all.append(pred)
            gts_all.append(gt)
            if is_consistent:
                preds_cons.append(pred)
                gts_cons.append(gt)

            records.append({
                "filename":      fname,
                "ground_truth":  gt,
                "predicted":     pred,
                "correct":       pred == gt,
                "score_fight":   round(s_F, 6),
                "score_nofight": round(s_N, 6),
                "consistent":    is_consistent,
                "split":         "eval",
            })

        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    if not preds_all:
        return {}, records

    kappa = cohen_kappa_score(gts_all, preds_all, labels=ALL_LABELS)
    acc   = accuracy_score(gts_cons, preds_cons) if preds_cons else 0.0

    p_per, r_per, f_per, _ = precision_recall_fscore_support(
        gts_all, preds_all, labels=ALL_LABELS, zero_division=0)
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        gts_all, preds_all, labels=ALL_LABELS, average="macro", zero_division=0)

    metrics = {
        "kappa":               kappa,
        "n_kappa":             len(preds_all),
        "accuracy_consistent": acc,
        "n_acc":               len(preds_cons),
        "fight_precision":     float(p_per[0]),
        "fight_recall":        float(r_per[0]),
        "fight_f1":            float(f_per[0]),
        "nofight_precision":   float(p_per[1]),
        "nofight_recall":      float(r_per[1]),
        "nofight_f1":          float(f_per[1]),
        "macro_precision":     float(p_mac),
        "macro_recall":        float(r_mac),
        "macro_f1":            float(f_mac),
    }
    return metrics, records

# ─────────────────────────────────────────────────────────────────────────────
# Training engine
# ─────────────────────────────────────────────────────────────────────────────
def run_one(cfg, clip_model, base_state, train_items,
            ann_df, path_index, preprocess, device):
    set_seed(cfg["seed"])
    model = fresh_adapter_model(clip_model, base_state, device,
                                cfg["dropout"], cfg["ratio"])

    for p in model.parameters():             p.requires_grad = False
    for p in model.img_adapter.parameters(): p.requires_grad = True
    for p in model.txt_adapter.parameters(): p.requires_grad = True

    params       = [p for p in model.parameters() if p.requires_grad]
    opt          = build_optimizer(cfg["optimizer"], params, cfg["lr"])
    total_steps  = cfg["epochs"] * len(train_items)
    warmup_steps = int(total_steps * 0.1)
    scheduler    = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    scaler       = GradScaler()

    rng        = random.Random(cfg["seed"])
    best_loss  = float("inf")
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        rng.shuffle(train_items)
        ep_loss = 0.0

        for path, lbl in train_items:
            try:
                frame_seed = rng.randint(0, 9999)
                media      = load_media(path, preprocess, device,
                                        n_frames=cfg["n_frames"],
                                        frame_seed=frame_seed)
                target     = torch.tensor([LABEL_IDX[lbl]], device=device)

                with autocast():
                    f_pos, n_pos = get_text_anchors(model, device)
                    feat         = model.encode_image(media)
                    s_F          = (feat @ f_pos.T).mean()
                    s_N          = (feat @ n_pos.T).mean()
                    logits       = torch.stack([s_F, s_N]).unsqueeze(0) / cfg["temp"]
                    loss         = F.cross_entropy(logits, target)

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                ep_loss += loss.item()

            except Exception as e:
                print(f"  Training skip {os.path.basename(path)}: {e}")

        avg = ep_loss / max(len(train_items), 1)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    train_paths          = [p for p, _ in train_items]
    metrics, inf_records = evaluate(model, ann_df, path_index, preprocess, device,
                                    temp=cfg["temp"], exclude_paths=train_paths,
                                    n_frames=cfg["n_frames"])
    return metrics, best_loss, best_state, inf_records

# ─────────────────────────────────────────────────────────────────────────────
# k-Center Greedy few-shot selection  (used once for shots_max)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def kcenter_select(paths, model, preprocess, device, k, n_frames=3):
    """Select k maximally-diverse samples via greedy k-center."""
    paths = sorted(paths, key=os.path.basename)
    if len(paths) <= k:
        return paths
    feats, valid = [], []
    for p in paths:
        try:
            media = load_media(p, preprocess, device,
                               n_frames=n_frames, frame_seed=GLOBAL_SEED)
            feats.append(model.encode_image(media).squeeze(0).cpu())
            valid.append(p)
        except Exception:
            continue
    if len(valid) <= k:
        return valid

    feats = torch.stack(feats)
    dist  = (1 - (feats @ feats.T)).clamp(min=0).numpy()
    rng   = random.Random(GLOBAL_SEED)
    sel   = [rng.randint(0, len(valid) - 1)]
    min_d = dist[sel[0]].copy()
    while len(sel) < k:
        for s in sel:
            min_d[s] = -1
        nxt = int(np.argmax(min_d))
        sel.append(nxt)
        min_d = np.minimum(min_d, dist[nxt])
    return [valid[i] for i in sel]


# ─────────────────────────────────────────────────────────────────────────────
# NEW: find the most redundant sample (inverse k-center) — used for reduction
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def kcenter_find_redundant(paths, model, preprocess, device, n_frames=3):
    """
    Return the path that is most redundant within the current set.

    Strategy: the redundant point is the one with the *smallest* distance
    to its nearest neighbour — removing it causes the least loss in coverage.
    """
    if len(paths) <= 1:
        return paths[0] if paths else None

    feats, valid = [], []
    for p in sorted(paths, key=os.path.basename):   # deterministic order
        try:
            media = load_media(p, preprocess, device,
                               n_frames=n_frames, frame_seed=GLOBAL_SEED)
            feats.append(model.encode_image(media).squeeze(0).cpu())
            valid.append(p)
        except Exception:
            continue

    if len(valid) <= 1:
        return valid[0] if valid else None

    feats = torch.stack(feats)
    dist  = (1 - (feats @ feats.T)).clamp(min=0).numpy()
    np.fill_diagonal(dist, np.inf)                   # ignore self-distance

    # For each point: min distance to any other selected point
    min_dist_to_others = dist.min(axis=1)

    # The most redundant point has the smallest such distance
    redundant_idx = int(np.argmin(min_dist_to_others))
    removed = valid[redundant_idx]
    print(f"    Redundant point removed: {os.path.basename(removed)} "
          f"(min-dist={min_dist_to_others[redundant_idx]:.4f})")
    return removed


def reduce_train_items(train_items, model, preprocess, device, n_frames=3):
    """
    Remove one sample per class (the most redundant one).
    Returns a new list that is a strict subset of train_items.
    """
    by_class = defaultdict(list)
    for path, lbl in train_items:
        by_class[lbl].append(path)

    to_remove = set()
    for lbl, paths in by_class.items():
        if len(paths) > 1:
            removed = kcenter_find_redundant(paths, model, preprocess,
                                             device, n_frames=n_frames)
            if removed:
                to_remove.add(removed)
        else:
            print(f"  Warning: only 1 sample for class '{lbl}', cannot reduce further.")

    new_items = [(p, lbl) for p, lbl in train_items if p not in to_remove]
    return new_items


def build_train_items(ann_df, path_index, model, preprocess, device,
                      n_shots=10, n_frames=3):
    """Build the initial (maximum) few-shot set via k-center greedy."""
    if "consistency" not in ann_df.columns:
        raise ValueError("'consistency' column not found.")

    consistent_df = ann_df[ann_df["consistency"].str.strip().str.lower() == "consistent"]
    print(f"Consistent pool: {len(consistent_df)} samples "
          f"(Fight: {(consistent_df['label']=='Fight').sum()}, "
          f"No Fight: {(consistent_df['label']=='No Fight').sum()})")

    items = []
    for lbl in ALL_LABELS:
        cands = sorted(
            [path_index[r["filename"]]
             for _, r in consistent_df[consistent_df["label"] == lbl].iterrows()
             if r["filename"] in path_index],
            key=os.path.basename
        )
        print(f"  [{lbl}] consistent pool: {len(cands)} -> selecting {n_shots}")
        chosen = kcenter_select(cands, model, preprocess, device,
                                k=n_shots, n_frames=n_frames)
        items.extend((p, lbl) for p in chosen)
    return items

# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_shots_csv(train_items, output_dir, n_shots):
    rows = [{"filename": os.path.basename(p), "full_path": p, "label": lbl}
            for p, lbl in train_items]
    path = os.path.join(output_dir, f"shots_{n_shots}shot.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved shots  -> {path}")


def save_inference_csv(inf_records, output_dir, n_shots):
    path = os.path.join(output_dir, f"inference_{n_shots}shot.csv")
    pd.DataFrame(inf_records).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved inference -> {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────
def print_inter_annotator(pair_kappas, mean_kappa):
    print("\n" + "="*62)
    print("Inter-annotator Cohen's Kappa  (held-out samples only)")
    print("="*62)
    for (a, b), k in pair_kappas.items():
        print(f"  {a:<15} vs  {b:<15}  kappa = {k:.4f}")
    print(f"  {'Mean kappa':<34}         = {mean_kappa:.4f}")
    print("="*62)


def print_metrics(m, tag=""):
    print(f"\n--- Model Evaluation Metrics {tag} ---")
    print(f"  Cohen's kappa              : {m['kappa']:.4f}  (n={m['n_kappa']})")
    print(f"  Accuracy (consistent only) : {m['accuracy_consistent']*100:.2f}%"
          f"  (n={m['n_acc']})")
    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-score':>10}")
    print(f"  {'Fight':<12} {m['fight_precision']:>10.4f}"
          f" {m['fight_recall']:>10.4f} {m['fight_f1']:>10.4f}")
    print(f"  {'No Fight':<12} {m['nofight_precision']:>10.4f}"
          f" {m['nofight_recall']:>10.4f} {m['nofight_f1']:>10.4f}")
    print(f"  {'Macro':<12} {m['macro_precision']:>10.4f}"
          f" {m['macro_recall']:>10.4f} {m['macro_f1']:>10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fight/No-Fight Binary Classification Adapter Fine-tuning v12"
    )
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--model",      required=True,
                        help="Pretrained checkpoint (.pth)")
    parser.add_argument("--output_dir", default="results_fight_v12")
    parser.add_argument("--n_frames",   type=int, default=25)
    parser.add_argument("--shots_min",  type=int, default=5,
                        help="Minimum shots per class (default 5)")
    parser.add_argument("--shots_max",  type=int, default=10,
                        help="Maximum shots per class — selected first (default 10)")
    parser.add_argument("--quick",      action="store_true",
                        help="Quick mode: one hyperparameter set only")
    args = parser.parse_args()

    set_seed(GLOBAL_SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Global seed: {GLOBAL_SEED}  |  "
          f"Frames per video: {args.n_frames}")
    print(f"Shots range: {args.shots_max} → {args.shots_min} "
          f"(nested, each set ⊆ previous)\n")

    # ── Load data ────────────────────────────────────────────────────────────
    ann_df, annotator_cols = load_annotations(args.annotation)
    path_index             = build_path_index(args.data_root)
    ann_df                 = ann_df[ann_df["filename"].isin(path_index)] \
                               .reset_index(drop=True)
    n_cons = (ann_df["consistency"] == "consistent").sum() \
             if "consistency" in ann_df.columns else "N/A"
    print(f"Matched: {len(ann_df)} | Fight: {(ann_df['label']=='Fight').sum()} "
          f"| No Fight: {(ann_df['label']=='No Fight').sum()} "
          f"| Consistent: {n_cons}")

    # ── CLIP ─────────────────────────────────────────────────────────────────
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    ckpt   = torch.load(args.model, map_location=device)
    base_m = ProjCLIPWithAdapter(clip_model).to(device)
    base_m.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    base_state = {k: v.clone() for k, v in base_m.state_dict().items()}

    # ── Hyperparameter grid ──────────────────────────────────────────────────
    if args.quick:
        grid = [("adamw", 5e-3, 100, 0.1, 0.2, 2.0)]
    else:
        grid = list(itertools.product(
            ["adamw", "adam"],
            [1e-4, 3e-4, 5e-4],
            [50, 80],
            [0.1],
            [0.2, 0.4],
            [2.0],
        ))
    print(f"Hyperparameter combinations: {len(grid)}\n")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 1: Select shots_max samples per class using k-center greedy
    # ═════════════════════════════════════════════════════════════════════════
    print(f"{'='*70}")
    print(f"  Building initial {args.shots_max}-shot set via k-Center Greedy")
    print(f"{'='*70}")
    current_train_items = build_train_items(
        ann_df, path_index, base_m, preprocess, device,
        n_shots=args.shots_max, n_frames=args.n_frames
    )

    # ═════════════════════════════════════════════════════════════════════════
    # Step 2: Outer loop shots_max → shots_min (descending, nested)
    # ═════════════════════════════════════════════════════════════════════════
    all_shots_summary = []

    for n_shots in range(args.shots_max, args.shots_min - 1, -1):
        # current_train_items already holds exactly n_shots per class
        print(f"\n{'='*70}")
        print(f"  SHOTS = {n_shots}  ({n_shots} per class, {n_shots*2} total)")
        print(f"{'='*70}")

        shots_dir = os.path.join(args.output_dir, f"{n_shots}shot")
        os.makedirs(shots_dir, exist_ok=True)

        train_filenames = {os.path.basename(p) for p, _ in current_train_items}
        train_paths     = [p for p, _ in current_train_items]
        print(f"Few-shot set ({len(current_train_items)} samples): "
              f"{sorted(train_filenames)}")

        save_shots_csv(current_train_items, shots_dir, n_shots)

        # ── Inter-annotator kappa ────────────────────────────────────────────
        pair_kappas, mean_iaa_kappa = {}, float("nan")
        if annotator_cols:
            pair_kappas, mean_iaa_kappa = compute_inter_annotator_kappa(
                ann_df, annotator_cols, exclude_filenames=train_filenames
            )
            print_inter_annotator(pair_kappas, mean_iaa_kappa)

        # ── Hyperparameter search ────────────────────────────────────────────
        results = []
        best_kappa       = -1.0
        best_cfg         = None
        best_ft_state    = None
        best_metrics     = None
        best_inf_records = []

        for i, (opt_name, lr, epochs, dropout, ratio, temp) in enumerate(grid, 1):
            cfg = dict(optimizer=opt_name, lr=lr, epochs=epochs, dropout=dropout,
                       ratio=ratio, temp=temp, seed=GLOBAL_SEED,
                       n_frames=args.n_frames)
            t0  = time.time()
            print(f"[{i:3d}/{len(grid)}] {opt_name} LR={lr:.0e} EP={epochs} "
                  f"Ratio={ratio}...", end=" ", flush=True)
            try:
                set_seed(GLOBAL_SEED)
                metrics, loss_val, ft_state, inf_records = run_one(
                    cfg, clip_model, base_state, current_train_items,
                    ann_df, path_index, preprocess, device
                )
                print(f"kappa={metrics['kappa']:.4f}(n={metrics['n_kappa']})  "
                      f"acc={metrics['accuracy_consistent']*100:.1f}%"
                      f"(n={metrics['n_acc']},consistent)  "
                      f"Fight-F1={metrics['fight_f1']:.3f}  "
                      f"NoFight-F1={metrics['nofight_f1']:.3f}  "
                      f"({time.time()-t0:.1f}s)")

                results.append({
                    "n_shots":             n_shots,
                    "optimizer":           opt_name,
                    "lr":                  lr,
                    "epochs":              epochs,
                    "dropout":             dropout,
                    "ratio":               ratio,
                    "temp":                temp,
                    "kappa":               metrics["kappa"],
                    "n_kappa_samples":     metrics["n_kappa"],
                    "accuracy_consistent": metrics["accuracy_consistent"],
                    "n_acc_samples":       metrics["n_acc"],
                    "fight_precision":     metrics["fight_precision"],
                    "fight_recall":        metrics["fight_recall"],
                    "fight_f1":            metrics["fight_f1"],
                    "nofight_precision":   metrics["nofight_precision"],
                    "nofight_recall":      metrics["nofight_recall"],
                    "nofight_f1":          metrics["nofight_f1"],
                    "macro_precision":     metrics["macro_precision"],
                    "macro_recall":        metrics["macro_recall"],
                    "macro_f1":            metrics["macro_f1"],
                    "loss":                loss_val,
                    "mean_iaa_kappa":      mean_iaa_kappa,
                })

                if metrics["kappa"] > best_kappa:
                    best_kappa       = metrics["kappa"]
                    best_cfg         = cfg
                    best_ft_state    = ft_state
                    best_metrics     = metrics
                    best_inf_records = inf_records

            except Exception as e:
                print(f"FAILED: {e}")
                import traceback; traceback.print_exc()

        if not results:
            print(f"  All experiments failed for {n_shots}-shot.")
        else:
            df_res   = pd.DataFrame(results).sort_values("kappa", ascending=False)
            csv_path = os.path.join(shots_dir, f"results_{n_shots}shot.csv")
            df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")

            shot_records = [
                {"filename": os.path.basename(p), "ground_truth": lbl,
                 "predicted": lbl, "correct": True,
                 "score_fight": None, "score_nofight": None,
                 "consistent": None, "split": "train_shot"}
                for p, lbl in current_train_items
            ]
            save_inference_csv(shot_records + best_inf_records, shots_dir, n_shots)

            best_pth = os.path.join(shots_dir, f"best_model_{n_shots}shot.pth")
            torch.save({
                "model_state_dict": best_ft_state,
                "config":           best_cfg,
                "kappa":            best_kappa,
                "n_shots":          n_shots,
                "labels":           ALL_LABELS,
                "mean_iaa_kappa":   mean_iaa_kappa,
                "pair_iaa_kappas":  {f"{a}_vs_{b}": v
                                     for (a, b), v in pair_kappas.items()},
            }, best_pth)

            print_inter_annotator(pair_kappas, mean_iaa_kappa)
            print_metrics(best_metrics, tag=f"[Best config — {n_shots}-shot]")
            print(f"Best config : {best_cfg}")
            print(f"  CSV     -> {csv_path}")
            print(f"  Model   -> {best_pth}")

            all_shots_summary.append({
                "n_shots":             n_shots,
                "best_kappa":          best_kappa,
                "accuracy_consistent": best_metrics["accuracy_consistent"],
                "fight_f1":            best_metrics["fight_f1"],
                "nofight_f1":          best_metrics["nofight_f1"],
                "macro_f1":            best_metrics["macro_f1"],
                "mean_iaa_kappa":      mean_iaa_kappa,
            })

        # ── Reduce for next iteration (skip reduction after shots_min) ───────
        if n_shots > args.shots_min:
            print(f"\n  Reducing {n_shots}-shot → {n_shots-1}-shot "
                  f"(removing most redundant sample per class)...")
            current_train_items = reduce_train_items(
                current_train_items, base_m, preprocess, device,
                n_frames=args.n_frames
            )

    # ── Cross-shots summary ──────────────────────────────────────────────────
    if all_shots_summary:
        # Sort by n_shots ascending for readability
        summary_df   = pd.DataFrame(all_shots_summary).sort_values("n_shots")
        summary_path = os.path.join(args.output_dir, "summary_all_shots.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n{'='*70}")
        print("SUMMARY ACROSS ALL SHOTS")
        print(f"{'='*70}")
        print(summary_df.to_string(index=False))
        print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()