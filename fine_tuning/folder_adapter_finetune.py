#!/usr/bin/env python3
"""
Folder-based ORACLE residual-adapter fine-tuning.

Dataset format:
  dataset_root/
  ├── train/ClassA, train/ClassB, ...
  └── test/ClassA,  test/ClassB,  ...

It loads a trained ORACLE .pth checkpoint, freezes CLIP encoders and ORACLE
projection heads, and trains only image/text residual adapters.
"""
from __future__ import annotations

import argparse, csv, json, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import clip
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".ts"}

PRESETS: Dict[str, Dict[str, object]] = {
    "fight_nofight": {
        "classes": ["Fight", "No Fight"],
        "prompts": {
            "Fight": [
                "pigs fighting aggressively with each other",
                "pigs in aggressive physical conflict",
                "pigs biting and head-knocking in fight",
                "pigs pushing and charging aggressively",
                "pigs engaged in agonistic interaction",
                "pigs showing violent confrontation",
            ],
            "No Fight": [
                "pigs showing no visible aggression",
                "pigs in calm peaceful interaction",
                "pigs coexisting without fighting",
                "pigs displaying non-aggressive behavior",
                "pigs in harmonious social behavior",
                "pigs interacting without conflict",
            ],
        },
    },
    "lying_posture": {
        "classes": ["Lateral Lying", "Sternal Lying", "Not Lying"],
        "prompts": {
            "Lateral Lying": [
                "a pig lying completely on its side",
                "a pig in lateral recumbent position",
                "a pig lying sideways with legs extended",
                "a pig resting on side",
                "a pig lying with full body contact on side",
                "a pig in side-lying position",
            ],
            "Sternal Lying": [
                "a pig lying on chest with legs tucked",
                "a pig in sternal recumbent position",
                "a pig resting on sternum like sphinx",
                "a pig lying on belly with head up",
                "a pig in chest-down position",
                "a pig prone on sternum",
            ],
            "Not Lying": [
                "a pig not lying down",
                "a pig in upright position not resting",
                "a pig standing or sitting but not lying",
                "a pig vertical not horizontal",
                "a pig upright not recumbent",
                "a pig active not lying down",
            ],
        },
    },
    "drinking_eating": {
        "classes": ["Drinking", "Eating", "Other"],
        "prompts": {
            "Drinking": [
                "a pig drinking water from nipple dispenser",
                "a pig using a nipple drinker",
                "a pig at the waterer taking a drink",
                "a pig lapping water from water source",
                "a pig accessing water for hydration",
                "a pig with mouth on water nipple",
            ],
            "Eating": [
                "a pig eating food from a trough",
                "a pig feeding at the feeder",
                "a pig with snout in the trough consuming feed",
                "a pig foraging and eating at feeding area",
                "a pig actively consuming feed",
                "a pig with head down eating from feeder",
            ],
            "Other": [
                "a pig not drinking water",
                "a pig showing no drinking behavior",
                "a pig not using the nipple drinker",
                "a pig not eating food",
                "a pig not consuming any feed",
                "a pig without access to food",
            ],
        },
    },
}

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_name(name: str) -> str:
    return " ".join(name.replace("_", " ").replace("-", " ").split()).lower()

def resolve_class_folder(split_dir: Path, class_name: str) -> Path:
    wanted = normalize_name(class_name)
    for child in split_dir.iterdir():
        if child.is_dir() and normalize_name(child.name) == wanted:
            return child
    existing = [p.name for p in split_dir.iterdir() if p.is_dir()]
    raise FileNotFoundError(f"Cannot find class folder for '{class_name}' under {split_dir}. Existing: {existing}")

def list_media_files(class_dir: Path) -> List[Path]:
    valid_ext = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    return sorted([p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_ext], key=lambda p: str(p).lower())

@dataclass
class FolderSample:
    path: Path
    label: str
    label_idx: int

class FolderBehaviourDataset(Dataset):
    def __init__(self, root: Path, split: str, classes: Sequence[str]):
        self.root, self.split, self.classes = Path(root), split, list(classes)
        self.label_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.samples: List[FolderSample] = []
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for label in self.classes:
            class_dir = resolve_class_folder(split_dir, label)
            files = list_media_files(class_dir)
            if not files:
                raise RuntimeError(f"No media files found in {class_dir}")
            self.samples.extend(FolderSample(p, label, self.label_to_idx[label]) for p in files)
        print(f"[{split}] total samples: {len(self.samples)}")
        for label in self.classes:
            print(f"  {label}: {sum(1 for s in self.samples if s.label == label)}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int) -> FolderSample: return self.samples[idx]

def sample_video_frames(path: Path, preprocess, n_frames: int, seed: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); raise RuntimeError(f"Cannot read video frame count: {path}")
    n = min(n_frames, total)
    rng = random.Random(seed + abs(hash(str(path))) % 100000)
    indices = sorted(rng.sample(range(total), k=n)) if total > n else list(range(total))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok, frame = cap.read()
        if ok:
            frames.append(preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    cap.release()
    if not frames: raise RuntimeError(f"No frame extracted from video: {path}")
    return torch.stack(frames)

def load_media_tensor(path: Path, preprocess, n_frames: int, seed: int) -> torch.Tensor:
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return sample_video_frames(path, preprocess, n_frames, seed)
    return preprocess(Image.open(path).convert("RGB")).unsqueeze(0)

class ProjCLIPWithAdapter(nn.Module):
    def __init__(self, clip_model, feature_dim: int = 512, dropout: float = 0.1, ratio: float = 0.2):
        super().__init__(); self.clip_model = clip_model; self.ratio = ratio
        self.image_projection = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(feature_dim, feature_dim)).float()
        self.text_projection = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(feature_dim, feature_dim)).float()
        hidden = feature_dim // 4
        self.img_adapter = nn.Sequential(nn.Linear(feature_dim, hidden, bias=False), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, feature_dim, bias=False)).float()
        self.txt_adapter = nn.Sequential(nn.Linear(feature_dim, hidden, bias=False), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, feature_dim, bias=False)).float()
    def train(self, mode: bool = True):
        super().train(mode); self.clip_model.eval(); self.image_projection.eval(); self.text_projection.eval(); return self
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(): raw = self.clip_model.encode_image(x).float(); base = self.image_projection(raw)
        return F.normalize(base + self.ratio * self.img_adapter(base), p=2, dim=-1).mean(dim=0, keepdim=True)
    def encode_text(self, toks: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(): raw = self.clip_model.encode_text(toks).float(); base = self.text_projection(raw)
        return F.normalize(base + self.ratio * self.txt_adapter(base), p=2, dim=-1)

def build_model_from_checkpoint(model_path: Path, clip_model, device: torch.device, dropout: float, ratio: float):
    ckpt = torch.load(model_path, map_location="cpu")
    feature_dim = int(ckpt.get("feature_dim", 512))
    model = ProjCLIPWithAdapter(clip_model, feature_dim, dropout, ratio).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded checkpoint: {model_path}"); print(f"  feature_dim={feature_dim}, missing={len(missing)}, unexpected={len(unexpected)}")
    for p in model.parameters(): p.requires_grad = False
    for p in model.img_adapter.parameters(): p.requires_grad = True
    for p in model.txt_adapter.parameters(): p.requires_grad = True
    return model

@torch.no_grad()
def build_text_anchors(model, prompts: Dict[str, List[str]], classes: Sequence[str], device: torch.device):
    model.eval(); anchors = []
    for label in classes:
        anchors.append(model.encode_text(clip.tokenize(prompts[label]).to(device)))
    return anchors

def compute_logits(img_feat, anchors, temp: float):
    return torch.stack([(img_feat @ a.T).mean() for a in anchors]).unsqueeze(0) / temp

def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, dataset, preprocess, prompts, classes, optimizer, scheduler, scaler, device, args, epoch: int) -> float:
    model.train(); order = list(range(len(dataset))); random.Random(args.seed + epoch).shuffle(order)
    total_loss, ok_count = 0.0, 0
    for idx in order:
        s = dataset[idx]
        try:
            media = load_media_tensor(s.path, preprocess, args.n_frames, args.seed + epoch).to(device)
            target = torch.tensor([s.label_idx], dtype=torch.long, device=device)
            with autocast(enabled=args.amp):
                anchors = build_text_anchors(model, prompts, classes, device)
                logits = compute_logits(model.encode_image(media), anchors, args.temp)
                loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total_loss += float(loss.item()); ok_count += 1
        except Exception as e:
            print(f"[train skip] {s.path}: {e}")
    return total_loss / max(1, ok_count)

@torch.no_grad()
def evaluate(model, dataset, preprocess, prompts, classes, device, args, split_name: str):
    model.eval(); anchors = build_text_anchors(model, prompts, classes, device)
    y_true, y_pred, records = [], [], []
    for s in dataset.samples:
        try:
            media = load_media_tensor(s.path, preprocess, args.n_frames, args.seed).to(device)
            logits = compute_logits(model.encode_image(media), anchors, args.temp)
            probs = torch.softmax(logits, dim=-1).squeeze(0); pred = int(torch.argmax(probs).item())
            y_true.append(s.label_idx); y_pred.append(pred)
            row = {"split": split_name, "path": str(s.path), "filename": s.path.name, "true_label": s.label, "pred_label": classes[pred], "correct": bool(pred == s.label_idx)}
            for i, cls in enumerate(classes):
                row[f"score_{cls}"] = round(float(logits[0, i].item()), 6); row[f"prob_{cls}"] = round(float(probs[i].item()), 6)
            records.append(row)
        except Exception as e:
            print(f"[eval skip] {s.path}: {e}")
    if not y_true: return {"accuracy": 0.0, "n": 0}, records
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)), "n": len(y_true), "classes": list(classes),
        "classification_report": classification_report(y_true, y_pred, labels=list(range(len(classes))), target_names=list(classes), zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist(),
    }, records

def save_csv(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records: return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)

def parse_classes_and_prompts(args):
    preset = PRESETS[args.preset]
    classes, prompts = list(preset["classes"]), dict(preset["prompts"])
    if args.class_names:
        classes = [x.strip() for x in args.class_names.split(",") if x.strip()]
    if args.prompts_json:
        with open(args.prompts_json, "r", encoding="utf-8") as f: prompts = json.load(f)
    for cls in classes:
        if cls not in prompts or not prompts[cls]: raise ValueError(f"Missing prompts for class '{cls}'")
    return classes, prompts

def main():
    ap = argparse.ArgumentParser(description="Folder-based residual-adapter fine-tuning for trained ORACLE checkpoints.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--preset", default="fight_nofight", choices=sorted(PRESETS.keys()))
    ap.add_argument("--class-names", default=None, help="Comma-separated class names; default uses preset classes.")
    ap.add_argument("--prompts-json", default=None, help="Optional JSON mapping class names to prompt lists.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ratio", type=float, default=0.2)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--n-frames", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    args = ap.parse_args(); set_seed(args.seed)
    classes, prompts = parse_classes_and_prompts(args)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nPreset: {args.preset}\nClasses: {classes}")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval(); [p.requires_grad_(False) for p in clip_model.parameters()]
    model = build_model_from_checkpoint(Path(args.model_path), clip_model, device, args.dropout, args.ratio)
    train_ds = FolderBehaviourDataset(Path(args.dataset_root), "train", classes)
    test_ds = FolderBehaviourDataset(Path(args.dataset_root), "test", classes)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, int(args.epochs * len(train_ds) * 0.1), max(1, args.epochs * len(train_ds)))
    scaler = GradScaler(enabled=args.amp)
    best_acc, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_ds, preprocess, prompts, classes, optimizer, scheduler, scaler, device, args, epoch)
        metrics, _ = evaluate(model, test_ds, preprocess, prompts, classes, device, args, "test")
        acc = metrics["accuracy"]; history.append({"epoch": epoch, "train_loss": loss, "test_accuracy": acc})
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={loss:.4f} | test_acc={acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    train_metrics, train_records = evaluate(model, train_ds, preprocess, prompts, classes, device, args, "train")
    test_metrics, test_records = evaluate(model, test_ds, preprocess, prompts, classes, device, args, "test")
    save_csv(train_records, out / "train_predictions.csv"); save_csv(test_records, out / "test_predictions.csv")
    save_json({"config": vars(args), "classes": classes, "prompts": prompts, "history": history, "best_test_accuracy": best_acc, "final_train_metrics": train_metrics, "final_test_metrics": test_metrics}, out / "fine_tuning_results.json")
    torch.save({"model_state_dict": model.state_dict(), "classes": classes, "prompts": prompts, "base_checkpoint": str(args.model_path), "config": vars(args), "final_test_metrics": test_metrics}, out / "best_adapter_model.pth")
    print("=" * 60); print(f"Final train accuracy: {train_metrics['accuracy']*100:.2f}% (n={train_metrics['n']})"); print(f"Final test accuracy:  {test_metrics['accuracy']*100:.2f}% (n={test_metrics['n']})"); print(f"Saved to: {out}")

if __name__ == "__main__":
    main()
