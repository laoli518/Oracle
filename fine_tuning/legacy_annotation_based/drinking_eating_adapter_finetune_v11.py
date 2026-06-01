#!/usr/bin/env python3
"""
pos_softmax \u6df1\u5ea6\u51bb\u7ed3+\u6b8b\u5dee Adapter \u5fae\u8c03 [\u996e\u6c34/\u91c7\u98df\u4efb\u52a1 - v11 \u4fee\u590d\u7248 + \u9012\u51cf\u9009\u6837]

\u4fee\u590d\u8bf4\u660e（\u76f8\u8f83\u4e8e v10）：
  [Fix 1] "Other" \u7c7b\u6539\u7528\u6b63\u5411\u6587\u672c\u63cf\u8ff0，\u66ff\u4ee3\u539f\u5148\u4e0d\u7a33\u5b9a\u7684\u5426\u5b9a\u53e5\u539f\u578b。
           CLIP \u6587\u672c\u7f16\u7801\u5668\u5bf9\u5426\u5b9a\u8bed\u4e49\u7406\u89e3\u504f\u5f31，\u6b63\u5411\u63cf\u8ff0\u80fd\u63d0\u4f9b\u66f4\u7a33\u5b9a\u7684\u8bed\u4e49\u951a\u70b9。
  [Fix 2] \u4e09\u7c7b\u5206\u6570\u7edf\u4e00\u4e3a mean(feat @ proto.T) \u5f62\u5f0f，\u6d88\u9664 Other \u4e0e\u5176\u4ed6\u4e24\u7c7b
           \u95f4\u7684\u5c3a\u5ea6\u4e0d\u5bf9\u79f0\u95ee\u9898（\u539f\u5148 Other \u7528\u5355\u5411\u91cf\u70b9\u79ef，D/E \u7528\u591a\u5411\u91cf\u5747\u503c）。
  [Fix 3] \u79fb\u9664\u5197\u4f59\u7684 other_w \u53c2\u6570，\u56e0\u4e3a\u5c3a\u5ea6\u5df2\u5bf9\u9f50，\u8be5\u53c2\u6570\u4e0d\u518d\u5fc5\u8981。
  [Fix 4] evaluate() \u4e0e run_one() \u4e2d\u7684 Other \u8ba1\u7b97\u903b\u8f91\u540c\u6b65\u4fee\u590d，\u4fdd\u6301\u8bad\u7ec3/
           \u63a8\u7406\u4e00\u81f4\u6027。
  [Fix 5] \u8d85\u53c2\u7f51\u683c\u6062\u590d\u5bf9 optimizer \u7684\u9009\u62e9\u641c\u7d22，\u9002\u5f53\u6269\u5927 lr/ratio \u7a7a\u95f4。

\u65b0\u589e\u529f\u80fd：
  [New 1] build_train_items() \u589e\u52a0 save_path \u53c2\u6570，\u5c06 few-shot \u9009\u6837\u7ed3\u679c\u4fdd\u5b58\u4e3a CSV。
  [New 2] evaluate() \u589e\u52a0 save_path \u53c2\u6570，\u5c06\u6bcfimages\u56fe\u7684\u63a8\u7406\u7ed3\u679c（\u9884\u6d4b\u6807\u7b7e、\u5404\u7c7b\u5f97\u5206、
           \u771f\u5b9e\u6807\u7b7e、\u5404\u6807\u6ce8\u8005\u6807\u7b7e）\u4fdd\u5b58\u4e3a CSV。

\u9012\u51cf\u9009\u6837（Decrement Shot Selection）：
  [Decr 1] \u7b2c\u4e00\u8f6e（shots=10）\u6b63\u5e38\u9009\u6837，\u751f\u6210\u521d\u59cb\u6837\u672c\u96c6。
  [Decr 2] \u540e\u7eed\u6bcf\u8f6e（shots=9,8,...,5），\u5728\u4e0a\u4e00\u8f6e\u5168\u90e8\u6837\u672c\u57fa\u7840\u4e0a，KCenter \u7b97\u6cd5
           \u79fb\u9664\u6700\u8fd1\u7684\u70b9，\u4fdd\u7559\u6700\u8fdc\u7684 k-1 \u4e2a\u70b9，\u5b9e\u73b0\u9012\u51cf。
  [Decr 3] build_train_items() \u65b0\u589e prev_items \u53c2\u6570，\u6309\u7c7b\u522b\u6574\u7406\u4e0a\u4e00\u8f6e\u5df2\u9009\u8def\u5f84
           \u5e76\u900f\u4f20\u7ed9 kcenter_select()。\u751f\u6210\u7684 CSV \u65b0\u589e is_removed \u5217\u6807\u8bb0\u88ab\u79fb\u9664\u7684\u6837\u672c。
  [Decr 4] run_shots() \u65b0\u589e prev_train_items \u53c2\u6570\u5e76\u900f\u4f20\u7ed9 build_train_items()。
  [Decr 5] main() \u4e2d shots \u5faa\u73af\u4ece\u9ad8\u5230\u4f4e\u9012\u51cf\u4f20\u9012 prev_train_items，\u5b9e\u73b0\u8de8\u8f6e\u9012\u51cf\u79ef\u7d2f：
           shots=10 \u6b63\u5e38\u9009\u6837；shots=9 \u5728 10 \u7684\u57fa\u7840\u4e0a\u79fb\u9664 1 \u4e2a\u6837\u672c；\u4ee5\u6b64\u7c7b\u63a8。
"""

import os, argparse, random, itertools, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import clip
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from torch.cuda.amp import GradScaler, autocast

# ─────────────────────────────────────────────────────────────────────────────
# ★ \u5168\u5c40\u786e\u5b9a\u6027\u79cd\u5b50
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────────────────────────────────────────
# \u6807\u7b7e\u4e0e\u63cf\u8ff0
# ─────────────────────────────────────────────────────────────────────────────
LABEL_DESCRIPTIONS = {
    "Drinking": {
        "positive": [
            "a pig drinking water from nipple dispenser",
            "a pig using a nipple drinker",
            "a pig at the waterer taking a drink",
            "a pig lapping water from water source",
            "a pig accessing water for hydration",
            "a pig with mouth on water nipple",
        ],
    },
    "Eating": {
        "positive": [
            "a pig eating food from a trough",
            "a pig feeding at the feeder",
            "a pig with snout in the trough consuming feed",
            "a pig foraging and eating at feeding area",
            "a pig actively consuming feed",
            "a pig with head down eating from feeder",
        ],
    },
    "Other": {
        "positive": [
            "a pig not drinking water",
            "a pig showing no drinking behavior",
            "a pig not using the nipple drinker",
            "a pig not eating food",
            "a pig not consuming any feed",
            "a pig without access to food",
        ],
    },
}

ALL_LABELS       = ["Drinking", "Eating", "Other"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
ANNOTATORS       = ["LWZ \u6807\u6ce8", "WD \u6807\u6ce8", "ZPY \u6807\u6ce8", "ZZL \u6807\u6ce8"]
LABEL_IDX        = {"Drinking": 0, "Eating": 1, "Other": 2}

# ─────────────────────────────────────────────────────────────────────────────
# \u6570\u636e\u52a0\u8f7d\u903b\u8f91
# ─────────────────────────────────────────────────────────────────────────────
def load_annotations(path):
    df = pd.read_excel(path, sheet_name="\u6807\u6ce8\u6c47\u603b")
    df["\u5171\u8bc6\u6807\u7b7e"] = df["LWZ \u6807\u6ce8"]
    consistent_fnames = set(df[df["\u4e00\u81f4\u6027"] == "\u4e00\u81f4"]["\u6587\u4ef6\u540d"].tolist())
    return df.reset_index(drop=True), consistent_fnames

def build_path_index(data_root):
    idx = {}
    for root, _, files in os.walk(data_root):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                idx[f] = os.path.join(root, f)
    return idx

def load_img(path, preprocess_fn, device):
    return preprocess_fn(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

# ─────────────────────────────────────────────────────────────────────────────
# \u6a21\u578b\u67b6\u6784 (Residual Adapter)
# ─────────────────────────────────────────────────────────────────────────────
class ProjCLIPWithAdapter(nn.Module):
    def __init__(self, clip_model, dropout=0.1, ratio=0.2):
        super().__init__()
        self.clip_model = clip_model
        self.ratio = ratio
        d = clip_model.visual.output_dim

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
            raw  = self.clip_model.encode_image(x).float()
            base = self.image_projection(raw)
        return self._norm(base + self.ratio * self.img_adapter(base))

    def encode_text(self, toks):
        with torch.no_grad():
            raw  = self.clip_model.encode_text(toks).float()
            base = self.text_projection(raw)
        return self._norm(base + self.ratio * self.txt_adapter(base))

def fresh_adapter_model(clip_model, base_state, device, dropout, ratio):
    m = ProjCLIPWithAdapter(clip_model, dropout=dropout, ratio=ratio).to(device)
    m.load_state_dict(base_state, strict=False)
    return m

# ─────────────────────────────────────────────────────────────────────────────
# I-JEPA \u8c03\u5ea6\u5668：\u6309\u8fed\u4ee3\u6b65\u66f4\u65b0\u7684\u4f59\u5f26\u9000\u706b
# ─────────────────────────────────────────────────────────────────────────────
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def build_optimizer(name: str, params, lr, weight_decay=1e-2):
    name = name.lower()
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    else:
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)

# ─────────────────────────────────────────────────────────────────────────────
# ★ Fix 2：\u7edf\u4e00\u6587\u672c\u539f\u578b\u8ba1\u7b97 — \u4e09\u7c7b\u5747\u91c7\u7528\u591a\u5411\u91cf\u77e9\u9635，\u5c3a\u5ea6\u5b8c\u5168\u5bf9\u9f50
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_text_anchors(model, device):
    """
    \u8fd4\u56de\u4e09\u4e2a\u6587\u672c\u539f\u578b\u77e9\u9635，\u5f62\u72b6\u5747\u4e3a [n_prompts, d]。
    \u63a8\u7406\u65f6\u7edf\u4e00\u7528 mean(feat @ proto.T) \u8ba1\u7b97\u76f8\u4f3c\u5ea6，\u4fdd\u8bc1\u5c3a\u5ea6\u4e00\u81f4。
    """
    d_pos = model.encode_text(
        clip.tokenize(LABEL_DESCRIPTIONS["Drinking"]["positive"]).to(device)
    )
    e_pos = model.encode_text(
        clip.tokenize(LABEL_DESCRIPTIONS["Eating"]["positive"]).to(device)
    )
    # ★ Other \u4f7f\u7528\u6b63\u5411\u63cf\u8ff0\u800c\u975e\u5426\u5b9a\u53e5
    o_pos = model.encode_text(
        clip.tokenize(LABEL_DESCRIPTIONS["Other"]["positive"]).to(device)
    )
    return d_pos, e_pos, o_pos

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation\u903b\u8f91
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, ann_df, consistent_fnames, path_index,
             preprocess, device, temp=1.0, exclude_paths=None,
             save_path=None):  # [New 2] \u65b0\u589e save_path \u53c2\u6570
    """
    save_path: \u82e5\u4e0d\u4e3a None，\u5219\u5c06\u6bcfimages\u56fe\u7684\u63a8\u7406\u660e\u7ec6\u4fdd\u5b58\u4e3a CSV。
               CSV \u5217：\u6587\u4ef6\u540d, \u8def\u5f84, \u9884\u6d4b\u6807\u7b7e, score_Drinking, score_Eating,
                       score_Other, \u5171\u8bc6\u6807\u7b7e, \u662f\u5426\u4e00\u81f4, LWZ\u6807\u6ce8, WD\u6807\u6ce8,
                       ZPY\u6807\u6ce8, ZZL\u6807\u6ce8
    """
    model.eval()
    exclude   = set(exclude_paths) if exclude_paths else set()
    # ★ Fix 4：\u63a8\u7406\u4e0e\u8bad\u7ec3\u4f7f\u7528\u540c\u4e00\u5957 anchor \u8ba1\u7b97\u51fd\u6570
    d_pos, e_pos, o_pos = get_text_anchors(model, device)

    acc_preds, acc_gt, kappa_preds = [], [], []
    kappa_gt = {a: [] for a in ANNOTATORS}

    # [New 2] \u7528\u4e8e\u6536\u96c6\u63a8\u7406\u660e\u7ec6\u7684\u5217\u8868
    inference_records = []

    for _, row in ann_df.iterrows():
        fname = row["\u6587\u4ef6\u540d"]
        p     = path_index.get(fname)
        if p is None or p in exclude:
            continue
        try:
            feat = model.encode_image(load_img(p, preprocess, device))
            # ★ Fix 2：\u4e09\u7c7b\u76f8\u4f3c\u5ea6\u8ba1\u7b97\u65b9\u5f0f\u5b8c\u5168\u4e00\u81f4
            s_D  = (feat @ d_pos.T).mean().item()
            s_E  = (feat @ e_pos.T).mean().item()
            s_O  = (feat @ o_pos.T).mean().item()
            scores = {
                "Drinking": s_D / temp,
                "Eating":   s_E / temp,
                "Other":    s_O / temp,
            }
            pred = max(scores, key=scores.get)

            kappa_preds.append(pred)
            for a in ANNOTATORS:
                kappa_gt[a].append(row[a])
            if fname in consistent_fnames:
                acc_preds.append(pred)
                acc_gt.append(row["\u5171\u8bc6\u6807\u7b7e"])

            # [New 2] \u8bb0\u5f55\u672c\u6761\u63a8\u7406\u7ed3\u679c
            if save_path is not None:
                inference_records.append({
                    "\u6587\u4ef6\u540d":         fname,
                    "\u8def\u5f84":           p,
                    "\u9884\u6d4b\u6807\u7b7e":       pred,
                    "score_Drinking": round(s_D / temp, 6),
                    "score_Eating":   round(s_E / temp, 6),
                    "score_Other":    round(s_O / temp, 6),
                    "\u5171\u8bc6\u6807\u7b7e":       row.get("\u5171\u8bc6\u6807\u7b7e", ""),
                    "\u662f\u5426\u4e00\u81f4":       "\u662f" if fname in consistent_fnames else "\u5426",
                    "LWZ \u6807\u6ce8":       row.get("LWZ \u6807\u6ce8", ""),
                    "WD \u6807\u6ce8":        row.get("WD \u6807\u6ce8", ""),
                    "ZPY \u6807\u6ce8":       row.get("ZPY \u6807\u6ce8", ""),
                    "ZZL \u6807\u6ce8":       row.get("ZZL \u6807\u6ce8", ""),
                })

        except Exception:
            continue

    acc = (
        sum(p == g for p, g in zip(acc_preds, acc_gt)) / len(acc_preds)
        if acc_preds else 0.0
    )
    if kappa_preds:
        kappas = {
            a: cohen_kappa_score(kappa_gt[a], kappa_preds, labels=ALL_LABELS)
            for a in ANNOTATORS
        }
        avg_k = sum(kappas.values()) / len(kappas)
    else:
        kappas, avg_k = {}, 0.0

    # [New 2] \u4fdd\u5b58\u63a8\u7406\u660e\u7ec6 CSV
    if save_path is not None and inference_records:
        inf_df = pd.DataFrame(inference_records)
        inf_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"  → \u63a8\u7406\u7ed3\u679c\u5df2\u4fdd\u5b58: {save_path} ({len(inference_records)} \u6761)")

    return avg_k, acc, kappas, len(exclude), len(acc_preds), len(kappa_preds)

# ─────────────────────────────────────────────────────────────────────────────
# \u8bad\u7ec3\u5f15\u64ce (I-JEPA \u7b56\u7565 + \u4fee\u590d\u540e\u7684 Other \u8ba1\u7b97)
# ─────────────────────────────────────────────────────────────────────────────
def run_one(cfg, clip_model, base_state, train_items,
            ann_df, consistent_fnames, path_index, preprocess, device,
            inference_save_path=None):  # [New 2] \u900f\u4f20\u63a8\u7406\u4fdd\u5b58\u8def\u5f84
    set_seed(cfg["seed"])
    model = fresh_adapter_model(clip_model, base_state, device, cfg["dropout"], cfg["ratio"])

    # \u9501\u5b9a Backbone，\u53ea\u5f00 Adapter \u68af\u5ea6
    for p in model.parameters():        p.requires_grad = False
    for p in model.img_adapter.parameters(): p.requires_grad = True
    for p in model.txt_adapter.parameters(): p.requires_grad = True
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    opt          = build_optimizer(cfg["optimizer"], trainable_params, cfg["lr"], weight_decay=1e-2)
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
                img_input = load_img(path, preprocess, device)
                target    = torch.tensor([LABEL_IDX[lbl]], device=device)

                with autocast():
                    # ★ Fix 2 & Fix 4：\u8bad\u7ec3\u65f6\u540c\u6837\u4f7f\u7528\u7edf\u4e00\u7684 anchor \u51fd\u6570
                    d_pos, e_pos, o_pos = get_text_anchors(model, device)
                    feat   = model.encode_image(img_input)
                    s_D    = (feat @ d_pos.T).mean()
                    s_E    = (feat @ e_pos.T).mean()
                    # ★ Fix 2：Other \u4e5f\u7528\u591a\u5411\u91cf\u5747\u503c，\u4e0d\u518d\u7528\u5355\u5411\u91cf\u70b9\u79ef
                    s_O    = (feat @ o_pos.T).mean()
                    logits = torch.stack([s_D, s_E, s_O]).unsqueeze(0) / cfg["temp"]
                    loss   = F.cross_entropy(logits, target)

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                scheduler.step()

                ep_loss += loss.item()
            except Exception:
                continue

        avg = ep_loss / max(len(train_items), 1)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    train_paths = [p for p, _ in train_items]
    avg_k, acc, kappas, n_skip, n_acc, n_kappa = evaluate(
        model, ann_df, consistent_fnames, path_index,
        preprocess, device, cfg["temp"], exclude_paths=train_paths,
        save_path=inference_save_path,  # [New 2] \u4f20\u5165\u4fdd\u5b58\u8def\u5f84
    )
    # \u4e0e Lying \u7248\u4fdd\u6301\u4e00\u81f4：\u8fd4\u56de 8 \u4e2a\u503c
    return avg_k, acc, kappas, best_loss, best_state, n_skip, n_acc, n_kappa

# ─────────────────────────────────────────────────────────────────────────────
# Few-shot \u9009\u62e9\u903b\u8f91 (K-Center + \u9012\u51cf\u9009\u6837)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def kcenter_select(paths, model, preprocess, device, k, seed=42, forced_paths=None):
    """
    paths        : \u5168\u90e8\u5019\u9009\u8def\u5f84\u5217\u8868。
    k            : \u76ee\u6807\u9009\u6837\u6570\u91cf。
    forced_paths : \u4e0a\u4e00\u8f6e\u5df2\u9009\u8def\u5f84\u5217\u8868。
                   \u9012\u51cf\u6a21\u5f0f：\u8fd9\u4e9b\u8def\u5f84\u662f\u4e0a\u8f6e\u7684\u5168\u90e8\u6837\u672c，\u76ee\u6807\u662f\u5728\u5176\u4e2d\u4fdd\u7559\u6700\u8fdc\u7684 k \u4e2a。
                   [Decr 1-2]
    """
    paths = sorted(paths, key=os.path.basename)
    if len(paths) <= k:
        return paths
    feats, valid = [], []
    for p in paths:
        try:
            f = model.encode_image(load_img(p, preprocess, device)).squeeze(0).cpu()
            feats.append(f)
            valid.append(p)
        except Exception:
            continue
    if len(valid) <= k:
        return valid
    feats = torch.stack(feats)
    dist  = (1 - (feats @ feats.T)).clamp(min=0).numpy()

    forced_set = set(forced_paths) if forced_paths else set()
    rng        = random.Random(seed)

    # ★ \u9012\u51cf\u6a21\u5f0f：forced_paths \u662f\u4e0a\u4e00\u8f6e\u7684\u5168\u90e8\u6837\u672c
    # \u4ece\u4e2d\u9009\u51fa\u6700\u8fdc\u7684 k \u4e2a（\u800c\u975e\u9009\u51fa\u65b0\u7684 k \u4e2a）
    if forced_set:
        # \u627e\u51fa\u4e0a\u4e00\u8f6e\u6837\u672c\u4e2d\u54ea\u4e9b\u5728 valid \u4e2d
        prev_indices = [i for i, p in enumerate(valid) if p in forced_set]
        
        if len(prev_indices) > k:
            # \u9700\u8981\u4ece prev_indices \u4e2d\u5220\u9664\u6700\u8fd1\u7684\u90a3\u4e2a
            # \u8ba1\u7b97 prev_indices \u4e4b\u95f4\u7684\u8ddd\u79bb\u77e9\u9635
            prev_dist = dist[np.ix_(prev_indices, prev_indices)]
            
            # \u4f7f\u7528\u53cd\u5411\u7684 k-center：\u8d2a\u5fc3\u5220\u9664\u6700\u8fd1\u7684\u70b9
            sel_idx = list(range(len(prev_indices)))  # \u521d\u59cb\u9009\u4e2d\u5168\u90e8
            
            while len(sel_idx) > k:
                # \u627e\u51fa\u5f53\u524d\u9009\u4e2d\u6837\u672c\u4e2d\u5230"\u5176\u4ed6\u9009\u4e2d\u6837\u672c"\u7684\u6700\u5c0f\u5e73\u5747\u8ddd\u79bb\u6700\u5c0f\u7684\u90a3\u4e2a
                min_avg_dist = np.inf
                remove_idx = -1
                
                for i, si in enumerate(sel_idx):
                    other_indices = [sel_idx[j] for j in range(len(sel_idx)) if j != i]
                    if other_indices:
                        avg_dist = prev_dist[si, other_indices].mean()
                        if avg_dist < min_avg_dist:
                            min_avg_dist = avg_dist
                            remove_idx = i
                
                sel_idx.pop(remove_idx)
            
            selected_prev_indices = [prev_indices[i] for i in sel_idx]
            return [valid[i] for i in selected_prev_indices]
        else:
            # \u4e0a\u8f6e\u6837\u672c\u4e0d\u8d85\u8fc7 k items, \u5168\u90e8\u4fdd\u7559
            return [valid[i] for i in prev_indices]
    
    # \u7b2c\u4e00\u8f6e（\u65e0 forced_paths）：\u6b63\u5e38 K-Center \u9009\u6837
    sel = [rng.randint(0, len(valid) - 1)]
    min_d = np.full(len(valid), np.inf)
    for s in sel:
        min_d = np.minimum(min_d, dist[s])
    
    while len(sel) < k:
        for s in sel:
            min_d[s] = -1
        nxt = int(np.argmax(min_d))
        sel.append(nxt)
        min_d = np.minimum(min_d, dist[nxt])
    
    return [valid[i] for i in sel]


def build_train_items(ann_df, consistent_fnames, path_index,
                      model, preprocess, device, n_shots=5, seed=42,
                      save_path=None, prev_items=None):
    """
    prev_items : \u4e0a\u4e00\u8f6e train_items（list of (path, label)）。
                 \u5728\u9012\u51cf\u6a21\u5f0f\u4e0b，\u8fd9\u662f\u4e0a\u4e00\u8f6e\u7684\u5168\u90e8\u6837\u672c，\u672c\u8f6e\u8981\u5728\u5176\u4e2d\u4fdd\u7559\u6700\u8fdc\u7684 n_shots \u4e2a。
                 \u751f\u6210\u7684 CSV \u542b is_removed \u5217\u6807\u8bb0\u88ab\u79fb\u9664\u7684\u6837\u672c。
                 [Decr 3]

    save_path  : \u82e5\u4e0d\u4e3a None，\u5219\u5c06 few-shot \u9009\u6837\u7ed3\u679c\u4fdd\u5b58\u4e3a CSV。
                 CSV \u5217：label, filename, path, is_removed
    """
    items   = []
    cons_df = ann_df[ann_df["\u6587\u4ef6\u540d"].isin(consistent_fnames)]
    fewshot_records = []

    # \u6309\u7c7b\u522b\u6574\u7406\u4e0a\u4e00\u8f6e\u5df2\u9009\u8def\u5f84  [Decr 3]
    prev_by_label = {lbl: [] for lbl in ALL_LABELS}
    if prev_items:
        for p, lbl in prev_items:
            prev_by_label[lbl].append(p)

    for lbl in ALL_LABELS:
        cands = sorted(
            [
                path_index[r["\u6587\u4ef6\u540d"]]
                for _, r in cons_df[cons_df["\u5171\u8bc6\u6807\u7b7e"] == lbl].iterrows()
                if r["\u6587\u4ef6\u540d"] in path_index
            ],
            key=os.path.basename,
        )
        forced = prev_by_label[lbl] if prev_items else None
        chosen = kcenter_select(
            cands, model, preprocess, device, n_shots, seed,
            forced_paths=forced,   # [Decr 3] \u9012\u51cf\u6a21\u5f0f\u4e0b\u900f\u4f20\u4e0a\u8f6e\u5168\u90e8\u6837\u672c
        )
        items.extend((p, lbl) for p in chosen)

        prev_set = set(prev_by_label[lbl])
        for p in chosen:
            fewshot_records.append({
                "label":    lbl,
                "filename": os.path.basename(p),
                "path":     p,
                "is_removed": "\u5426" if p in prev_set else "\u662f",  # \u6807\u8bb0\u88ab\u79fb\u9664\u7684\u6837\u672c
            })

    # [New 1] \u4fdd\u5b58 few-shot \u9009\u6837 CSV
    if save_path is not None and fewshot_records:
        fs_df = pd.DataFrame(fewshot_records)
        fs_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"Few-shot \u9009\u6837\u5df2\u4fdd\u5b58: {save_path} ({len(fewshot_records)} \u6761)")

    return items

# ─────────────────────────────────────────────────────────────────────────────
# \u5355\u6b21 shots \u5b9e\u9a8c（\u62bd\u51fa\u4e3a\u72ec\u7acb\u51fd\u6570，\u4fbf\u4e8e\u5faa\u73af\u8c03\u7528）
# ─────────────────────────────────────────────────────────────────────────────
def run_shots(shots, grid, clip_model, base_state, ann_df, consistent_fnames,
              path_index, eval_preprocess, device, output_dir, seed, base_m,
              prev_train_items=None):   # [Decr 4] \u65b0\u589e prev_train_items \u53c2\u6570
    """
    \u9488\u5bf9\u7ed9\u5b9a shots \u503c\u8dd1\u5b8c\u6574\u8d85\u53c2\u7f51\u683c，\u7ed3\u679c\u4fdd\u5b58\u5230 output_dir/shots_{shots}/ \u5b50\u76ee\u5f55。
    \u8fd4\u56de\u8be5 shots \u4e0b\u7684\u6700\u4f73 (avg_kappa, cfg, ft_state, train_items)。

    prev_train_items: \u4e0a\u4e00\u8f6e（shots+1）\u7684\u8bad\u7ec3\u6837\u672c\u5217\u8868，\u7528\u4e8e\u9012\u51cf\u9009\u6837。[Decr 4]
    """
    shot_dir = os.path.join(output_dir, f"shots_{shots}")
    os.makedirs(shot_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  shots = {shots}   \u5b50\u76ee\u5f55: {shot_dir}")
    if prev_train_items is not None:
        print(f"  （\u5728\u4e0a\u4e00\u8f6e {len(prev_train_items)} \u4e2a\u6837\u672c\u57fa\u7840\u4e0a\u9012\u51cf\u9009\u6837）")
    print(f"{'='*70}")

    # ── Few-shot \u9009\u6837（\u9012\u51cf）──────────────────────────────────────────────
    fewshot_csv = os.path.join(shot_dir, "fewshot_samples.csv")
    train_items = build_train_items(
        ann_df, consistent_fnames, path_index,
        base_m, eval_preprocess, device,
        n_shots=shots, seed=seed,
        save_path=fewshot_csv,
        prev_items=prev_train_items,   # [Decr 4] \u900f\u4f20\u4e0a\u4e00\u8f6e\u9009\u6837
    )

    # ── \u8d85\u53c2\u7f51\u683c\u641c\u7d22 ───────────────────────────────────────────────────────
    results = []
    best_avg_k, best_cfg, best_ft_state = -1.0, None, None

    for i, (opt_name, lr, epochs, dropout, ratio, temp) in enumerate(grid, 1):
        cfg = dict(
            optimizer=opt_name, lr=lr, epochs=epochs,
            dropout=dropout, ratio=ratio, temp=temp, seed=seed,
        )
        t0 = time.time()
        print(
            f"  [{i:3d}/{len(grid)}] {opt_name} LR={lr:.0e} EP={epochs} "
            f"Ratio={ratio} Temp={temp}...",
            end=" ", flush=True,
        )

        inf_csv = os.path.join(
            shot_dir,
            f"inference_{i:03d}_{opt_name}_lr{lr:.0e}_ep{epochs}"
            f"_ratio{ratio}_temp{temp}.csv",
        )

        try:
            avg_k, acc, kappas, b_loss, ft_state, n_skip, n_acc, n_kappa = run_one(
                cfg, clip_model, base_state, train_items,
                ann_df, consistent_fnames, path_index, eval_preprocess, device,
                inference_save_path=inf_csv,
            )
            print(f"κ={avg_k:.4f} Acc={acc*100:.1f}% ({time.time()-t0:.1f}s)")

            results.append({
                "shots":         shots,
                "optimizer":     cfg["optimizer"],
                "lr":            cfg["lr"],
                "epochs":        cfg["epochs"],
                "dropout":       cfg["dropout"],
                "ratio":         cfg["ratio"],
                "temp":          cfg["temp"],
                "avg_kappa":     avg_k,
                "accuracy":      acc,
                "n_acc":         n_acc,
                "n_kappa":       n_kappa,
                "loss":          b_loss,
                "inference_csv": inf_csv,
            })

            if avg_k > best_avg_k:
                best_avg_k    = avg_k
                best_cfg      = cfg
                best_ft_state = ft_state

        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()

    if not results:
        print(f"  ⚠️  shots={shots} \u6240\u6709\u5b9e\u9a8c\u5747\u5931\u8d25，\u8df3\u8fc7。")
        return best_avg_k, best_cfg, best_ft_state, train_items

    # ── \u4fdd\u5b58\u672c shots \u6c47\u603b\u8868 ────────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values("avg_kappa", ascending=False)
    csv_path = os.path.join(shot_dir, "results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ── \u4fdd\u5b58\u672c shots \u6700\u4f73\u6a21\u578b ──────────────────────────────────────────────
    best_pth = os.path.join(shot_dir, "best_model.pth")
    torch.save({
        "model_state_dict": best_ft_state,
        "config":           best_cfg,
        "avg_kappa":        best_avg_k,
        "shots":            shots,
    }, best_pth)

    # ── \u4fdd\u5b58\u6700\u4f73\u6a21\u578b\u7684\u63a8\u7406\u7ed3\u679c ─────────────────────────────────────────────
    print(f"  → \u4fdd\u5b58 shots={shots} \u6700\u4f73\u6a21\u578b\u63a8\u7406\u7ed3\u679c...")
    best_model = fresh_adapter_model(
        clip_model, best_ft_state, device, best_cfg["dropout"], best_cfg["ratio"]
    )
    best_inf_csv = os.path.join(shot_dir, "inference_best_model.csv")
    train_paths  = [p for p, _ in train_items]
    evaluate(
        best_model, ann_df, consistent_fnames, path_index,
        eval_preprocess, device, best_cfg["temp"],
        exclude_paths=train_paths,
        save_path=best_inf_csv,
    )

    print(f"  ✅ shots={shots} \u5b8c\u6210。\u6700\u4f73 κ={best_avg_k:.4f}")
    print(f"     \u6c47\u603b: {csv_path}")
    print(f"     \u6a21\u578b: {best_pth}")
    print(f"     \u63a8\u7406: {best_inf_csv}")

    return best_avg_k, best_cfg, best_ft_state, train_items


# ─────────────────────────────────────────────────────────────────────────────
# \u4e3b\u7a0b\u5e8f
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--model",      required=True)
    parser.add_argument("--output_dir", default="drinking_results_v12")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--shots_min",  type=int, default=5,
                        help="shots \u5faa\u73af\u7ec8\u70b9（\u9012\u51cf\u5230\u6b64，\u542b）")
    parser.add_argument("--shots_max",  type=int, default=10,
                        help="shots \u5faa\u73af\u8d77\u70b9（\u4ece\u6b64\u5f00\u59cb\u9012\u51cf，\u542b）")
    parser.add_argument("--quick",      action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── \u516c\u5171\u6570\u636e\u4e0e\u6a21\u578b（\u53ea\u52a0\u8f7d\u4e00\u6b21）───────────────────────────────────────
    ann_df, consistent_fnames = load_annotations(args.annotation)
    path_index = build_path_index(args.data_root)
    ann_df = ann_df[ann_df["\u6587\u4ef6\u540d"].isin(path_index)].reset_index(drop=True)
    consistent_fnames &= set(path_index.keys())

    clip_model, eval_preprocess = clip.load("ViT-B/32", device=device)
    ckpt   = torch.load(args.model, map_location=device)
    base_m = ProjCLIPWithAdapter(clip_model).to(device)
    base_m.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    base_state = {k: v.clone() for k, v in base_m.state_dict().items()}

    # ── \u8d85\u53c2\u7f51\u683c（\u6240\u6709 shots \u5171\u7528）────────────────────────────────────────
    if args.quick:
        grid = [("adamw", 5e-3, 100, 0.1, 0.2, 2.0)]
    else:
        grid = list(itertools.product(
            ["adamw", "adam"],
            [1e-4, 3e-4, 5e-4],
            [50, 80],
            [0.1],
            [0.2, 0.4],
            [1.0, 2.0],
        ))

    # ★ \u4fee\u6539：\u4ece\u9ad8\u5230\u4f4e\u9012\u51cf shots（10 → 9 → 8 → ... → 5）
    shots_range = range(args.shots_max, args.shots_min - 1, -1)
    print(f"\n🚀 shots \u5faa\u73af（\u9012\u51cf\u6a21\u5f0f）: {list(shots_range)}，\u6bcf\u8f6e\u8d85\u53c2\u7ec4\u5408\u6570: {len(grid)}")
    print(f"   \u9012\u51cf\u9009\u6837\u6a21\u5f0f：\u7b2c\u4e00\u8f6e\u9009 {args.shots_max} items, \u540e\u7eed\u6bcf\u8f6e\u79fb\u9664 1 \u4e2a\u6700\u8fd1\u7684\u6837\u672c")

    # ── shots \u5faa\u73af（\u9012\u51cf\u4e32\u8054）────────────────────────────────────────────
    all_summary      = []          # \u8de8 shots \u6c47\u603b
    global_best_k    = -1.0
    global_best_cfg  = None
    global_best_state = None
    global_best_shots = None
    global_best_train = None

    # ★ \u6539\u4e3a\u9012\u51cf\u6a21\u5f0f：\u7ef4\u62a4\u5f53\u524d\u9009\u4e2d\u7684\u5168\u90e8\u6837\u672c，\u6bcf\u8f6e\u5220\u9664\u4e00\u4e2a
    all_selected_items = None      # \u521d\u59cb\u5316，\u7b2c\u4e00\u8f6e\u4e3a None（\u6b63\u5e38\u9009\u6837）

    for shots in shots_range:
        # ★ \u82e5\u4e0d\u662f\u9996\u8f6e（shots == shots_max），\u5219\u5728\u4e0a\u4e00\u8f6e\u6837\u672c\u57fa\u7840\u4e0a\u51cf\u6837
        prev_train_for_decrement = all_selected_items if shots < args.shots_max else None
        
        best_k, best_cfg, best_state_s, train_items = run_shots(
            shots, grid, clip_model, base_state,
            ann_df, consistent_fnames, path_index,
            eval_preprocess, device,
            args.output_dir, args.seed, base_m,
            prev_train_items=prev_train_for_decrement,   # ★ \u4f20\u5165\u4e0a\u8f6e\u5168\u90e8\u6837\u672c（\u9012\u51cf\u6a21\u5f0f）
        )

        # ★ \u4fdd\u5b58\u672c\u8f6e\u5168\u90e8\u9009\u4e2d\u6837\u672c，\u4f9b\u4e0b\u4e00\u8f6e\u9012\u51cf\u4f7f\u7528
        all_selected_items = train_items

        if best_cfg is not None:
            all_summary.append({
                "shots":     shots,
                "avg_kappa": best_k,
                **{k: v for k, v in best_cfg.items() if k != "seed"},
            })
        if best_k > global_best_k:
            global_best_k     = best_k
            global_best_cfg   = best_cfg
            global_best_state = best_state_s
            global_best_shots = shots
            global_best_train = train_items

    # ── \u8de8 shots \u6c47\u603b\u8868 ───────────────────────────────────────────────────
    if all_summary:
        summary_df = pd.DataFrame(all_summary).sort_values("avg_kappa", ascending=False)
        summary_csv = os.path.join(args.output_dir, "summary_all_shots.csv")
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print(f"\n📊 \u8de8 shots \u6c47\u603b\u5df2\u4fdd\u5b58: {summary_csv}")
        print(summary_df.to_string(index=False))

    # ── \u5168\u5c40\u6700\u4f73\u6a21\u578b ──────────────────────────────────────────────────────
    if global_best_state is not None:
        global_pth = os.path.join(args.output_dir, "best_model_global.pth")
        torch.save({
            "model_state_dict": global_best_state,
            "config":           global_best_cfg,
            "avg_kappa":        global_best_k,
            "shots":            global_best_shots,
        }, global_pth)

        print(f"\n🏆 \u5168\u5c40\u6700\u4f73: shots={global_best_shots}  κ={global_best_k:.4f}")
        print(f"   Model saved: {global_pth}")

        # \u5168\u5c40\u6700\u4f73\u63a8\u7406\u7ed3\u679c
        print("   \u4fdd\u5b58\u5168\u5c40\u6700\u4f73\u63a8\u7406\u7ed3\u679c...")
        global_best_model = fresh_adapter_model(
            clip_model, global_best_state, device,
            global_best_cfg["dropout"], global_best_cfg["ratio"]
        )
        global_inf_csv     = os.path.join(args.output_dir, "inference_global_best.csv")
        global_train_paths = [p for p, _ in global_best_train]
        evaluate(
            global_best_model, ann_df, consistent_fnames, path_index,
            eval_preprocess, device, global_best_cfg["temp"],
            exclude_paths=global_train_paths,
            save_path=global_inf_csv,
        )
        print(f"   \u63a8\u7406\u7ed3\u679c: {global_inf_csv}")

    print("\n✅ \u5168\u90e8\u5b8c\u6210。")


if __name__ == "__main__":
    main()