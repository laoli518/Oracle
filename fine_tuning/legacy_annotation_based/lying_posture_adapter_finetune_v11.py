#!/usr/bin/env python3
"""
pos_softmax \u6df1\u5ea6\u51bb\u7ed3+\u6b8b\u5dee Adapter \u5fae\u8c03 v11（\u9012\u51cf shots \u7248）
\u57fa\u4e8e v11 \u4fee\u6539，\u65b0\u589e：
  ★ \u9012\u51cf few-shot \u9009\u6837：\u5148\u9009 shots_max \u4e2a\u6837\u672c，
    \u4e4b\u540e\u6bcf\u8f6e\u5728\u4e0a\u4e00\u8f6e\u57fa\u7840\u4e0a\u6bcf\u7c7b\u5404\u79fb\u9664 1 \u4e2a"\u6700\u4e0d\u91cd\u8981"\u7684\u6837\u672c
    （\u5373\u4e0e\u5176\u4f59\u5df2\u9009\u6837\u672c\u4f59\u5f26\u8ddd\u79bb\u6700\u5c0f\u7684\u70b9），\u4fdd\u8bc1\u6837\u672c\u96c6\u5408\u5355\u8c03\u7f29\u5c0f。
"""

import os, argparse, random, itertools, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import clip
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
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
    "Lateral Lying": {
        "positive": [
            "a pig lying completely on its side",
            "a pig in lateral recumbent position",
            "a pig lying sideways with legs extended",
            "a pig resting on side",
            "a pig lying with full body contact on side",
            "a pig in side-lying position",
        ]
    },
    "Sternal Lying": {
        "positive": [
            "a pig lying on chest with legs tucked",
            "a pig in sternal recumbent position",
            "a pig resting on sternum like sphinx",
            "a pig lying on belly with head up",
            "a pig in chest-down position",
            "a pig prone on sternum",
        ]
    },
    "Not Lying": {
        "positive": [
            "a pig not lying down",
            "a pig in upright position not resting",
            "a pig standing or sitting but not lying",
            "a pig vertical not horizontal",
            "a pig upright not recumbent",
            "a pig active not lying down",
        ]
    }
}

ALL_LABELS       = ["Lateral Lying", "Sternal Lying", "Not Lying"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
ANNOTATORS       = ["LWZ \u6807\u6ce8", "WD \u6807\u6ce8", "ZPY \u6807\u6ce8", "ZZL \u6807\u6ce8"]
LABEL_IDX        = {"Lateral Lying": 0, "Sternal Lying": 1, "Not Lying": 2}

# ─────────────────────────────────────────────────────────────────────────────
# \u6570\u636e\u52a0\u8f7d\u4e0e\u7d22\u5f15
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
            nn.Dropout(dropout), nn.Linear(hidden_d, d, bias=True)
        ).float()
        self.txt_adapter = nn.Sequential(
            nn.Linear(d, hidden_d, bias=False), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_d, d, bias=True)
        ).float()

        nn.init.zeros_(self.img_adapter[-1].weight)
        nn.init.zeros_(self.img_adapter[-1].bias)
        nn.init.zeros_(self.txt_adapter[-1].weight)
        nn.init.zeros_(self.txt_adapter[-1].bias)

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
        return self._norm(base_feat + self.ratio * self.img_adapter(base_feat))

    def encode_text(self, toks):
        with torch.no_grad():
            raw       = self.clip_model.encode_text(toks).float()
            base_feat = self.text_projection(raw)
        return self._norm(base_feat + self.ratio * self.txt_adapter(base_feat))

def fresh_adapter_model(clip_model, base_state, device, dropout, ratio):
    m = ProjCLIPWithAdapter(clip_model, dropout=dropout, ratio=ratio).to(device)
    m.load_state_dict(base_state, strict=False)
    nn.init.zeros_(m.img_adapter[-1].weight)
    nn.init.zeros_(m.img_adapter[-1].bias)
    nn.init.zeros_(m.txt_adapter[-1].weight)
    nn.init.zeros_(m.txt_adapter[-1].bias)
    return m

# ─────────────────────────────────────────────────────────────────────────────
# \u8c03\u5ea6\u5668 & \u4f18\u5316\u5668
# ─────────────────────────────────────────────────────────────────────────────
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
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
# \u6587\u672c\u539f\u578b\u8ba1\u7b97
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_text_anchors(model, device):
    l_pos = model.encode_text(clip.tokenize(LABEL_DESCRIPTIONS["Lateral Lying"]["positive"]).to(device))
    s_pos = model.encode_text(clip.tokenize(LABEL_DESCRIPTIONS["Sternal Lying"]["positive"]).to(device))
    n_pos = model.encode_text(clip.tokenize(LABEL_DESCRIPTIONS["Not Lying"]["positive"]).to(device))
    return l_pos, s_pos, n_pos

def softmax_predict(img_feat, l_pos, s_pos, n_pos, temp=1.0):
    s_L = (img_feat @ l_pos.T).squeeze(0).mean().item()
    s_S = (img_feat @ s_pos.T).squeeze(0).mean().item()
    s_N = (img_feat @ n_pos.T).squeeze(0).mean().item()
    scores = {"Lateral Lying": s_L / temp, "Sternal Lying": s_S / temp, "Not Lying": s_N / temp}
    return max(scores, key=scores.get), s_L, s_S, s_N

# ─────────────────────────────────────────────────────────────────────────────
# ★ \u4fdd\u5b58 few-shot \u9009\u6837\u8bb0\u5f55
# ─────────────────────────────────────────────────────────────────────────────
def save_fewshot_samples(train_items, output_dir, shots):
    """\u5c06 few-shot \u9009\u62e9\u7684\u6837\u672c\u8def\u5f84\u548c\u6807\u7b7e\u4fdd\u5b58\u4e3a CSV。"""
    rows = [{"shots": shots, "label": lbl, "filename": os.path.basename(p), "path": p}
            for p, lbl in train_items]
    df = pd.DataFrame(rows)
    save_path = os.path.join(output_dir, f"fewshot_samples_shots{shots}.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  [\u4fdd\u5b58] few-shot \u6837\u672c -> {save_path}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation\u903b\u8f91（\u542b\u63a8\u7406\u7ed3\u679c\u6536\u96c6）
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, ann_df, consistent_fnames, path_index, preprocess, device,
             temp=1.0, exclude_paths=None, collect_preds=False):
    model.eval()
    exclude = set(exclude_paths) if exclude_paths else set()
    l_pos, s_pos, n_pos = get_text_anchors(model, device)

    acc_preds, acc_gt = [], []
    kappa_preds = []
    kappa_gt = {a: [] for a in ANNOTATORS}
    annotator_labels = {a: [] for a in ANNOTATORS}
    pred_records = []

    for _, row in ann_df.iterrows():
        fname = row["\u6587\u4ef6\u540d"]
        p = path_index.get(fname)
        if p is None or p in exclude:
            continue

        feat = model.encode_image(load_img(p, preprocess, device))
        pred, s_L, s_S, s_N = softmax_predict(feat, l_pos, s_pos, n_pos, temp)

        kappa_preds.append(pred)
        for a in ANNOTATORS:
            kappa_gt[a].append(row[a])
            annotator_labels[a].append(row[a])

        is_consistent = fname in consistent_fnames
        gt_consensus  = row["\u5171\u8bc6\u6807\u7b7e"] if is_consistent else ""
        if is_consistent:
            acc_preds.append(pred)
            acc_gt.append(gt_consensus)

        if collect_preds:
            rec = {
                "filename":       fname,
                "path":           p,
                "pred_label":     pred,
                "score_lateral":  round(s_L, 6),
                "score_sternal":  round(s_S, 6),
                "score_notlying": round(s_N, 6),
                "is_consistent":  is_consistent,
                "gt_consensus":   gt_consensus,
                "in_train":       p in (exclude_paths or set()),
            }
            for a in ANNOTATORS:
                rec[a] = row[a]
            pred_records.append(rec)

    if acc_preds:
        acc       = accuracy_score(acc_gt, acc_preds)
        precision = precision_score(acc_gt, acc_preds, average='macro', zero_division=0)
        recall    = recall_score(acc_gt, acc_preds, average='macro', zero_division=0)
        f1        = f1_score(acc_gt, acc_preds, average='macro', zero_division=0)
    else:
        acc = precision = recall = f1 = 0.0

    if kappa_preds:
        kappas = {a: cohen_kappa_score(kappa_gt[a], kappa_preds, labels=ALL_LABELS) for a in ANNOTATORS}
        avg_k  = sum(kappas.values()) / len(kappas)
    else:
        kappas, avg_k = {}, 0.0

    inter_kappas = []
    for a1, a2 in itertools.combinations(ANNOTATORS, 2):
        if annotator_labels[a1]:
            inter_kappas.append(
                cohen_kappa_score(annotator_labels[a1], annotator_labels[a2], labels=ALL_LABELS)
            )
    avg_inter_kappa = sum(inter_kappas) / len(inter_kappas) if inter_kappas else 0.0

    n_skipped = sum(1 for _, row in ann_df.iterrows() if path_index.get(row["\u6587\u4ef6\u540d"]) in exclude)
    return (avg_k, acc, precision, recall, f1, avg_inter_kappa,
            kappas, n_skipped, len(acc_preds), len(kappa_preds), pred_records)

# ─────────────────────────────────────────────────────────────────────────────
# \u8bad\u7ec3\u5f15\u64ce
# ─────────────────────────────────────────────────────────────────────────────
def run_one(cfg, clip_model, base_state, train_items, ann_df, consistent_fnames,
            path_index, preprocess, device, collect_preds=False):
    set_seed(cfg["seed"])
    model = fresh_adapter_model(clip_model, base_state, device, cfg["dropout"], cfg["ratio"])

    for p in model.parameters():             p.requires_grad = False
    for p in model.img_adapter.parameters(): p.requires_grad = True
    for p in model.txt_adapter.parameters(): p.requires_grad = True

    params       = [p for p in model.parameters() if p.requires_grad]
    opt          = build_optimizer(cfg["optimizer"], params, cfg["lr"], weight_decay=1e-2)
    total_steps  = cfg["epochs"] * len(train_items)
    warmup_steps = int(total_steps * 0.1)
    scheduler    = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    scaler       = GradScaler()

    rng       = random.Random(cfg["seed"])
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        rng.shuffle(train_items)
        ep_loss = 0.0
        l_pos, s_pos, n_pos = get_text_anchors(model, device)

        for path, lbl in train_items:
            try:
                img_input = load_img(path, preprocess, device)
                target    = torch.tensor([LABEL_IDX[lbl]], device=device)

                with autocast():
                    feat   = model.encode_image(img_input)
                    s_L    = (feat @ l_pos.T).mean()
                    s_S    = (feat @ s_pos.T).mean()
                    s_N    = (feat @ n_pos.T).mean()
                    logits = torch.stack([s_L, s_S, s_N]).unsqueeze(0) / cfg["temp"]
                    loss   = F.cross_entropy(logits, target)

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
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

    (avg_k, acc, precision, recall, f1, avg_inter_kappa,
     kappas, n_skipped, n_acc, n_kappa, pred_records) = evaluate(
        model, ann_df, consistent_fnames, path_index, preprocess, device,
        cfg["temp"], exclude_paths=train_paths, collect_preds=collect_preds
    )

    return (avg_k, acc, precision, recall, f1, avg_inter_kappa,
            kappas, best_loss, best_state, n_skipped, n_acc, n_kappa, pred_records)

# ─────────────────────────────────────────────────────────────────────────────
# ★ Few-shot \u6837\u672c\u9009\u62e9（\u9012\u51cf\u7248）
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def kcenter_select_initial(paths, model, preprocess, device, k, seed=42):
    """
    \u9996\u8f6e：\u4ece paths \u4e2d\u7528 k-center \u8d2a\u5fc3\u9009 k \u4e2a\u6837\u672c（\u539f\u59cb\u903b\u8f91）。
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

    rng   = random.Random(seed)
    first = rng.randint(0, len(valid) - 1)
    sel   = [first]
    min_d = dist[first].copy()

    while len(sel) < k:
        for s in sel:
            min_d[s] = -1
        nxt = int(np.argmax(min_d))
        sel.append(nxt)
        min_d = np.minimum(min_d, dist[nxt])

    return [valid[i] for i in sel]


@torch.no_grad()
def kcenter_remove_one(paths, model, preprocess, device):
    """
    ★ \u9012\u51cf\u903b\u8f91：\u4ece\u5df2\u9009\u7684 paths \u4e2d\u79fb\u9664"\u6700\u4e0d\u91cd\u8981"\u7684 1 \u4e2a\u6837\u672c。

    "\u6700\u4e0d\u91cd\u8981"\u5b9a\u4e49\u4e3a：\u4e0e\u5176\u4f59\u6837\u672c\u7684\u6700\u5c0f\u4f59\u5f26\u8ddd\u79bb\u6700\u5927\u7684\u70b9
    （\u5373\u5b83\u88ab\u5176\u4ed6\u70b9"\u8986\u76d6"\u5f97\u6700\u597d，\u79fb\u9664\u540e\u5bf9\u8986\u76d6\u635f\u5931\u6700\u5c0f）。

    \u7b56\u7565\u7b49\u4ef7\u4e8e：\u79fb\u9664 k-center \u96c6\u5408\u4e2d\u6700\u540e\u88ab\u9009\u5165\u7684\u90a3\u4e2a\u70b9，
    \u5373\u5bf9\u96c6\u5408\u4e2d\u6bcf\u4e2a\u70b9\u8ba1\u7b97"\u82e5\u79fb\u9664\u5b83，\u5176\u4f59\u70b9\u5bf9\u5168\u96c6\u7684\u6700\u5927\u6700\u8fd1\u8ddd\u79bb
    \u589e\u91cf\u6700\u5c0f"——\u4f46\u6b64\u5904\u7528\u8fd1\u4f3c\u5feb\u901f\u7248：
    \u627e\u5230\u96c6\u5408\u5185\u90e8\u70b9\u5bf9\u8ddd\u79bb\u6700\u5c0f\u7684\u90a3\u5bf9，\u79fb\u9664\u5176\u4e2d\u4e0d\u662f"\u7b2c\u4e00\u4e2a\u9009\u5165"\u7684\u70b9。

    \u5b9e\u9645\u5b9e\u73b0：\u8ba1\u7b97\u96c6\u5408\u5185\u6240\u6709\u70b9\u4e24\u4e24\u4f59\u5f26\u8ddd\u79bb，
    \u627e\u5230\u96c6\u5408\u5185\u4f59\u5f26\u76f8\u4f3c\u5ea6\u6700\u9ad8（\u8ddd\u79bb\u6700\u5c0f）\u7684\u4e00\u5bf9 (i, j)，
    \u79fb\u9664\u5176\u4e2d\u7d22\u5f15\u8f83\u5927\u7684\u70b9（\u7b80\u5355\u8fd1\u4f3c，\u4e0e\u589e\u91cf\u9009\u6837\u5bf9\u79f0）。
    """
    if len(paths) <= 1:
        return paths  # \u65e0\u6cd5\u518d\u51cf

    feats, valid = [], []
    for p in paths:
        try:
            f = model.encode_image(load_img(p, preprocess, device)).squeeze(0).cpu()
            feats.append(f)
            valid.append(p)
        except Exception:
            continue

    if len(valid) <= 1:
        return valid

    feats = torch.stack(feats)  # (n, d)
    # \u4f59\u5f26\u76f8\u4f3c\u5ea6\u77e9\u9635，\u5bf9\u89d2\u7f6e -inf \u6392\u9664\u81ea\u8eab
    sim = (feats @ feats.T).numpy()
    np.fill_diagonal(sim, -np.inf)

    # \u627e\u96c6\u5408\u5185\u76f8\u4f3c\u5ea6\u6700\u9ad8\u7684\u70b9\u5bf9\u4e2d\u7684"\u8f83\u5197\u4f59"\u90a3\u4e2a：
    # \u5bf9\u6bcf\u4e2a\u70b9，\u53d6\u5b83\u4e0e\u96c6\u5408\u5185\u5176\u4ed6\u70b9\u7684\u6700\u5927\u76f8\u4f3c\u5ea6（\u5373\u6700\u8fd1\u90bb\u76f8\u4f3c\u5ea6）
    # \u79fb\u9664\u6700\u5927\u76f8\u4f3c\u5ea6\u6700\u9ad8\u7684\u70b9（\u8be5\u70b9\u88ab\u5176\u4ed6\u70b9\u8986\u76d6\u5f97\u6700\u597d，\u6700\u5197\u4f59）
    max_sim_per_point = sim.max(axis=1)   # shape (n,)
    remove_idx = int(np.argmax(max_sim_per_point))

    remaining = [p for i, p in enumerate(valid) if i != remove_idx]
    print(f"    [\u9012\u51cf] \u79fb\u9664: {os.path.basename(valid[remove_idx])}")
    return remaining


def build_train_items_initial(ann_df, consistent_fnames, path_index, model, preprocess,
                               device, n_shots, seed=42):
    """
    ★ \u9996\u8f6e：\u7528 k-center \u4e3a\u6bcflabels\u9009 n_shots \u4e2a\u6837\u672c。
    """
    items = []
    cons_df = ann_df[ann_df["\u6587\u4ef6\u540d"].isin(consistent_fnames)]
    for lbl in ALL_LABELS:
        cands = sorted(
            [path_index[r["\u6587\u4ef6\u540d"]]
             for _, r in cons_df[cons_df["\u5171\u8bc6\u6807\u7b7e"] == lbl].iterrows()
             if r["\u6587\u4ef6\u540d"] in path_index],
            key=os.path.basename
        )
        chosen = kcenter_select_initial(cands, model, preprocess, device,
                                        k=n_shots, seed=seed)
        items.extend((p, lbl) for p in chosen)
    return items


def build_train_items_decrement(prev_items, model, preprocess, device):
    """
    ★ \u9012\u51cf\u8f6e：\u5728\u4e0a\u4e00\u8f6e\u9009\u6837\u57fa\u7840\u4e0a，\u6bcflabels\u7c7b\u522b\u5404\u79fb\u9664 1 \u4e2a\u6700\u5197\u4f59\u6837\u672c。

    \u53c2\u6570
    ----
    prev_items : list[(path, label)]  \u4e0a\u4e00\u8f6e\u7684\u8bad\u7ec3\u6837\u672c\u5217\u8868

    \u8fd4\u56de
    ----
    list[(path, label)]  \u51cf\u5c11\u540e\u7684\u8bad\u7ec3\u6837\u672c\u5217\u8868
    """
    # \u6309\u6807\u7b7e\u5206\u7ec4
    by_label: dict[str, list[str]] = {lbl: [] for lbl in ALL_LABELS}
    for p, lbl in prev_items:
        by_label[lbl].append(p)

    new_items = []
    for lbl in ALL_LABELS:
        paths = by_label[lbl]
        print(f"  [{lbl}] \u5f53\u524d {len(paths)} \u4e2a\u6837\u672c -> \u79fb\u9664 1 \u4e2a ...", end=" ")
        if len(paths) <= 1:
            print("\u6837\u672c\u4e0d\u8db3，\u4fdd\u7559\u539f\u6837。")
            new_items.extend((p, lbl) for p in paths)
            continue
        remaining = kcenter_remove_one(paths, model, preprocess, device)
        new_items.extend((p, lbl) for p in remaining)

    return new_items

# ─────────────────────────────────────────────────────────────────────────────
# \u5355\u4e2a shots \u503c\u7684\u5b8c\u6574\u5b9e\u9a8c
# ─────────────────────────────────────────────────────────────────────────────
def run_shots_experiment(shots, args, clip_model, base_state, base_m,
                         ann_df, consistent_fnames, path_index, eval_preprocess, device,
                         prev_train_items=None, is_first_round=False):
    """
    \u9488\u5bf9\u7ed9\u5b9a shots \u6570\u91cf\u8dd1\u5b8c\u6574\u8d85\u53c2\u641c\u7d22。

    \u53c2\u6570
    ----
    prev_train_items : list[(path, label)] | None
        \u4e0a\u4e00\u8f6e（shots+1）\u9009\u597d\u7684\u8bad\u7ec3\u6837\u672c，\u7528\u4e8e\u9012\u51cf\u88c1\u526a。
        \u9996\u8f6e（shots_max）\u4f20 None，\u4ece\u5934\u7528 k-center \u9009\u6837。
    is_first_round : bool
        \u662f\u5426\u4e3a\u9996\u8f6e（shots_max），\u9996\u8f6e\u7528 k-center \u521d\u59cb\u9009\u6837。

    \u8fd4\u56de
    ----
    best_row      : dict      \u6700\u4f73\u8d85\u53c2\u5bf9\u5e94\u7684\u7ed3\u679c\u884c
    df_res        : DataFrame \u672c\u8f6e\u6240\u6709\u8d85\u53c2\u7ed3\u679c
    train_items   : list      \u672c\u8f6e\u6700\u7ec8\u4f7f\u7528\u7684\u8bad\u7ec3\u6837\u672c（\u4f9b\u4e0b\u4e00\u8f6e\u9012\u51cf）
    """
    print(f"\n{'='*60}")
    print(f"  ★ shots = {shots}")
    if not is_first_round:
        print(f"    （\u5728\u4e0a\u4e00\u8f6e {shots+1} shots \u57fa\u7840\u4e0a\u6bcf\u7c7b\u5404\u79fb\u9664 1 \u4e2a\u5197\u4f59\u6837\u672c）")
    print(f"{'='*60}")

    shots_dir = os.path.join(args.output_dir, f"shots_{shots}")
    os.makedirs(shots_dir, exist_ok=True)

    # ── 1. \u9009\u6837（\u9996\u8f6e k-center，\u540e\u7eed\u9012\u51cf）────────────────────────────────────
    if is_first_round:
        print(f"  [\u9996\u8f6e] \u7528 k-center \u4ece\u5168\u91cf\u5019\u9009\u4e2d\u9009 {shots} shots ...")
        train_items = build_train_items_initial(
            ann_df, consistent_fnames, path_index,
            base_m, eval_preprocess, device,
            n_shots=shots, seed=args.seed
        )
    else:
        print(f"  [\u9012\u51cf] \u6bcf\u7c7b\u79fb\u9664 1 \u4e2a\u6700\u5197\u4f59\u6837\u672c ...")
        train_items = build_train_items_decrement(
            prev_train_items, base_m, eval_preprocess, device
        )

    save_fewshot_samples(train_items, shots_dir, shots)

    # \u6253\u5370\u672c\u8f6e\u6837\u672c\u5217\u8868
    if not is_first_round and prev_train_items is not None:
        prev_paths  = {p for p, _ in prev_train_items}
        cur_paths   = {p for p, _ in train_items}
        removed     = prev_paths - cur_paths
        print(f"  \u672c\u8f6e\u79fb\u9664 {len(removed)} \u4e2a\u6837\u672c：")
        for p in sorted(removed):
            lbl = next(l for pp, l in prev_train_items if pp == p)
            print(f"    [{lbl}] {os.path.basename(p)}")

    # ── 2. \u8d85\u53c2\u7f51\u683c ──────────────────────────────────────────────────────────
    if args.quick:
        optimizer_l, lr_l, epochs_l, dropout_l, ratio_l = ["adamw"], [5e-3], [100], [0.1], [0.2]
    else:
        optimizer_l = ["adamw", "adam"]
        lr_l        = [1e-4, 3e-4, 5e-4]
        epochs_l    = [50, 80]
        dropout_l   = [0.1]
        ratio_l     = [0.2, 0.4]

    grid    = list(itertools.product(optimizer_l, lr_l, epochs_l, dropout_l, ratio_l, [2.0]))
    results = []
    best_avg_k, best_cfg, best_ft_state, best_pred_records = -1.0, None, None, []

    for i, (opt_name, lr, epochs, dropout, ratio, temp) in enumerate(grid, 1):
        cfg = dict(optimizer=opt_name, lr=lr, epochs=epochs,
                   dropout=dropout, ratio=ratio, temp=temp, seed=args.seed)
        t0  = time.time()
        print(f"  [{i:3d}/{len(grid)}] {opt_name} LR={lr:.0e} EP={epochs} Ratio={ratio}...",
              end=" ", flush=True)

        try:
            (avg_k, acc, precision, recall, f1, avg_inter_kappa,
             kappas, best_loss_val, ft_state, n_skipped, n_acc, n_kappa, _) = run_one(
                cfg, clip_model, base_state, train_items,
                ann_df, consistent_fnames, path_index, eval_preprocess, device,
                collect_preds=False
            )

            print(f"Pred-κ={avg_k:.4f} Inter-κ={avg_inter_kappa:.4f} "
                  f"Acc={acc*100:.1f}% F1={f1:.4f} ({time.time()-t0:.1f}s)")

            results.append({
                "shots":     shots,
                "optimizer": cfg["optimizer"],
                "lr":        cfg["lr"],
                "epochs":    cfg["epochs"],
                "dropout":   cfg["dropout"],
                "ratio":     cfg["ratio"],
                "temp":      cfg["temp"],
                "avg_pred_kappa":            avg_k,
                "avg_inter_annotator_kappa": avg_inter_kappa,
                "accuracy":  acc,
                "precision": precision,
                "recall":    recall,
                "f1_score":  f1,
                "kappa_n":   n_kappa,
                "acc_n":     n_acc,
                "loss":      best_loss_val,
            })

            if avg_k > best_avg_k:
                best_avg_k    = avg_k
                best_cfg      = cfg
                best_ft_state = ft_state

        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()

    if not results:
        print(f"  ❌ shots={shots} \u6240\u6709\u5b9e\u9a8c\u5747\u5931\u8d25，\u8df3\u8fc7。")
        return None, None, train_items

    # ── 3. \u4fdd\u5b58\u8d85\u53c2\u641c\u7d22\u7ed3\u679c ──────────────────────────────────────────────────
    df_res = pd.DataFrame(results).sort_values("avg_pred_kappa", ascending=False)
    csv_path = os.path.join(shots_dir, f"results_shots{shots}.csv")
    df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  [\u4fdd\u5b58] \u8d85\u53c2\u641c\u7d22\u7ed3\u679c -> {csv_path}")

    # ── 4. \u7528\u6700\u4f73\u8d85\u53c2\u91cd\u8dd1\u4e00\u6b21，\u6536\u96c6\u5b8c\u6574\u63a8\u7406\u660e\u7ec6 ─────────────────────────────
    print(f"  [\u63a8\u7406] \u7528\u6700\u4f73\u8d85\u53c2\u91cd\u65b0\u63a8\u7406\u5e76\u6536\u96c6\u9010\u56fe\u7ed3\u679c ...")
    try:
        (best_avg_k_re, acc_re, precision_re, recall_re, f1_re, avg_inter_re,
         kappas_re, _, best_ft_state_re, _, _, _, pred_records) = run_one(
            best_cfg, clip_model, base_state, train_items,
            ann_df, consistent_fnames, path_index, eval_preprocess, device,
            collect_preds=True
        )
        best_ft_state     = best_ft_state_re
        best_pred_records = pred_records
    except Exception as e:
        print(f"  ⚠ \u91cd\u8dd1\u63a8\u7406\u5931\u8d25: {e}，\u4f7f\u7528\u9996\u6b21\u7ed3\u679c。")

    # ── 5. \u4fdd\u5b58\u9010\u56fe\u63a8\u7406\u7ed3\u679c ──────────────────────────────────────────────────
    if best_pred_records:
        pred_df = pd.DataFrame(best_pred_records)
        train_fnames = {os.path.basename(p) for p, _ in train_items}
        pred_df["is_train_sample"] = pred_df["filename"].isin(train_fnames)
        pred_csv = os.path.join(shots_dir, f"inference_results_shots{shots}.csv")
        pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
        print(f"  [\u4fdd\u5b58] \u63a8\u7406\u660e\u7ec6 ({len(pred_df)} images) -> {pred_csv}")

    # ── 6. \u4fdd\u5b58\u6700\u4f73\u6a21\u578b ──────────────────────────────────────────────────────
    best_pth = os.path.join(shots_dir, f"best_model_shots{shots}.pth")
    torch.save({
        "model_state_dict": best_ft_state,
        "config":           best_cfg,
        "avg_kappa":        best_avg_k,
        "shots":            shots,
    }, best_pth)
    print(f"  [\u4fdd\u5b58] \u6700\u4f73\u6a21\u578b -> {best_pth}")
    print(f"  ✅ shots={shots} \u5b8c\u6210，\u6700\u4f73 Pred-κ = {best_avg_k:.4f}")

    best_row = df_res.iloc[0].to_dict()
    return best_row, df_res, train_items

# ─────────────────────────────────────────────────────────────────────────────
# \u4e3b\u6d41\u7a0b
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--model",      required=True)
    parser.add_argument("--output_dir", default="results_v11")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--shots_min",  type=int, default=5,  help="shots \u8303\u56f4\u4e0b\u754c（\u542b）")
    parser.add_argument("--shots_max",  type=int, default=10, help="shots \u8303\u56f4\u4e0a\u754c（\u542b），\u9996\u8f6e\u4ece\u6b64\u5f00\u59cb")
    parser.add_argument("--quick",      action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── \u5168\u5c40\u6570\u636e\u52a0\u8f7d ──────────────────────────────────────────────────────────
    ann_df, consistent_fnames = load_annotations(args.annotation)
    path_index = build_path_index(args.data_root)
    ann_df = ann_df[ann_df["\u6587\u4ef6\u540d"].isin(path_index)].reset_index(drop=True)
    consistent_fnames = consistent_fnames & set(path_index.keys())
    print(f"\u603b\u56fe\u50cf: {len(path_index)}  \u4e00\u81f4\u6027\u6837\u672c: {len(consistent_fnames)}")

    # ── CLIP + base \u6a21\u578b\u52a0\u8f7d ──────────────────────────────────────────────────
    clip_model, eval_preprocess = clip.load("ViT-B/32", device=device)
    ckpt   = torch.load(args.model, map_location=device)
    base_m = ProjCLIPWithAdapter(clip_model).to(device)
    base_m.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    base_state = {k: v.clone() for k, v in base_m.state_dict().items()}

    # ── ★ shots \u5faa\u73af（\u9012\u51cf：\u4ece shots_max \u5230 shots_min）──────────────────────
    # \u987a\u5e8f：shots_max, shots_max-1, ..., shots_min
    shots_range      = range(args.shots_max, args.shots_min - 1, -1)
    summary_rows     = []
    all_grids        = []
    prev_train_items = None   # \u9996\u8f6e\u65e0\u524d\u7f6e\u9009\u6837

    for shots in shots_range:
        is_first = (shots == args.shots_max)   # ★ \u9996\u8f6e\u6807\u5fd7

        best_row, df_res, train_items = run_shots_experiment(
            shots, args, clip_model, base_state, base_m,
            ann_df, consistent_fnames, path_index, eval_preprocess, device,
            prev_train_items=prev_train_items,
            is_first_round=is_first             # ★ \u4f20\u5165\u9996\u8f6e\u6807\u5fd7
        )

        # ★ \u5c06\u672c\u8f6e\u9009\u6837\u4f5c\u4e3a\u4e0b\u4e00\u8f6e\u7684\u9012\u51cf\u57fa\u7840
        prev_train_items = train_items

        if best_row is not None:
            best_row["shots"] = shots
            summary_rows.append(best_row)
        if df_res is not None:
            all_grids.append(df_res)

    # ── \u6c47\u603b\u6240\u6709 shots \u7684\u6700\u4f73\u7ed3\u679c（\u6309 shots \u5347\u5e8f\u5c55\u793a）──────────────────────
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("shots")
        summary_csv = os.path.join(args.output_dir, "summary_all_shots.csv")
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print(f"\n{'='*60}")
        print(f"✅ \u5168\u90e8 shots \u5b9e\u9a8c\u5b8c\u6210！\u6c47\u603b\u7ed3\u679c -> {summary_csv}")
        print(summary_df[["shots", "avg_pred_kappa", "accuracy", "f1_score",
                           "avg_inter_annotator_kappa"]].to_string(index=False))

    # ── \u6240\u6709\u8d85\u53c2\u7ec4\u5408\u7684\u5b8c\u6574\u7ed3\u679c\u5408\u5e76\u4fdd\u5b58 ──────────────────────────────────────
    if all_grids:
        full_grid_df  = pd.concat(all_grids, ignore_index=True)
        full_grid_csv = os.path.join(args.output_dir, "all_grid_results.csv")
        full_grid_df.to_csv(full_grid_csv, index=False, encoding="utf-8-sig")
        print(f"\u6240\u6709\u8d85\u53c2\u7ed3\u679c\u5408\u5e76 -> {full_grid_csv}")


if __name__ == "__main__":
    main()