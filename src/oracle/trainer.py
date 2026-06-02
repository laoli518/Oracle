"""Training loop for projection heads."""

import logging
import os
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PigBehaviorDirectDataset
from .evaluation import (
    collate_precomputed_batch,
    evaluate_direct_model,
    evaluate_validation_full_descriptions,
)
from .model import CustomCLIPContrastiveTrainer
from .utils import _save_json, _save_per_epoch_metrics

logger = logging.getLogger(__name__)

def train_direct_model(
    model: CustomCLIPContrastiveTrainer,
    train_dataset: PigBehaviorDirectDataset,
    val_dataset: PigBehaviorDirectDataset,
    device,
    num_epochs: int = 20,
    learning_rate: float = 1e-5,
    batch_size: int = 128,
    save_dir: Optional[str] = None,
):
    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=lambda b: b,
            
        )

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4, eps=1e-8
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_acc  = 0.0
    best_model_state = None
    per_epoch_metrics: List[dict] = []

    logger.info("Start training...")

    for epoch in range(num_epochs):
        # ── train stage：train mode（random select descriptions） ───────────────────────────────
        train_dataset.set_train_mode(seed=42 + epoch)
        torch.manual_seed(42 + epoch) 
        g = torch.Generator()
        g.manual_seed(42 + epoch)   # Each epoch uses a fixed but different shuffle order
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda b: b,
            generator=g,
        )
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for step, batch in enumerate(pbar):
            try:
                image_feats, text_feats, label_indices, is_positive_pairs, labels, _ = \
                    collate_precomputed_batch(batch, device)

                optimizer.zero_grad()
                proj_img, proj_txt = model(image_feats, text_feats)
                if proj_img is None:
                    continue

                total_loss, pos_loss, neg_loss = model.compute_contrastive_loss(
                    proj_img, proj_txt, is_positive_pairs
                )
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_losses.append(total_loss.item())
                pbar.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    pos=f"{pos_loss.item():.4f}",
                    neg=f"{neg_loss.item():.4f}",
                )
            except Exception as e:
                logger.warning(f"Step {step}: {e}")

        if not epoch_losses:
            logger.error(f"Epoch {epoch+1}: No successful step was completed. Terminating training")
            break

        avg_train_loss = float(np.mean(epoch_losses))
        train_losses.append(avg_train_loss)
        scheduler.step()

        # ── 评估阶段：eval 模式（全量正/负描述） ─────────────────────────────
        logger.info(f"Epoch {epoch+1}: Set the dataset to eval mode for evaluation....")
        train_dataset.set_eval_mode()
        val_dataset.set_eval_mode()

        train_eval_loader = make_loader(train_dataset, shuffle=False)
        val_eval_loader   = make_loader(val_dataset,   shuffle=False)

        train_acc, train_eval_loss, train_per_class, train_sims = evaluate_direct_model(
            model, train_eval_loader, device,
            save_similarity_values=True, epoch_idx=epoch + 1,
            split_name="train", save_dir=save_dir,
        )
        val_acc, val_eval_loss, val_per_class, val_sims = evaluate_direct_model(
            model, val_eval_loader, device,
            save_similarity_values=True, epoch_idx=epoch + 1,
            split_name="val", save_dir=save_dir,
        )
        val_losses.append(val_eval_loss)
        val_accuracies.append(val_acc)

        full_eval = evaluate_validation_full_descriptions(
            model, val_dataset, device,
            save_dir=save_dir, epoch_idx=epoch + 1,
        )

        if save_dir:
            merged_path = os.path.join(
                save_dir, "similarity", f"similarity_values_epoch_{epoch + 1}.json"
            )
            _save_json({"epoch": epoch + 1, "train": train_sims, "val": val_sims}, merged_path)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train": {
                "overall_acc": train_acc,
                "avg_loss":    float(train_eval_loss),
                "per_class":   train_per_class,
            },
            "val": {
                "overall_acc": val_acc,
                "avg_loss":    float(val_eval_loss),
                "per_class":   val_per_class,
            },
            "val_full_desc_eval": full_eval,
        }
        per_epoch_metrics.append(epoch_metrics)
        if save_dir:
            _save_per_epoch_metrics(save_dir, epoch + 1, epoch_metrics)

        logger.info(
            f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} "
            f"train_acc={train_acc:.2f}%  val_acc={val_acc:.2f}%"
            f"  [eval_mode: full positive/negative descriptions]"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(f"  ✓ New best validation accuracy: {best_val_acc:.2f}%")

    if best_model_state:
        model.load_state_dict(best_model_state)

    return {
        "train_losses":      train_losses,
        "val_losses":        val_losses,
        "val_accuracies":    val_accuracies,
        "best_val_acc":      best_val_acc,
        "per_epoch_metrics": per_epoch_metrics,
    }
