# ============================================================
# training.py
# Learning-rate scheduling and the train/validate/fit loop.
#
# Best-checkpoint selection is based on val_dice + val_iou computed with
# the SAME postprocessing (threshold + optional morphology) used by
# evaluation.evaluate_model, so "best" means best under the exact metric
# you actually report -- not an approximation of it.
#
# use_amp enables automatic mixed precision (torch.autocast + GradScaler)
# to cut GPU memory usage, with no change to the loss/metrics computed.
# ============================================================

import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .metrics import dice_score, iou_score
from .postprocessing import postprocess_prediction

logger = logging.getLogger(__name__)


def poly_lr_scheduler(optimizer, init_lr, epoch, max_epoch, power=0.9):
    lr = init_lr * (1 - epoch / max_epoch) ** power

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def logits_to_probs(outputs):
    """Mirror each model's predict_proba() logic, without a second forward pass."""
    logits = outputs["logits"]

    if isinstance(logits, list):
        probs = [torch.sigmoid(logit) for logit in logits]
        return torch.mean(torch.stack(probs, dim=0), dim=0)

    return torch.sigmoid(logits)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp=False):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(
    model, loader, criterion, device,
    threshold=0.5, kernel_size=10, use_postprocessing=True, use_amp=False
):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
            probs = logits_to_probs(outputs)

        probs_np = probs.float().cpu().numpy()
        masks_np = masks.cpu().numpy()
        batch_size = images.size(0)

        for i in range(batch_size):
            pred_mask = postprocess_prediction(
                probs_np[i, 0],
                threshold=threshold,
                kernel_size=kernel_size,
                use_postprocessing=use_postprocessing
            )
            gt_mask = masks_np[i, 0].astype(np.uint8)

            # float(...): dice_score/iou_score return numpy scalars, which
            # must not leak into the checkpoint dict below -- torch.load's
            # weights_only=True default (PyTorch >= 2.6) rejects numpy
            # scalar types.
            total_dice += float(dice_score(pred_mask, gt_mask))
            total_iou += float(iou_score(pred_mask, gt_mask))

        total_loss += loss.item() * batch_size

    num_samples = len(loader.dataset)
    return total_loss / num_samples, total_dice / num_samples, total_iou / num_samples


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    init_lr,
    checkpoint_path,
    model_key,
    model_display_name,
    fold_id,
    power=0.9,
    threshold=0.5,
    kernel_size=10,
    use_postprocessing=True,
    use_amp=False
):
    best_val_score = float("-inf")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {
        "model": [],
        "fold": [],
        "epoch": [],
        "learning_rate": [],
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_score": []
    }

    for epoch in range(1, epochs + 1):
        lr = poly_lr_scheduler(optimizer, init_lr=init_lr, epoch=epoch, max_epoch=epochs, power=power)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp=use_amp)
        val_loss, val_dice, val_iou = validate_one_epoch(
            model, val_loader, criterion, device,
            threshold=threshold, kernel_size=kernel_size,
            use_postprocessing=use_postprocessing, use_amp=use_amp
        )
        val_score = val_dice + val_iou

        history["model"].append(model_key)
        history["fold"].append(fold_id)
        history["epoch"].append(epoch)
        history["learning_rate"].append(lr)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_score"].append(val_score)

        logger.info(
            f"[{model_display_name} | Fold {fold_id}] "
            f"Epoch [{epoch}/{epochs}] "
            f"LR: {lr:.7f} Train Loss: {train_loss:.5f} Val Loss: {val_loss:.5f} "
            f"Val Dice: {val_dice:.5f} Val IoU: {val_iou:.5f} Val Score: {val_score:.5f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(
                {
                    "epoch": epoch,
                    "model_key": model_key,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "val_score": val_score
                },
                checkpoint_path
            )
            logger.info(f"Best model saved (val_score={val_score:.5f}): {checkpoint_path}")

    return pd.DataFrame(history)
