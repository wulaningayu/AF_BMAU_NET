# ============================================================
# evaluation.py
# Model evaluation, result summaries, and loss-curve plotting.
# ============================================================

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm

from .postprocessing import postprocess_prediction, compute_deepest_vertical_pocket
from .metrics import compute_all_metrics
from .visualization import save_overlay

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    fold_id,
    model_key,
    model_display_name,
    prediction_dir,
    overlay_dir,
    threshold=0.5,
    kernel_size=10,
    use_postprocessing=True,
    show_gt_on_overlay=False,
    save_predictions=True,
    save_overlays=True
):
    model.eval()
    results = []

    fold_pred_dir = Path(prediction_dir) / model_key / f"fold_{fold_id}"
    fold_overlay_dir = Path(overlay_dir) / model_key / f"fold_{fold_id}"

    fold_pred_dir.mkdir(parents=True, exist_ok=True)
    fold_overlay_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc=f"Testing {model_display_name} Fold {fold_id}"):
        images = batch["image"].to(device)
        masks = batch["mask"].cpu().numpy()
        image_paths = batch["image_path"]

        probs = model.predict_proba(images).cpu().numpy()
        batch_size = images.size(0)

        for i in range(batch_size):
            prob_map = probs[i, 0]
            gt_mask = masks[i, 0].astype(np.uint8)

            pred_mask = postprocess_prediction(
                prob_map,
                threshold=threshold,
                kernel_size=kernel_size,
                use_postprocessing=use_postprocessing
            )
            # dice/iou/precision/etc. are scale-invariant ratios, so it's
            # fine to compute them at the model's working resolution.
            metrics = compute_all_metrics(pred_mask, gt_mask)

            image_filename = Path(image_paths[i]).name
            image_stem = Path(image_paths[i]).stem

            # DVP is a physical distance in pixels, so it must be measured
            # at the original image's resolution -- not the (possibly much
            # smaller) working resolution -- or its value silently changes
            # meaning whenever image_size changes. Read the original image
            # once, resize masks up to it, and reuse that single resized
            # pred_mask both for the DVP measurement below and for the
            # overlay, so the CSV and the overlay always agree.
            original_image = cv2.imread(str(image_paths[i]), cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                raise FileNotFoundError(f"Cannot read original image for DVP measurement: {image_paths[i]}")
            original_h, original_w = original_image.shape[:2]

            pred_mask_full = cv2.resize(
                pred_mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST
            )
            gt_mask_full = cv2.resize(
                gt_mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST
            )
            dvp_info = compute_deepest_vertical_pocket(pred_mask_full)

            row = {
                "model": model_key,
                "model_name": model_display_name,
                "filename": image_filename,
                "fold": fold_id,
                **metrics,
                "pred_dvp_px": dvp_info["length_px"],
                "pred_dvp_x": dvp_info["x"],
                "pred_dvp_y_top": dvp_info["y_top"],
                "pred_dvp_y_bottom": dvp_info["y_bottom"],
                "dvp_image_width": original_w,
                "dvp_image_height": original_h
            }
            results.append(row)

            if save_predictions:
                pred_save_path = fold_pred_dir / f"{image_stem}_pred.png"
                cv2.imwrite(str(pred_save_path), pred_mask * 255)
                prob_save_path = fold_pred_dir / f"{image_stem}_prob.png"
                cv2.imwrite(str(prob_save_path), (prob_map * 255).astype(np.uint8))

            if save_overlays:
                overlay_save_path = fold_overlay_dir / f"{image_stem}_overlay.png"
                save_overlay(
                    image=original_image,
                    filename=image_filename,
                    gt_mask=gt_mask_full,
                    pred_mask=pred_mask_full,
                    save_path=overlay_save_path,
                    model_name=model_display_name,
                    dice_value=metrics["dice"],
                    iou_value=metrics["iou"],
                    alpha=0.35,
                    show_gt=show_gt_on_overlay
                )

    return pd.DataFrame(results)


def print_summary(df, title):
    metric_cols = ["dice", "iou", "precision", "sensitivity", "specificity", "hd95", "pred_dvp_px"]
    available_cols = [col for col in metric_cols if col in df.columns]

    summary_mean = df[available_cols].mean(numeric_only=True)
    summary_std = df[available_cols].std(numeric_only=True)

    summary_df = pd.DataFrame({"mean": summary_mean, "std": summary_std})

    # hd95 is undefined (NaN) when the prediction or ground truth mask is
    # empty. pandas' mean()/std() silently drop those rows -- surface the
    # count so a model that sometimes predicts nothing isn't hidden by an
    # hd95_mean computed over fewer, easier samples than the other metrics.
    if "hd95" in df.columns:
        hd95_nan_count = int(df["hd95"].isna().sum())
        if hd95_nan_count > 0:
            logger.warning(
                f"{hd95_nan_count}/{len(df)} samples had an undefined hd95 "
                "(empty prediction or empty ground truth mask) and were excluded "
                "from the hd95 mean/std above."
            )

    logger.info("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70 + f"\n{summary_df}\n" + "=" * 70)
    return summary_df


def summarize_all_models(all_results_df, models_to_run, model_display_names):
    metric_cols = ["dice", "iou", "precision", "sensitivity", "specificity", "hd95", "pred_dvp_px"]
    rows = []
    for model_key in models_to_run:
        model_df = all_results_df[all_results_df["model"] == model_key]
        if model_df.empty:
            continue
        row = {"model": model_key, "model_name": model_display_names[model_key]}
        for col in metric_cols:
            row[f"{col}_mean"] = model_df[col].mean()
            row[f"{col}_std"] = model_df[col].std()
        if "hd95" in model_df.columns:
            row["hd95_nan_count"] = int(model_df["hd95"].isna().sum())
            row["hd95_valid_count"] = int(model_df["hd95"].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_loss_curves(history_df, result_dir, models_to_run, model_display_names):
    loss_dir = Path(result_dir) / "loss_curves"
    loss_dir.mkdir(parents=True, exist_ok=True)

    if history_df.empty:
        return

    avg_history = (
        history_df
        .groupby(["model", "epoch"], as_index=False)[["train_loss", "val_loss"]]
        .mean()
    )

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for model_key in models_to_run:
        model_hist = avg_history[avg_history["model"] == model_key]
        if model_hist.empty:
            continue
        plt.plot(model_hist["epoch"], model_hist["train_loss"], label=model_display_names[model_key])
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    for model_key in models_to_run:
        model_hist = avg_history[avg_history["model"] == model_key]
        if model_hist.empty:
            continue
        plt.plot(model_hist["epoch"], model_hist["val_loss"], label=model_display_names[model_key])
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(loss_dir / "train_val_loss_comparison_all_models.png", dpi=300)
    plt.close()

    # Also save separated training and validation curves.
    for loss_col, title, fname in [
        ("train_loss", "Training Loss Comparison", "training_loss_comparison.png"),
        ("val_loss", "Validation Loss Comparison", "validation_loss_comparison.png")
    ]:
        plt.figure(figsize=(9, 6))
        for model_key in models_to_run:
            model_hist = avg_history[avg_history["model"] == model_key]
            if model_hist.empty:
                continue
            plt.plot(model_hist["epoch"], model_hist[loss_col], label=model_display_names[model_key])
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_dir / fname, dpi=300)
        plt.close()

    avg_history.to_csv(loss_dir / "average_loss_by_model_epoch.csv", index=False)
