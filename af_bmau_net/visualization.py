# ============================================================
# visualization.py
# Overlay rendering and side-by-side model comparison panels.
# ============================================================

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .postprocessing import compute_deepest_vertical_pocket


def save_overlay(
    image,
    filename,
    gt_mask,
    pred_mask,
    save_path,
    model_name=None,
    dice_value=None,
    iou_value=None,
    alpha=0.35,
    show_gt=False
):
    """
    Save original-resolution overlay.
    Prediction area is shown as transparent yellow on the original ultrasound image.

    `image` is the original grayscale ultrasound frame (already loaded), and
    `gt_mask`/`pred_mask` must already be resized to match its resolution.
    The caller (evaluate_model) does this resize once and reuses the same
    pred_mask for its DVP measurement, so the saved CSV and this overlay
    always agree on where/how long the deepest vertical pocket is.
    """
    original_h, original_w = image.shape[:2]

    gt_mask = (gt_mask > 0).astype(np.uint8)
    pred_mask = (pred_mask > 0).astype(np.uint8)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = image_rgb.copy()

    # Yellow in BGR
    overlay[pred_mask == 1] = (0, 255, 255)
    blended = cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)

    pred_contours, _ = cv2.findContours(
        pred_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, pred_contours, -1, (0, 165, 255), 2)

    if show_gt:
        gt_contours, _ = cv2.findContours(
            gt_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(blended, gt_contours, -1, (0, 255, 0), 1)

    dvp_info = compute_deepest_vertical_pocket(pred_mask)
    if dvp_info["x"] is not None:
        x = int(dvp_info["x"])
        y_top = int(dvp_info["y_top"])
        y_bottom = int(dvp_info["y_bottom"])
        cv2.line(blended, (x, y_top), (x, y_bottom), (255, 255, 0), 2)
        cv2.circle(blended, (x, y_top), 4, (0, 0, 255), -1)
        cv2.circle(blended, (x, y_bottom), 4, (0, 0, 255), -1)

    # Information panel
    text_lines = []
    if model_name is not None:
        text_lines.append(f"Model: {model_name}")
    text_lines.append(f"File: {filename}")
    if dice_value is not None:
        text_lines.append(f"Dice: {dice_value:.4f}")
    if iou_value is not None:
        text_lines.append(f"IoU : {iou_value:.4f}")
    text_lines.append(f"Size: {original_w} x {original_h}")

    panel = blended.copy()
    panel_width = 270
    panel_height = 12 + 22 * len(text_lines)
    cv2.rectangle(panel, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
    blended = cv2.addWeighted(panel, 0.35, blended, 0.65, 0)

    y = 25
    for line in text_lines:
        cv2.putText(
            blended,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        y += 22

    cv2.imwrite(str(save_path), blended)


def make_panel_title(image, title):
    """Add a top title bar to an image for comparison panels."""
    h, w = image.shape[:2]
    bar_h = 34
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[bar_h:, :] = image
    cv2.putText(canvas, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def resize_keep_aspect(image, target_width=320):
    h, w = image.shape[:2]
    scale = target_width / float(w)
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)


def create_model_comparison_panels(
    all_results_df,
    overlay_dir,
    image_dir,
    mask_dir,
    mask_suffix,
    models_to_run,
    model_display_names,
    save_comparison_panels=True,
    comparison_panel_width=320
):
    """Create side-by-side panels: original, GT, U-Net, U-Net++, ResUNet, AF-BMAU-Net."""
    if not save_comparison_panels:
        return

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    overlay_dir = Path(overlay_dir)
    comparison_dir = overlay_dir / "model_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    unique_items = all_results_df[["filename", "fold"]].drop_duplicates()

    for _, item in tqdm(unique_items.iterrows(), total=len(unique_items), desc="Creating comparison panels"):
        filename = item["filename"]
        fold_id = int(item["fold"])
        stem = Path(filename).stem

        image_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            candidate = image_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            continue

        gt_path = mask_dir / f"{stem}{mask_suffix}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        original_h, original_w = image.shape[:2]

        panels = [make_panel_title(resize_keep_aspect(image_rgb, comparison_panel_width), "Input")]

        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is not None:
            gt = cv2.resize(gt, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            gt = (gt > 127).astype(np.uint8)
            gt_rgb = image_rgb.copy()
            gt_overlay = gt_rgb.copy()
            gt_overlay[gt == 1] = (0, 255, 0)
            gt_rgb = cv2.addWeighted(gt_overlay, 0.35, gt_rgb, 0.65, 0)
            gt_contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt_rgb, gt_contours, -1, (0, 255, 0), 2)
            panels.append(make_panel_title(resize_keep_aspect(gt_rgb, comparison_panel_width), "Ground Truth"))

        for model_key in models_to_run:
            overlay_path = overlay_dir / model_key / f"fold_{fold_id}" / f"{stem}_overlay.png"
            if overlay_path.exists():
                overlay_img = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
                overlay_img = resize_keep_aspect(overlay_img, comparison_panel_width)
                panels.append(make_panel_title(overlay_img, model_display_names[model_key]))

        if len(panels) < 3:
            continue

        # Align heights by padding to maximum height
        max_h = max(p.shape[0] for p in panels)
        padded = []
        for p in panels:
            if p.shape[0] < max_h:
                pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
                p = np.vstack([p, pad])
            padded.append(p)

        panel_image = np.hstack(padded)
        fold_comparison_dir = comparison_dir / f"fold_{fold_id}"
        fold_comparison_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(fold_comparison_dir / f"{stem}_comparison.png"), panel_image)
