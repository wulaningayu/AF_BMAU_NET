# ============================================================
# metrics.py
# Segmentation evaluation metrics.
# ============================================================

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_erosion as scipy_binary_erosion


def confusion_components(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(target)).sum()

    return tp, fp, fn, tn


def dice_score(pred, target, eps=1e-7):
    tp, fp, fn, _ = confusion_components(pred, target)
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def iou_score(pred, target, eps=1e-7):
    tp, fp, fn, _ = confusion_components(pred, target)
    return (tp + eps) / (tp + fp + fn + eps)


def precision_score(pred, target, eps=1e-7):
    tp, fp, _, _ = confusion_components(pred, target)
    return (tp + eps) / (tp + fp + eps)


def sensitivity_score(pred, target, eps=1e-7):
    tp, _, fn, _ = confusion_components(pred, target)
    return (tp + eps) / (tp + fn + eps)


def specificity_score(pred, target, eps=1e-7):
    _, fp, _, tn = confusion_components(pred, target)
    return (tn + eps) / (tn + fp + eps)


def hd95(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 or target.sum() == 0:
        return np.nan

    pred_border = pred ^ scipy_binary_erosion(pred)
    target_border = target ^ scipy_binary_erosion(target)

    if pred_border.sum() == 0 or target_border.sum() == 0:
        return np.nan

    dt_pred = distance_transform_edt(~pred_border)
    dt_target = distance_transform_edt(~target_border)

    dist_pred_to_target = dt_target[pred_border]
    dist_target_to_pred = dt_pred[target_border]

    distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])

    if distances.size == 0:
        return np.nan

    return float(np.percentile(distances, 95))


def compute_all_metrics(pred, target):
    return {
        "dice": dice_score(pred, target),
        "iou": iou_score(pred, target),
        "precision": precision_score(pred, target),
        "sensitivity": sensitivity_score(pred, target),
        "specificity": specificity_score(pred, target),
        "hd95": hd95(pred, target)
    }
