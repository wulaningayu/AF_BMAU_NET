# ============================================================
# postprocessing.py
# Mask post-processing and deepest-vertical-pocket measurement.
# ============================================================

import cv2
import numpy as np


def threshold_mask(prob_map, threshold=0.5):
    return (prob_map >= threshold).astype(np.uint8)


def morphology_postprocess(mask, kernel_size=10):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8
    )

    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)
    largest = (labels == largest_label).astype(np.uint8)

    return largest


def postprocess_prediction(prob_map, threshold=0.5, kernel_size=10, use_postprocessing=True):
    mask = threshold_mask(prob_map, threshold)

    if use_postprocessing:
        mask = morphology_postprocess(mask, kernel_size)
        mask = keep_largest_component(mask)

    return mask


def compute_deepest_vertical_pocket(mask):
    h, w = mask.shape

    max_length_px = 0
    best_column = None
    best_top = None
    best_bottom = None

    for x in range(w):
        ys = np.where(mask[:, x] > 0)[0]

        if len(ys) == 0:
            continue

        y_top = int(ys.min())
        y_bottom = int(ys.max())
        length = int(y_bottom - y_top + 1)

        if length > max_length_px:
            max_length_px = length
            best_column = int(x)
            best_top = y_top
            best_bottom = y_bottom

    return {
        "length_px": max_length_px,
        "x": best_column,
        "y_top": best_top,
        "y_bottom": best_bottom
    }


def convert_px_to_mm(length_px, mm_per_pixel):
    return length_px * mm_per_pixel
