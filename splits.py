# ============================================================
# splits.py
# Generate a fixed K-fold split of the dataset and save it to
# splits/kfold_splits.csv, so every experiment (every config file in
# configs/) trains and validates on exactly the same folds -- a fair
# comparison across models and loss functions.
#
# kfold_splits.csv has one row per image, with columns:
#   image, mask       -- filenames
#   val_fold          -- the fold this image is held out for validation on
#   fold_1 .. fold_N  -- "train" or "val" for that image, spelled out
#                        explicitly for every fold
#
# Run this once before train.py, and again whenever the dataset
# (data/images, data/masks) changes:
#
#   python splits.py                    # uses configs/default.yaml
#   python splits.py configs/other.yaml # or an explicit config
# ============================================================

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

from af_bmau_net.config import Config, CONFIGS_DIR, SPLITS_DIR, SPLITS_CSV, SPLITS_META, load_config
from af_bmau_net.utils import get_image_mask_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("splits")


def create_splits(cfg):
    image_dir = cfg.image_dir_path
    mask_dir = cfg.mask_dir_path

    image_paths, mask_paths = get_image_mask_paths(
        image_dir, mask_dir, image_prefix=cfg.image_prefix, mask_suffix=cfg.mask_suffix
    )

    num_samples = len(image_paths)
    n_splits = min(cfg.num_folds, num_samples)
    if n_splits < 2:
        raise ValueError(f"Need at least 2 samples to split; found {num_samples}.")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)

    val_fold_by_index = {}
    for fold_id, (_, val_idx) in enumerate(kfold.split(image_paths), start=1):
        for i in val_idx:
            val_fold_by_index[i] = fold_id

    # One row per image. "val_fold" is the fold this image validates on.
    # "fold_N" columns spell out train/val explicitly for every fold, so
    # you can see at a glance which images are train vs. val in each fold.
    rows = []
    for i in range(num_samples):
        row = {
            "image": Path(image_paths[i]).name,
            "mask": Path(mask_paths[i]).name,
            "val_fold": val_fold_by_index[i],
        }
        for fold_id in range(1, n_splits + 1):
            row[f"fold_{fold_id}"] = "val" if val_fold_by_index[i] == fold_id else "train"
        rows.append(row)

    splits_df = pd.DataFrame(rows).sort_values(["val_fold", "image"]).reset_index(drop=True)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(SPLITS_CSV, index=False)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_samples": num_samples,
        "num_folds": n_splits,
        "seed": cfg.seed,
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "image_prefix": cfg.image_prefix,
        "mask_suffix": cfg.mask_suffix,
    }
    with open(SPLITS_META, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved {n_splits}-fold split of {num_samples} samples to {SPLITS_CSV}")
    for fold_id in range(1, n_splits + 1):
        fold_size = int((splits_df["val_fold"] == fold_id).sum())
        logger.info(f"  fold {fold_id}: {fold_size} validation samples, {num_samples - fold_size} training samples")

    return splits_df


def main():
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = CONFIGS_DIR / "default.yaml"

    if config_path.exists():
        logger.info(f"Using dataset/seed/num_folds settings from {config_path}")
        cfg = load_config(config_path)
    else:
        logger.info(f"{config_path} not found, using built-in defaults")
        cfg = Config()

    create_splits(cfg)


if __name__ == "__main__":
    main()
