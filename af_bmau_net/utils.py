# ============================================================
# utils.py
# General setup utilities: directories, seeding, device, dataset pairing.
# ============================================================

import glob
import logging
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)


def create_directories(image_dir, mask_dir, experiments_dir):
    for directory in [image_dir, mask_dir, experiments_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentPaths:
    experiment_dir: Path
    checkpoint_dir: Path
    prediction_dir: Path
    overlay_dir: Path
    result_dir: Path
    log_dir: Path


def create_experiment_paths(experiments_dir, run_name=None):
    """
    Create a distinct, timestamped folder for a single run of the pipeline,
    so that checkpoints/predictions/overlays/results from different
    experiments (and different config files) never mix or overwrite
    each other.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]
    folder_name = f"run_{run_name}_{timestamp}_{unique_suffix}" if run_name else f"run_{timestamp}_{unique_suffix}"
    experiment_dir = Path(experiments_dir) / folder_name

    paths = ExperimentPaths(
        experiment_dir=experiment_dir,
        checkpoint_dir=experiment_dir / "checkpoints",
        prediction_dir=experiment_dir / "predictions",
        overlay_dir=experiment_dir / "overlays",
        result_dir=experiment_dir / "results",
        log_dir=experiment_dir / "logs",
    )

    for directory in [
        paths.experiment_dir,
        paths.checkpoint_dir,
        paths.prediction_dir,
        paths.overlay_dir,
        paths.result_dir,
        paths.log_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    return paths


def save_config_snapshot(cfg, experiment_dir):
    """Dump the exact config used for this run, so a past experiment's
    settings can always be recovered from its own folder."""
    with open(Path(experiment_dir) / "config_used.yaml", "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    return device


def get_image_mask_paths(image_dir, mask_dir, image_prefix=None, mask_suffix="_mask"):
    """
    Pair images and masks using the following convention:

        data/images/CEO001.jpg       -> data/masks/CEO001_mask.png
        data/images/CEO002.jpg       -> data/masks/CEO002_mask.png
        ...

    The image extension can be .jpg, .jpeg, .png, .bmp, .tif, or .tiff.
    The mask is preferably .png, but this function also searches other common
    image extensions if the .png mask is not found.
    """

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    mask_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(str(image_dir / ext)))

    image_paths = sorted(image_paths)

    # Optional filtering for the current dataset naming pattern CEO001 ... CEO720
    if image_prefix is not None:
        image_paths = [
            path for path in image_paths
            if Path(path).stem.startswith(image_prefix)
        ]

    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"No images found in {image_dir}. "
            "For this project, put images such as CEO001.jpg in data/images/."
        )

    mask_paths = []
    missing_masks = []

    for image_path in image_paths:
        image_stem = Path(image_path).stem

        # Primary expected format: CEO001_mask.png
        primary_mask_path = mask_dir / f"{image_stem}{mask_suffix}.png"

        if primary_mask_path.exists():
            mask_paths.append(str(primary_mask_path))
            continue

        # Fallback: search other common extensions, e.g., CEO001_mask.jpg
        found_mask = None
        for ext in mask_extensions:
            candidate = mask_dir / f"{image_stem}{mask_suffix}{ext}"
            if candidate.exists():
                found_mask = candidate
                break

        if found_mask is None:
            missing_masks.append(str(primary_mask_path))
            mask_paths.append(str(primary_mask_path))
        else:
            mask_paths.append(str(found_mask))

    if len(missing_masks) > 0:
        logger.error("Missing masks. First 10 missing expected files:")
        for item in missing_masks[:10]:
            logger.error(item)

        raise FileNotFoundError(
            "Some mask files were not found. "
            "For CEO001.jpg, the mask must be named CEO001_mask.png "
            "and placed in data/masks/."
        )

    logger.info("Sample image-mask pairs:")
    for image_path, mask_path in list(zip(image_paths, mask_paths))[:5]:
        logger.info(f"  {Path(image_path).name}  ->  {Path(mask_path).name}")

    return image_paths, mask_paths


def load_kfold_splits(splits_csv, image_paths, mask_paths):
    """
    Load the fixed K-fold split produced by splits.py and translate it into
    (train_idx, val_idx, fold_id) index tuples aligned with the given
    image_paths, matched by filename. This is what makes every experiment
    train/validate on exactly the same folds, regardless of run order or
    how image_paths happened to be gathered.
    """
    splits_csv = Path(splits_csv)
    if not splits_csv.exists():
        raise FileNotFoundError(
            f"No split file found at {splits_csv}. "
            "Run `python splits.py` once to generate a fixed K-fold split "
            "of the dataset before training, so every experiment compares fairly."
        )

    splits_df = pd.read_csv(splits_csv)
    fold_by_image = dict(zip(splits_df["image"], splits_df["val_fold"]))

    image_names = [Path(p).name for p in image_paths]
    missing = sorted(set(image_names) - set(fold_by_image))
    if missing:
        raise ValueError(
            f"{len(missing)} image(s) in the current dataset are missing from {splits_csv} "
            f"(e.g. {missing[:5]}). Regenerate it with `python splits.py` so the split "
            "matches the current dataset."
        )

    fold_ids = sorted(set(fold_by_image.values()))
    fold_splits = []
    for fold_id in fold_ids:
        val_idx = np.array([i for i, name in enumerate(image_names) if fold_by_image[name] == fold_id])
        train_idx = np.array([i for i, name in enumerate(image_names) if fold_by_image[name] != fold_id])
        fold_splits.append((train_idx, val_idx, fold_id))

    return fold_splits
