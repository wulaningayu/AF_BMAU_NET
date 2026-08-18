# ============================================================
# pipeline.py
# Main cross-validation pipeline tying together data, models,
# training, evaluation, and reporting for all comparative models.
# ============================================================

import logging
import warnings

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import PROJECT_ROOT, SPLITS_CSV
from .utils import (
    create_directories,
    create_experiment_paths,
    save_config_snapshot,
    set_seed,
    get_device,
    get_image_mask_paths,
    load_kfold_splits,
)
from .logger import setup_logger
from .dataset import AmnioticFluidDataset, get_train_transform, ALBUMENTATIONS_AVAILABLE
from .models import create_model, VALID_MODEL_KEYS
from .losses import create_loss, VALID_LOSS_NAMES
from .training import fit
from .evaluation import (
    evaluate_model,
    print_summary,
    summarize_all_models,
    plot_loss_curves,
)
from .visualization import create_model_comparison_panels

logger = logging.getLogger(__name__)


def validate_cfg(cfg):
    unknown_models = [m for m in cfg.models_to_run if m not in VALID_MODEL_KEYS]
    if unknown_models:
        raise ValueError(
            f"Unknown model_key(s) in models_to_run: {unknown_models}. "
            f"Valid options: {list(VALID_MODEL_KEYS)}"
        )

    missing_display_names = [m for m in cfg.models_to_run if m not in cfg.model_display_names]
    if missing_display_names:
        raise ValueError(f"model_display_names is missing entries for: {missing_display_names}")

    if cfg.loss_name not in VALID_LOSS_NAMES:
        raise ValueError(
            f"Unknown loss_name '{cfg.loss_name}'. Valid options: {list(VALID_LOSS_NAMES)}"
        )


def run_cross_validation(cfg):
    warnings.filterwarnings("ignore")
    validate_cfg(cfg)

    image_dir = cfg.image_dir_path
    mask_dir = cfg.mask_dir_path
    experiments_dir = cfg.experiments_dir_path

    create_directories(image_dir, mask_dir, experiments_dir)
    paths = create_experiment_paths(experiments_dir, run_name=cfg.name)
    setup_logger(paths.log_dir)
    save_config_snapshot(cfg, paths.experiment_dir)
    set_seed(cfg.seed)

    logger.info(f"Experiment folder: {paths.experiment_dir}")
    logger.info(f"Config name: {cfg.name}")
    logger.info("Comparative Segmentation Project")
    logger.info("Models: " + ", ".join([cfg.model_display_names[m] for m in cfg.models_to_run]))
    logger.info(f"Project root  : {PROJECT_ROOT}")
    logger.info(f"Image folder  : {image_dir}")
    logger.info(f"Mask folder   : {mask_dir}")
    logger.info(f"Image size    : {cfg.image_size}")
    logger.info(f"Batch size    : {cfg.batch_size}")
    logger.info(f"Epochs        : {cfg.epochs}")
    logger.info(f"Splits file   : {SPLITS_CSV}")
    logger.info(f"Loss function : {cfg.loss_name}")
    logger.info(f"Augmentation  : {cfg.use_augmentation and ALBUMENTATIONS_AVAILABLE}")
    logger.info(
        f"Postprocess   : {cfg.use_postprocessing} "
        f"(morph kernel {cfg.morph_kernel_size} @512px -> {cfg.effective_morph_kernel_size} @{cfg.image_size}px)"
    )

    device = get_device()
    use_amp = cfg.use_amp and device.type == "cuda"
    logger.info(f"Mixed precision (AMP): {use_amp}")

    image_paths, mask_paths = get_image_mask_paths(
        image_dir, mask_dir, image_prefix=cfg.image_prefix, mask_suffix=cfg.mask_suffix
    )

    num_samples = len(image_paths)
    logger.info(f"Total image-mask pairs: {num_samples}")
    if num_samples < 2:
        raise ValueError("At least 2 image-mask pairs are needed for training/validation.")

    fold_splits = load_kfold_splits(SPLITS_CSV, image_paths, mask_paths)
    n_splits = len(fold_splits)
    logger.info(f"Loaded fixed {n_splits}-fold split from {SPLITS_CSV}")

    all_model_results = []
    all_histories = []

    for model_key in cfg.models_to_run:
        model_display_name = cfg.model_display_names[model_key]
        logger.info("\n" + "#" * 90 + f"\nTraining and evaluating model: {model_display_name}\n" + "#" * 90)

        model_results = []

        for train_idx, val_idx, fold_id in fold_splits:
            logger.info("\n" + "=" * 80 + f"\n{model_display_name} | Fold {fold_id}/{n_splits}\n" + "=" * 80)

            set_seed(cfg.seed + fold_id)

            train_images = [image_paths[i] for i in train_idx]
            train_masks = [mask_paths[i] for i in train_idx]
            val_images = [image_paths[i] for i in val_idx]
            val_masks = [mask_paths[i] for i in val_idx]

            logger.info(f"Training samples   : {len(train_images)}")
            logger.info(f"Validation samples : {len(val_images)}")

            train_dataset = AmnioticFluidDataset(
                train_images, train_masks, image_size=cfg.image_size,
                transform=get_train_transform(cfg.use_augmentation)
            )
            val_dataset = AmnioticFluidDataset(val_images, val_masks, image_size=cfg.image_size, transform=None)

            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available()
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available()
            )

            model = create_model(
                model_key,
                in_channels=1,
                out_channels=1,
                base_channels=cfg.base_channels,
                deep_supervision=cfg.deep_supervision
            ).to(device)

            criterion = create_loss(cfg)

            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

            fold_checkpoint_dir = paths.checkpoint_dir / model_key / f"fold_{fold_id}"
            fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = fold_checkpoint_dir / "best_model.pth"

            history_df = fit(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epochs=cfg.epochs,
                init_lr=cfg.learning_rate,
                checkpoint_path=checkpoint_path,
                model_key=model_key,
                model_display_name=model_display_name,
                fold_id=fold_id,
                power=cfg.power,
                threshold=cfg.threshold,
                kernel_size=cfg.effective_morph_kernel_size,
                use_postprocessing=cfg.use_postprocessing,
                use_amp=use_amp
            )

            model_result_dir = paths.result_dir / model_key
            model_result_dir.mkdir(parents=True, exist_ok=True)
            history_path = model_result_dir / f"training_history_fold_{fold_id}.csv"
            history_df.to_csv(history_path, index=False)
            all_histories.append(history_df)

            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

            fold_df = evaluate_model(
                model=model,
                loader=val_loader,
                device=device,
                fold_id=fold_id,
                model_key=model_key,
                model_display_name=model_display_name,
                prediction_dir=paths.prediction_dir,
                overlay_dir=paths.overlay_dir,
                threshold=cfg.threshold,
                kernel_size=cfg.effective_morph_kernel_size,
                use_postprocessing=cfg.use_postprocessing,
                show_gt_on_overlay=cfg.show_gt_on_overlay,
                save_predictions=True,
                save_overlays=True
            )

            fold_result_path = model_result_dir / f"results_fold_{fold_id}.csv"
            fold_df.to_csv(fold_result_path, index=False)
            model_results.append(fold_df)
            all_model_results.append(fold_df)

            print_summary(fold_df, f"{model_display_name} Fold {fold_id} Summary")

            # Free GPU memory before the next fold/model.
            del model, optimizer
            torch.cuda.empty_cache()

        model_final_df = pd.concat(model_results, axis=0)
        model_final_path = paths.result_dir / model_key / "cross_validation_results.csv"
        model_final_df.to_csv(model_final_path, index=False)
        model_summary_df = print_summary(model_final_df, f"{model_display_name} Final Cross-Validation Summary")
        model_summary_df.to_csv(paths.result_dir / model_key / "cross_validation_summary.csv")

    final_df = pd.concat(all_model_results, axis=0)
    final_result_path = paths.result_dir / "all_models_cross_validation_results.csv"
    final_df.to_csv(final_result_path, index=False)

    comparison_summary_df = summarize_all_models(final_df, cfg.models_to_run, cfg.model_display_names)
    comparison_summary_path = paths.result_dir / "model_comparison_summary.csv"
    comparison_summary_df.to_csv(comparison_summary_path, index=False)

    if len(all_histories) > 0:
        history_all_df = pd.concat(all_histories, axis=0)
        history_all_path = paths.result_dir / "all_models_training_history.csv"
        history_all_df.to_csv(history_all_path, index=False)
        plot_loss_curves(history_all_df, paths.result_dir, cfg.models_to_run, cfg.model_display_names)

    create_model_comparison_panels(
        final_df,
        paths.overlay_dir,
        image_dir,
        mask_dir,
        cfg.mask_suffix,
        cfg.models_to_run,
        cfg.model_display_names,
        save_comparison_panels=cfg.save_comparison_panels,
        comparison_panel_width=cfg.comparison_panel_width
    )

    logger.info("Experiment completed.")
    logger.info(f"Experiment folder          : {paths.experiment_dir}")
    logger.info(f"All model detailed results : {final_result_path}")
    logger.info(f"Model comparison summary   : {comparison_summary_path}")
    logger.info(f"Loss curves folder         : {paths.result_dir / 'loss_curves'}")
    logger.info(f"Predicted masks folder     : {paths.prediction_dir}")
    logger.info(f"Overlays folder            : {paths.overlay_dir}")

    return paths
