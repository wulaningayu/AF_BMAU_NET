# ============================================================
# train.py
# Entry point for the comparative segmentation project:
# U-Net, U-Net++, ResUNet, and AF-BMAU-Net
# (Amniotic Fluid Boundary-Aware Multi-Scale Attention U-Net)
#
# 1. Put ultrasound images in data/images/
# 2. Put binary masks in data/masks/
# 3. Run splits.py once to generate a fixed K-fold split (splits/kfold_splits.csv)
#    shared by every experiment, so all configs compare fairly.
# 4. Add one or more YAML files to configs/ (see configs/default.yaml).
# 5. Run this file. Every YAML file in configs/ is run as its own
#    experiment, saved in its own timestamped folder under
#    outputs/experiments/.
#
# See af_bmau_net/config.py for all configuration options.
# ============================================================

import logging
import sys

from af_bmau_net import run_cross_validation, load_config, find_config_files
from af_bmau_net.config import CONFIGS_DIR, SPLITS_CSV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("train")


def main():
    if not SPLITS_CSV.exists():
        logger.error(
            f"No split file found at {SPLITS_CSV}. "
            "Run `python splits.py` once to generate a fixed K-fold split "
            "before training, so every experiment compares fairly."
        )
        sys.exit(1)

    config_files = find_config_files(CONFIGS_DIR)

    if not config_files:
        logger.error(
            f"No YAML config files found in {CONFIGS_DIR}. "
            "Add one (see configs/default.yaml) and rerun."
        )
        sys.exit(1)

    logger.info(f"Found {len(config_files)} config file(s) in {CONFIGS_DIR}:")
    for path in config_files:
        logger.info(f"  - {path.name}")

    failed = []
    for path in config_files:
        logger.info("\n" + "*" * 90 + f"\nRunning config: {path.name}\n" + "*" * 90)
        try:
            cfg = load_config(path)
            run_cross_validation(cfg)
        except Exception:
            logger.exception(f"Config {path.name} failed, continuing with remaining configs")
            failed.append(path.name)

    if failed:
        logger.error(f"{len(failed)}/{len(config_files)} config(s) failed: {failed}")
        sys.exit(1)

    logger.info("All configs completed successfully.")


if __name__ == "__main__":
    main()
