# ============================================================
# config.py
# Central configuration for the comparative segmentation project:
# U-Net, U-Net++, ResUNet, and AF-BMAU-Net
# (Amniotic Fluid Boundary-Aware Multi-Scale Attention U-Net)
#
# Every training run is driven by a YAML file in configs/ (see
# configs/default.yaml). train.py loads and runs every YAML file it
# finds there, each in its own experiment folder. Any field omitted
# from a YAML file falls back to the default below.
# ============================================================

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

# morph_kernel_size (below) is specified relative to this reference
# resolution and auto-scaled to whatever image_size is actually in use
# (see Config.effective_morph_kernel_size), so it stays proportionally
# correct if image_size changes instead of silently going stale.
MORPH_KERNEL_REFERENCE_IMAGE_SIZE = 512

# Fixed K-fold split shared by every experiment, generated once by
# splits.py, so all configs train/validate on identical folds. Not
# configurable per-YAML on purpose -- see splits.py.
SPLITS_DIR = PROJECT_ROOT / "splits"
SPLITS_CSV = SPLITS_DIR / "kfold_splits.csv"
SPLITS_META = SPLITS_DIR / "kfold_splits_meta.json"


@dataclass
class Config:
    # Name of this configuration. Defaults to the YAML filename (without
    # extension) and is used to label its experiment folder.
    name: str = "default"

    # Put ultrasound images in data/images/ and binary masks in data/masks/,
    # both relative to the project root (the folder containing train.py),
    # unless an absolute path is given.
    image_dir: str = "data/images"
    mask_dir: str = "data/masks"
    output_dir: str = "outputs"

    # Lowered from 512 to 256 to fit GPU memory; combined with use_amp this
    # should have headroom to go back to 512 later if desired.
    image_size: int = 256
    batch_size: int = 8
    epochs: int = 50

    # Number of folds to generate when running splits.py. Training itself
    # always uses whatever fold structure is saved in splits/kfold_splits.csv,
    # regardless of this value, so every experiment compares fairly.
    num_folds: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    power: float = 0.9
    threshold: float = 0.5

    # Important for Windows + Spyder. Keep 0 to avoid DataLoader multiprocessing problems.
    num_workers: int = 2

    # Dataset naming pattern.
    # Your current files are expected to be:
    # data/images/CEO001.jpg ... CEO720.jpg
    # data/masks/CEO001_mask.png ... CEO720_mask.png
    # Set image_prefix: "CEO" to only read files whose names start with CEO.
    # Set to null if you want to read all image files in data/images.
    image_prefix: Optional[str] = "CEO"
    mask_suffix: str = "_mask"

    seed: int = 42
    base_channels: int = 64
    deep_supervision: bool = True
    use_augmentation: bool = True
    use_postprocessing: bool = True
    # Mixed precision training (torch.autocast + GradScaler) to reduce GPU
    # memory usage. Only takes effect when training on CUDA; ignored on CPU.
    use_amp: bool = True
    # Morphological open/close kernel size, specified for a 512px reference
    # image. Automatically scaled for the actual image_size -- see
    # effective_morph_kernel_size below. Tune this number as if you were
    # always working at 512px.
    morph_kernel_size: int = 10

    # Loss function. One of "dice", "focal", "boundary", or "hybrid"
    # (dice + focal + boundary, weighted by lambda_dice/lambda_focal/
    # lambda_boundary below). See af_bmau_net/losses.py:create_loss().
    loss_name: str = "hybrid"

    # Hybrid loss weights (only used when loss_name is "hybrid")
    lambda_dice: float = 0.5
    lambda_focal: float = 0.3
    lambda_boundary: float = 0.2

    # Focal loss parameters (used by loss_name "focal" and "hybrid")
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Comparative experiment settings.
    # Folder-safe model keys are used for outputs.
    models_to_run: List[str] = field(
        default_factory=lambda: ["UNet", "UNetPP", "ResUNet", "AF_BMAU_Net"]
    )
    model_display_names: Dict[str, str] = field(
        default_factory=lambda: {
            "UNet": "U-Net",
            "UNetPP": "U-Net++",
            "ResUNet": "Residual U-Net",
            "AF_BMAU_Net": "AF-BMAU-Net",
        }
    )

    # If True, a side-by-side comparison panel will be generated after all models finish.
    save_comparison_panels: bool = True
    comparison_panel_width: int = 320
    show_gt_on_overlay: bool = False

    def resolve_path(self, value):
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def image_dir_path(self):
        return self.resolve_path(self.image_dir)

    @property
    def mask_dir_path(self):
        return self.resolve_path(self.mask_dir)

    @property
    def output_dir_path(self):
        return self.resolve_path(self.output_dir)

    @property
    def experiments_dir_path(self):
        return self.output_dir_path / "experiments"

    @property
    def effective_morph_kernel_size(self):
        scale = self.image_size / MORPH_KERNEL_REFERENCE_IMAGE_SIZE
        return max(1, round(self.morph_kernel_size * scale))


def load_config(yaml_path):
    """Load a Config from a YAML file, applying defaults for any omitted field."""
    yaml_path = Path(yaml_path)

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    known_fields = {f.name for f in fields(Config)}
    unknown_fields = set(data) - known_fields
    if unknown_fields:
        raise ValueError(
            f"Unknown config key(s) in {yaml_path}: {sorted(unknown_fields)}. "
            f"Valid keys are: {sorted(known_fields)}"
        )

    data.setdefault("name", yaml_path.stem)
    return Config(**data)


def find_config_files(configs_dir=CONFIGS_DIR):
    """Return every *.yaml/*.yml file in configs_dir, sorted by filename."""
    configs_dir = Path(configs_dir)
    return sorted(
        list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
    )
