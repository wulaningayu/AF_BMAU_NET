# AF-BMAU-Net

AF-BMAU-Net (**Amniotic Fluid Boundary-Aware Multi-Scale Attention U-Net**) is a PyTorch project for binary segmentation of amniotic-fluid regions in grayscale ultrasound images. It provides a complete K-fold cross-validation pipeline for training and comparing AF-BMAU-Net with three segmentation baselines:

- U-Net
- U-Net++
- Residual U-Net (ResUNet)
- AF-BMAU-Net

The proposed model combines a speckle-aware input block, residual encoding, a multi-scale dilated bottleneck, boundary-aware attention, full-scale skip fusion, and deep supervision. The pipeline supports Dice, focal, boundary, and weighted hybrid losses, as well as augmentation, mixed-precision training, morphological post-processing, metric reporting, prediction overlays, and model-comparison panels.

## Project structure

```text
AF_BMAU_NET/
|-- af_bmau_net/            # Dataset, models, losses, training, and evaluation code
|   `-- models/             # AF-BMAU-Net and baseline architectures
|-- configs/                # Experiment YAML files loaded by train.py
|-- data/
|   |-- images/             # Ultrasound images (not tracked by Git)
|   `-- masks/              # Binary segmentation masks (not tracked by Git)
|-- splits/                 # Fixed K-fold assignments
|-- outputs/                # Checkpoints and experiment results (not tracked by Git)
|-- splits.py               # Generates reproducible K-fold splits
`-- train.py                # Runs all experiment configurations
```

## Requirements

- Python 3.9 or newer
- A CUDA-capable GPU is recommended; CPU execution is supported but will be much slower
- PyTorch with the appropriate CUDA build if GPU training is required

Install the Python dependencies in a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install PyTorch using the command recommended for your platform at [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining packages:

```bash
pip install numpy pandas pyyaml opencv-python albumentations scikit-learn matplotlib tqdm
```

For a CPU-only setup, PyTorch can normally be installed with:

```bash
pip install torch
```

## Dataset preparation

Place grayscale ultrasound images in `data/images/` and their binary masks in `data/masks/`. Each mask must have the same base name as its image followed by `_mask`.

Example:

```text
data/
|-- images/
|   |-- CEO001.jpg
|   `-- CEO002.jpg
`-- masks/
    |-- CEO001_mask.png
    `-- CEO002_mask.png
```

Supported image and mask formats are PNG, JPG/JPEG, BMP, TIF, and TIFF. By default, only image names beginning with `CEO` are included. Change `image_prefix` in a YAML configuration to another prefix, or set it to `null` to include every supported image in the directory.

Masks are converted to binary values using a pixel threshold of 127.

## Running the project

### 1. Select or create an experiment configuration

Experiment files are stored in `configs/`. The supplied configurations cover every combination of the four models and four loss functions. Important settings include:

```yaml
name: af_bmau_net_hybrid
image_size: 256
batch_size: 8
epochs: 25
num_folds: 5
models_to_run:
  - AF_BMAU_Net
loss_name: hybrid
```

Valid model keys are `UNet`, `UNetPP`, `ResUNet`, and `AF_BMAU_Net`. Valid loss names are `dice`, `focal`, `boundary`, and `hybrid`. Omitted settings use the defaults defined in `af_bmau_net/config.py`.

### 2. Generate fixed K-fold splits

Run this once before training, and run it again whenever the dataset changes:

```bash
python splits.py configs/af_bmau_net_hybrid.yaml
```

This creates `splits/kfold_splits.csv` and `splits/kfold_splits_meta.json`. The fixed assignments ensure that all model and loss experiments use the same validation folds.

### 3. Start training

```bash
python train.py
```

**Important:** `train.py` automatically runs every `.yaml` and `.yml` file currently present in `configs/`, in filename order. Keep only the configurations you want to execute in that directory when starting a run. The supplied set represents 16 separate experiments and may take considerable time.

For a quick smoke test, create a YAML file in `configs/` with reduced settings such as:

```yaml
name: debug
image_size: 256
batch_size: 1
epochs: 1
num_folds: 2
models_to_run:
  - UNet
loss_name: dice
```

Generate the split using that file, then run `python train.py`. Remember that every other YAML file left in `configs/` will also be executed.

On Windows, set `num_workers: 0` in the configuration if DataLoader multiprocessing causes problems or training is launched from an IDE.

## Outputs

Each configuration creates an independent timestamped directory under:

```text
outputs/experiments/run_<config-name>_<timestamp>_<id>/
```

It contains:

- `config_used.yaml`: snapshot of the exact experiment settings
- `checkpoints/`: best model checkpoint for each fold
- `predictions/`: predicted binary masks
- `overlays/`: predictions drawn over the original ultrasound images
- `results/`: fold metrics, cross-validation summaries, comparison tables, and loss curves
- `logs/`: experiment logs

Evaluation includes segmentation metrics such as Dice score, IoU, precision, recall, and related statistics. The pipeline also measures the deepest vertical pocket from predictions at the original image resolution.

## Reproducibility notes

- K-fold assignments are saved and reused across experiments.
- Random seeds are controlled through the `seed` configuration value.
- CUDA mixed precision is enabled by `use_amp: true` and is ignored automatically on CPU.
- Training selects CUDA when available and otherwise falls back to CPU.
- Dataset contents and generated outputs are excluded from version control.

## Configuration reference

All supported configuration fields and their defaults are documented in `af_bmau_net/config.py`. The root-level `config_default.yaml` is a full configuration example; files used by `train.py` must be placed inside `configs/`.
