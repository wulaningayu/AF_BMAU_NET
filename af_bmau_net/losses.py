# ============================================================
# losses.py
# Loss functions: Dice, Focal, Boundary, and weighted combinations of
# them. The active loss is selected and weighted from the config YAML
# (cfg.loss_name) via create_loss().
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)

        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        focal = alpha_t * ((1.0 - pt) ** self.gamma) * bce

        return focal.mean()


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def gradient_map(self, x):
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))

        return dx + dy

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        pred_edge = self.gradient_map(probs)
        target_edge = self.gradient_map(targets)

        return F.l1_loss(pred_edge, target_edge)


class CombinedLoss(nn.Module):
    """Weighted sum of named per-scale loss components, e.g. dice + focal + boundary."""

    def __init__(self, weighted_components):
        super().__init__()
        self.component_modules = nn.ModuleList([module for _, module, _ in weighted_components])
        self.weights = [weight for _, _, weight in weighted_components]

    def forward(self, logits, targets):
        total = 0.0
        for module, weight in zip(self.component_modules, self.weights):
            total = total + weight * module(logits, targets)
        return total


class DeepSupervisionLoss(nn.Module):
    """
    Wraps a per-scale loss module so it can consume a model's raw output
    dict ({"logits": ...}), where logits is either a single tensor or a
    list of tensors at multiple scales (deep supervision).
    """

    def __init__(self, base_loss, scale_weights=(0.4, 0.25, 0.20, 0.15)):
        super().__init__()
        self.base_loss = base_loss
        self.scale_weights = list(scale_weights)

    def forward(self, outputs, targets):
        logits = outputs["logits"]

        if not isinstance(logits, list):
            return self.base_loss(logits, targets)

        # More weight for the highest-resolution/final output.
        weights = self.scale_weights[:len(logits)]
        total_loss = 0.0

        for weight, logit in zip(weights, logits):
            if logit.shape[2:] != targets.shape[2:]:
                logit = F.interpolate(
                    logit,
                    size=targets.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
            total_loss = total_loss + weight * self.base_loss(logit, targets)

        return total_loss


VALID_LOSS_NAMES = ("dice", "focal", "boundary", "hybrid")

_SINGLE_LOSS_BUILDERS = {
    "dice": lambda cfg: DiceLoss(),
    "focal": lambda cfg: FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma),
    "boundary": lambda cfg: BoundaryLoss(),
}


def create_loss(cfg):
    """
    Build the training loss from cfg.loss_name:
      - "dice", "focal", or "boundary": that single component alone.
      - "hybrid": dice + focal + boundary, weighted by cfg.lambda_dice /
        cfg.lambda_focal / cfg.lambda_boundary.
    The result always accepts a model's raw {"logits": ...} output and
    transparently handles both single-scale and deep-supervised
    (list of logits) models.
    """
    if cfg.loss_name == "hybrid":
        base_loss = CombinedLoss([
            ("dice", DiceLoss(), cfg.lambda_dice),
            ("focal", FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma), cfg.lambda_focal),
            ("boundary", BoundaryLoss(), cfg.lambda_boundary),
        ])
    elif cfg.loss_name in _SINGLE_LOSS_BUILDERS:
        base_loss = _SINGLE_LOSS_BUILDERS[cfg.loss_name](cfg)
    else:
        raise ValueError(
            f"Unknown loss_name '{cfg.loss_name}'. Valid options: {list(VALID_LOSS_NAMES)}"
        )

    return DeepSupervisionLoss(base_loss)
