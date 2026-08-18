# ============================================================
# factory.py
# Factory function for all comparative models.
# ============================================================

from .af_bmau_net import AFBMAUNet
from .baselines import UNetBaseline, ResidualUNetBaseline, UNetPlusPlusBaseline

VALID_MODEL_KEYS = ("UNet", "UNetPP", "ResUNet", "AF_BMAU_Net")


def create_model(model_key, in_channels=1, out_channels=1, base_channels=64, deep_supervision=True):
    """Factory function for all comparative models."""
    if model_key == "UNet":
        return UNetBaseline(in_channels, out_channels, base_channels)
    if model_key == "UNetPP":
        return UNetPlusPlusBaseline(in_channels, out_channels, base_channels)
    if model_key == "ResUNet":
        return ResidualUNetBaseline(in_channels, out_channels, base_channels)
    if model_key == "AF_BMAU_Net":
        return AFBMAUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            deep_supervision=deep_supervision
        )
    raise ValueError(f"Unknown model_key: {model_key}. Valid options: {list(VALID_MODEL_KEYS)}")
