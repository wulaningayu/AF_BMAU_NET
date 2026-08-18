# ============================================================
# baselines.py
# Comparative baseline models: U-Net, Residual U-Net, U-Net++.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, ResidualConvBlock


class UNetBaseline(nn.Module):
    """Standard U-Net baseline for binary segmentation."""
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, c5)

        self.dec4 = ConvBlock(c5 + c4, c4)
        self.dec3 = ConvBlock(c4 + c3, c3)
        self.dec2 = ConvBlock(c3 + c2, c2)
        self.dec1 = ConvBlock(c2 + c1, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def _up(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self._up(b, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))

        return {"logits": self.head(d1), "attention_maps": None}

    def predict_proba(self, x):
        output = self.forward(x)
        return torch.sigmoid(output["logits"])


class ResidualUNetBaseline(nn.Module):
    """Residual U-Net baseline using residual convolutional blocks."""
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = ResidualConvBlock(in_channels, c1)
        self.enc2 = ResidualConvBlock(c1, c2)
        self.enc3 = ResidualConvBlock(c2, c3)
        self.enc4 = ResidualConvBlock(c3, c4)
        self.bottleneck = ResidualConvBlock(c4, c5)

        self.dec4 = ResidualConvBlock(c5 + c4, c4)
        self.dec3 = ResidualConvBlock(c4 + c3, c3)
        self.dec2 = ResidualConvBlock(c3 + c2, c2)
        self.dec1 = ResidualConvBlock(c2 + c1, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def _up(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self._up(b, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))

        return {"logits": self.head(d1), "attention_maps": None}

    def predict_proba(self, x):
        output = self.forward(x)
        return torch.sigmoid(output["logits"])


class UNetPlusPlusBaseline(nn.Module):
    """U-Net++ baseline with nested dense skip pathways."""
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = ConvBlock(in_channels, c1)
        self.conv1_0 = ConvBlock(c1, c2)
        self.conv2_0 = ConvBlock(c2, c3)
        self.conv3_0 = ConvBlock(c3, c4)
        self.conv4_0 = ConvBlock(c4, c5)

        self.conv0_1 = ConvBlock(c1 + c2, c1)
        self.conv1_1 = ConvBlock(c2 + c3, c2)
        self.conv2_1 = ConvBlock(c3 + c4, c3)
        self.conv3_1 = ConvBlock(c4 + c5, c4)

        self.conv0_2 = ConvBlock(c1 * 2 + c2, c1)
        self.conv1_2 = ConvBlock(c2 * 2 + c3, c2)
        self.conv2_2 = ConvBlock(c3 * 2 + c4, c3)

        self.conv0_3 = ConvBlock(c1 * 3 + c2, c1)
        self.conv1_3 = ConvBlock(c2 * 3 + c3, c2)

        self.conv0_4 = ConvBlock(c1 * 4 + c2, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def _up(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1))

        return {"logits": self.head(x0_4), "attention_maps": None}

    def predict_proba(self, x):
        output = self.forward(x)
        return torch.sigmoid(output["logits"])
