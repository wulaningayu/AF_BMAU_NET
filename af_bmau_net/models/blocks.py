# ============================================================
# blocks.py
# Shared building blocks used across the comparative models.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeckleAwareInputBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.depthwise = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        identity = self.proj(x)

        out = self.conv(x)
        out = self.depthwise(out)
        out = self.bn(out)
        out = self.relu(out)

        out = out + identity
        return out


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.proj = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        identity = self.proj(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity
        out = self.relu(out)

        return out


class MultiScaleDilatedBottleneck(nn.Module):
    def __init__(self, in_channels=512, out_channels=512):
        super().__init__()

        branch_channels = out_channels // 4

        self.branch1 = self._make_branch(in_channels, branch_channels, dilation=1)
        self.branch2 = self._make_branch(in_channels, branch_channels, dilation=2)
        self.branch4 = self._make_branch(in_channels, branch_channels, dilation=4)
        self.branch8 = self._make_branch(in_channels, branch_channels, dilation=8)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_branch(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b4 = self.branch4(x)
        b8 = self.branch8(x)

        out = torch.cat([b1, b2, b4, b8], dim=1)
        out = self.fuse(out)

        return out


class BoundaryAwareAttentionGate(nn.Module):
    def __init__(self, encoder_channels, gating_channels, inter_channels):
        super().__init__()

        self.theta_x = nn.Conv2d(
            encoder_channels,
            inter_channels,
            kernel_size=1,
            bias=False
        )

        self.phi_g = nn.Conv2d(
            gating_channels,
            inter_channels,
            kernel_size=1,
            bias=False
        )

        self.edge_branch = nn.Sequential(
            nn.Conv2d(
                encoder_channels,
                inter_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inter_channels,
                inter_channels,
                kernel_size=1,
                bias=False
            )
        )

        self.psi = nn.Conv2d(
            inter_channels,
            1,
            kernel_size=1,
            bias=True
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_feature, gating_feature):
        if gating_feature.shape[2:] != encoder_feature.shape[2:]:
            gating_feature = F.interpolate(
                gating_feature,
                size=encoder_feature.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        theta = self.theta_x(encoder_feature)
        phi = self.phi_g(gating_feature)
        edge = self.edge_branch(encoder_feature)

        attention = self.relu(theta + phi + edge)
        attention = self.sigmoid(self.psi(attention))

        filtered_feature = encoder_feature * attention

        return filtered_feature, attention


class FullScaleFusionDecoderBlock(nn.Module):
    def __init__(self, current_channels, skip_channels_list, out_channels):
        super().__init__()

        self.current_proj = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.skip_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for ch in skip_channels_list
        ])

        total_channels = out_channels * (1 + len(skip_channels_list))
        self.fuse = ResidualConvBlock(total_channels, out_channels)

    def forward(self, current_feature, skip_features):
        target_size = current_feature.shape[2:]
        current_feature = self.current_proj(current_feature)

        resized_skips = []

        for feature, proj in zip(skip_features, self.skip_projs):
            if feature.shape[2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False
                )

            resized_skips.append(proj(feature))

        fused = torch.cat([current_feature] + resized_skips, dim=1)
        out = self.fuse(fused)

        return out


class ConvBlock(nn.Module):
    """Two-layer convolutional block used in U-Net and U-Net++."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
