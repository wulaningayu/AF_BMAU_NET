# ============================================================
# af_bmau_net.py
# AF-BMAU-Net: Amniotic Fluid Boundary-Aware Multi-Scale Attention U-Net
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    SpeckleAwareInputBlock,
    ResidualConvBlock,
    MultiScaleDilatedBottleneck,
    BoundaryAwareAttentionGate,
    FullScaleFusionDecoderBlock,
)


class AFBMAUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        deep_supervision=True
    ):
        super().__init__()

        self.deep_supervision = deep_supervision

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # A. Speckle-Aware Input Block
        self.input_block = SpeckleAwareInputBlock(
            in_channels=in_channels,
            out_channels=c1
        )

        # B. Residual Encoder
        self.enc1 = ResidualConvBlock(c1, c1)
        self.enc2 = ResidualConvBlock(c1, c2)
        self.enc3 = ResidualConvBlock(c2, c3)
        self.enc4 = ResidualConvBlock(c3, c4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # C. Multi-Scale Dilated Bottleneck
        self.bottleneck = MultiScaleDilatedBottleneck(
            in_channels=c4,
            out_channels=c4
        )

        # D. Boundary-Aware Attention Gates
        self.att1 = BoundaryAwareAttentionGate(c1, c4, c1)
        self.att2 = BoundaryAwareAttentionGate(c2, c4, c2)
        self.att3 = BoundaryAwareAttentionGate(c3, c4, c3)
        self.att4 = BoundaryAwareAttentionGate(c4, c4, c4)

        skip_channels = [c1, c2, c3, c4]

        # E. Full-Scale Skip Fusion Decoder
        self.dec4 = FullScaleFusionDecoderBlock(
            current_channels=c4,
            skip_channels_list=skip_channels,
            out_channels=c4
        )

        self.dec3 = FullScaleFusionDecoderBlock(
            current_channels=c4,
            skip_channels_list=skip_channels,
            out_channels=c3
        )

        self.dec2 = FullScaleFusionDecoderBlock(
            current_channels=c3,
            skip_channels_list=skip_channels,
            out_channels=c2
        )

        self.dec1 = FullScaleFusionDecoderBlock(
            current_channels=c2,
            skip_channels_list=skip_channels,
            out_channels=c1
        )

        # F. Deep Supervision Heads
        self.head4 = nn.Conv2d(c4, out_channels, kernel_size=1)
        self.head3 = nn.Conv2d(c3, out_channels, kernel_size=1)
        self.head2 = nn.Conv2d(c2, out_channels, kernel_size=1)
        self.head1 = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        # A
        s = self.input_block(x)

        # B
        e1 = self.enc1(s)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # C
        b = self.bottleneck(e4)

        # D
        a1, att_map1 = self.att1(e1, b)
        a2, att_map2 = self.att2(e2, b)
        a3, att_map3 = self.att3(e3, b)
        a4, att_map4 = self.att4(e4, b)

        skip_features = [a1, a2, a3, a4]
        attention_maps = [att_map1, att_map2, att_map3, att_map4]

        # E
        d4 = self.dec4(b, skip_features)

        d4_up = F.interpolate(
            d4,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )
        d3 = self.dec3(d4_up, skip_features)

        d3_up = F.interpolate(
            d3,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )
        d2 = self.dec2(d3_up, skip_features)

        d2_up = F.interpolate(
            d2,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )
        d1 = self.dec1(d2_up, skip_features)

        # F
        logit4 = self.head4(d4)
        logit3 = self.head3(d3)
        logit2 = self.head2(d2)
        logit1 = self.head1(d1)

        logits = [logit1, logit2, logit3, logit4]

        logits = [
            F.interpolate(
                logit,
                size=input_size,
                mode="bilinear",
                align_corners=False
            )
            for logit in logits
        ]

        if self.deep_supervision:
            return {
                "logits": logits,
                "attention_maps": attention_maps
            }

        return {
            "logits": logits[0],
            "attention_maps": attention_maps
        }

    def predict_proba(self, x):
        output = self.forward(x)
        logits = output["logits"]

        if isinstance(logits, list):
            probs = [torch.sigmoid(logit) for logit in logits]
            prob = torch.mean(torch.stack(probs, dim=0), dim=0)
        else:
            prob = torch.sigmoid(logits)

        return prob
