"""The :class:`SmokeUNet` segmentation architecture.

A deliberately compact encoder-decoder U-Net (~1.9M parameters) so it trains
in minutes on a laptop GPU/MPS and runs in a few milliseconds per exhaust ROI
at inference time.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SmokeUNet", "DoubleConv", "count_parameters"]


class DoubleConv(nn.Module):
    """``(conv 3x3 -> BN -> ReLU) x 2``, the standard U-Net block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.block(x)


class SmokeUNet(nn.Module):
    """Binary smoke-segmentation U-Net.

    Four encoder stages (``base``, ``2*base``, ``4*base``, ``8*base``), a
    ``16*base`` bottleneck, and four bilinear-upsampling decoder stages with
    skip concatenation.  The head is a 1x1 convolution producing a single
    **logit** channel -- apply ``sigmoid`` yourself (the training loss expects
    raw logits).

    Args:
        in_ch: Input channels (3 for BGR/RGB).
        base: Width of the first encoder stage.  ``base=16`` gives ~1.9M params.

    Shape:
        - input ``(N, in_ch, H, W)`` with ``H``/``W`` divisible by 16
          (256x256 is the trained resolution)
        - output ``(N, 1, H, W)``
    """

    def __init__(self, in_ch: int = 3, base: int = 16) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.base = int(base)

        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        c5 = base * 16

        self.enc1 = DoubleConv(in_ch, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = DoubleConv(c4, c5)

        self.dec4 = DoubleConv(c5 + c4, c4)
        self.dec3 = DoubleConv(c4 + c3, c3)
        self.dec2 = DoubleConv(c3 + c2, c2)
        self.dec1 = DoubleConv(c2 + c1, c1)

        self.head = nn.Conv2d(c1, 1, kernel_size=1)

        self._init_weights()
        self._init_head_prior()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _init_head_prior(self, positive_rate: float = 0.09) -> None:
        """Bias the output layer towards the dataset's background prior.

        Kaiming-initialising a 1x1 logit head makes the untrained network
        predict ~50% smoke everywhere, and the first several epochs are spent
        undoing that.  Starting from ``logit(positive_rate)`` (the RetinaNet
        prior-initialisation trick) means the network begins at the correct
        base rate and can spend its budget on structure instead.

        Args:
            positive_rate: Expected fraction of smoke pixels in the data.
        """
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        if self.head.bias is not None:
            prior = float(min(max(positive_rate, 1e-4), 1.0 - 1e-4))
            nn.init.constant_(self.head.bias, math.log(prior / (1.0 - prior)))

    @staticmethod
    def _up_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Bilinearly upsample *x* to *skip*'s size and concatenate on channels."""
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network.

        Args:
            x: ``(N, in_ch, H, W)`` float tensor, normalised by the caller.

        Returns:
            ``(N, 1, H, W)`` raw logits.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(self._up_cat(b, e4))
        d3 = self.dec3(self._up_cat(d4, e3))
        d2 = self.dec2(self._up_cat(d3, e2))
        d1 = self.dec1(self._up_cat(d2, e1))

        return self.head(d1)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of (trainable) parameters."""
        params: Sequence[torch.nn.Parameter] = list(self.parameters())
        return int(sum(p.numel() for p in params if p.requires_grad or not trainable_only))


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters of any module (module-level convenience helper)."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only))


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    net = SmokeUNet()
    dummy = torch.zeros(2, 3, 256, 256)
    out = net(dummy)
    print(f"SmokeUNet params={net.count_parameters():,} out={tuple(out.shape)}")
