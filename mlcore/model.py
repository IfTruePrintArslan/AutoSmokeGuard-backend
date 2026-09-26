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

__all__ = ["SmokeUNet", "ResNetUNet", "DoubleConv", "count_parameters", "build_arch"]


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


# --------------------------------------------------------------------------- #
# ResNet-encoder U-Net
# --------------------------------------------------------------------------- #
#: Encoder channel widths per stage, for the ResNet variants supported here.
_RESNET_STAGES = {
    "resnet18": (64, 64, 128, 256, 512),
    "resnet34": (64, 64, 128, 256, 512),
}


class _DecoderBlock(nn.Module):
    """Upsample to the skip's size, concatenate, then two 3x3 convs."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        size = skip.shape[-2:] if skip is not None else [s * 2 for s in x.shape[-2:]]
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResNetUNet(nn.Module):
    """U-Net with a torchvision ResNet encoder.

    Why this architecture is in the product
    ---------------------------------------
    :class:`SmokeUNet` learns its encoder from the smoke corpora alone.  With
    150 annotated surveillance frames that is not enough to learn general
    visual features, and it shows: the published PoVSSeg methods this project
    is measured against (PSP-Net, FCN-32s, PAN, DB-Net, DeepLab-v3) all use
    ImageNet-pretrained backbones, and all of them beat a from-scratch U-Net.
    This class closes that gap by starting from ImageNet features.

    **No new dependency.**  ``torchvision`` is already a pinned backend
    requirement, so the encoder comes from ``torchvision.models`` rather than
    from ``segmentation_models_pytorch``/``timm`` -- which would have added
    three packages, for four Python/OS combinations in CI, to obtain the same
    ResNet weights.

    **Nothing is downloaded at inference.**  ``pretrained`` is a *training*
    switch.  A shipped checkpoint restores every encoder weight from its own
    ``state_dict``, so :class:`~mlcore.segmenter.SmokeSegmenter` constructs
    this with ``pretrained=False`` and never touches the network or the torch
    hub cache.

    Args:
        encoder: ``"resnet18"`` or ``"resnet34"``.
        pretrained: Load ImageNet weights into the encoder (training only).
        decoder_ch: Decoder widths, deepest first.

    Shape:
        - input ``(N, 3, H, W)``, ``H`` and ``W`` multiples of 32
        - output ``(N, 1, H, W)`` raw logits
    """

    def __init__(
        self,
        encoder: str = "resnet34",
        pretrained: bool = False,
        decoder_ch: Sequence[int] = (256, 128, 64, 32, 16),
    ) -> None:
        super().__init__()
        if encoder not in _RESNET_STAGES:
            raise ValueError(
                f"unsupported encoder {encoder!r}; known: {sorted(_RESNET_STAGES)}")
        self.encoder_name = str(encoder)
        self.decoder_ch = tuple(int(c) for c in decoder_ch)

        import torchvision.models as tvm

        weights = None
        if pretrained:
            weights = {
                "resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
                "resnet34": tvm.ResNet34_Weights.IMAGENET1K_V1,
            }[encoder]
        net = getattr(tvm, encoder)(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)   # /2   64
        self.pool = net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2         # /4 64, /8 128
        self.layer3, self.layer4 = net.layer3, net.layer4         # /16 256, /32 512
        del net

        e0, e1, e2, e3, e4 = _RESNET_STAGES[encoder]
        d0, d1, d2, d3, d4 = self.decoder_ch
        self.dec4 = _DecoderBlock(e4, e3, d0)
        self.dec3 = _DecoderBlock(d0, e2, d1)
        self.dec2 = _DecoderBlock(d1, e1, d2)
        self.dec1 = _DecoderBlock(d2, e0, d3)
        self.dec0 = _DecoderBlock(d3, 0, d4)
        self.head = nn.Conv2d(d4, 1, kernel_size=1)

        self._init_decoder()
        self._init_head_prior()

    def _init_decoder(self) -> None:
        for block in (self.dec4, self.dec3, self.dec2, self.dec1, self.dec0):
            for module in block.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def _init_head_prior(self, positive_rate: float = 0.09) -> None:
        """Start the logit head at the data's base rate (see :class:`SmokeUNet`)."""
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        if self.head.bias is not None:
            prior = float(min(max(positive_rate, 1e-4), 1.0 - 1e-4))
            nn.init.constant_(self.head.bias, math.log(prior / (1.0 - prior)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network.

        Args:
            x: ``(N, 3, H, W)`` float tensor, normalised by the caller.

        Returns:
            ``(N, 1, H, W)`` raw logits.
        """
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        d = self.dec4(s4, s3)
        d = self.dec3(d, s2)
        d = self.dec2(d, s1)
        d = self.dec1(d, s0)
        d = self.dec0(d, None)
        return self.head(d)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of (trainable) parameters."""
        return int(sum(p.numel() for p in self.parameters()
                       if p.requires_grad or not trainable_only))


def build_arch(arch: str, *, base: int = 16, pretrained: bool = False) -> nn.Module:
    """Construct the architecture named by a checkpoint's ``arch`` field.

    Checkpoints written before the ResNet encoders existed carry
    ``arch="SmokeUNet"`` or no ``arch`` at all; both mean :class:`SmokeUNet`,
    so old checkpoints keep loading unchanged.

    Args:
        arch: ``"SmokeUNet"`` or ``"ResNetUNet:<encoder>"``.
        base: First-stage width, for :class:`SmokeUNet` only.
        pretrained: Training-only switch; see :class:`ResNetUNet`.

    Raises:
        ValueError: The name is not a known architecture.
    """
    name = str(arch or "SmokeUNet")
    if name == "SmokeUNet":
        return SmokeUNet(in_ch=3, base=int(base))
    if name.startswith("ResNetUNet:"):
        return ResNetUNet(encoder=name.split(":", 1)[1], pretrained=bool(pretrained))
    raise ValueError(f"unknown architecture {arch!r}")


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters of any module (module-level convenience helper)."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only))


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    dummy = torch.zeros(2, 3, 256, 256)
    for spec in ("SmokeUNet", "ResNetUNet:resnet18", "ResNetUNet:resnet34"):
        net = build_arch(spec)
        out = net(dummy)
        print(f"{spec} params={count_parameters(net):,} out={tuple(out.shape)}")
