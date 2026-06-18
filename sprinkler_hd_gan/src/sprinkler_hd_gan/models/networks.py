from __future__ import annotations

import functools
from typing import Optional, Sequence

import torch
import torch.nn as nn


def get_norm_layer(norm_type: str) -> type[nn.Module]:
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False)
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True)
    raise ValueError(norm_type)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, norm_layer: type[nn.Module]) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            norm_layer(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            norm_layer(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GlobalGenerator(nn.Module):
    """pix2pixHD-style global generator (coarse path), adapted for arbitrary input channels."""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int,
        n_downsampling: int = 4,
        n_blocks: int = 9,
        norm_layer: Optional[type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = get_norm_layer("instance")

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        ch = ngf
        for _ in range(n_downsampling):
            model += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=0),
                norm_layer(ch * 2),
                nn.ReLU(True),
            ]
            ch *= 2

        for _ in range(n_blocks):
            model += [ResidualBlock(ch, norm_layer)]

        for _ in range(n_downsampling):
            model += [
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ch // 2),
                nn.ReLU(True),
            ]
            ch //= 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ch, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator with feature hooks for pix2pixHD feature matching."""

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: Optional[type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = get_norm_layer("instance")

        kw = 4
        padw = 1
        sequence: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        feats: list[torch.Tensor] = []
        h = x
        for layer in self.model:
            h = layer(h)
            if isinstance(layer, nn.Conv2d):
                feats.append(h)
        return h, feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc: int,
        ndf: int,
        n_layers: int,
        num_D: int,
        norm_layer: Optional[type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.num_D = num_D
        self.nets = nn.ModuleList()
        for _ in range(num_D):
            net = NLayerDiscriminator(input_nc, ndf=ndf, n_layers=n_layers, norm_layer=norm_layer)
            self.nets.append(net)

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.avg_pool2d(x, kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        results: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
        h = x
        for i, net in enumerate(self.nets):
            if i != 0:
                h = self.downsample(h)
            results.append(net(h))
        return results

    def compute_gan_loss(
        self,
        pred_fake_maps: Sequence[tuple[torch.Tensor, list[torch.Tensor]]],
        pred_real_maps: Sequence[tuple[torch.Tensor, list[torch.Tensor]]],
        criterion: nn.Module,
    ) -> torch.Tensor:
        loss = 0.0
        for pf, pr in zip(pred_fake_maps, pred_real_maps):
            loss += criterion(pf[0], torch.zeros_like(pf[0]))
            loss += criterion(pr[0], torch.ones_like(pr[0]))
        return loss / self.num_D

    def compute_feature_matching_loss(
        self,
        pred_fake_maps: Sequence[tuple[torch.Tensor, list[torch.Tensor]]],
        pred_real_maps: Sequence[tuple[torch.Tensor, list[torch.Tensor]]],
        lambda_feat: float,
    ) -> torch.Tensor:
        loss = 0.0
        count = 0
        for pf, pr in zip(pred_fake_maps, pred_real_maps):
            for a, b in zip(pf[1], pr[1]):
                loss += torch.nn.functional.l1_loss(a, b)
                count += 1
        if count == 0:
            return pred_fake_maps[0][0].new_zeros(())
        return loss * lambda_feat / count
