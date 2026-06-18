from __future__ import annotations

import torch

from sprinkler_hd_gan.models.networks import GlobalGenerator, MultiScaleDiscriminator


def test_gan_forward():
    b, c_in, h, w = 2, 6, 128, 160
    x = torch.randn(b, c_in, h, w)
    y = torch.randn(b, 3, h, w)

    g = GlobalGenerator(c_in, 3, ngf=32, n_downsampling=3, n_blocks=3)
    fake = g(x)
    assert fake.shape == (b, 3, h, w)

    d = MultiScaleDiscriminator(c_in + 3, ndf=32, n_layers=2, num_D=2)
    for si in range(d.num_D):
        def at_scale(t, si):
            cur = t
            for _ in range(si):
                cur = torch.nn.functional.avg_pool2d(cur, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            return cur

        hx = at_scale(x, si)
        pr = d.nets[si](torch.cat([hx, at_scale(y, si)], dim=1))
        pf = d.nets[si](torch.cat([hx, at_scale(fake.detach(), si)], dim=1))
        assert pr[0].ndim == 4
        assert pf[0].ndim == 4
