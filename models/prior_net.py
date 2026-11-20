"""Prior Information Acquisition Network (PIAN).

Takes a masked image and compresses the visible content into a compact
256-dimensional prior vector that conditions the diffusion network.
"""

import torch
import torch.nn as nn


def inner_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, act=nn.LeakyReLU(0.1, inplace=True)):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class Prior_Extraction_Network(nn.Module):
    """PIAN: masked image -> compact prior vector.

    ``prior_dim`` is the dimension of the produced prior vector (256 in the paper).
    """

    def __init__(self, n_feats=64, n_encoder_res=6, prior_dim=256, downsample_factor=4):
        super(Prior_Extraction_Network, self).__init__()

        self.prior_dim = prior_dim
        self.loss_func = nn.L1Loss(reduction='mean')

        Enconder = [nn.Conv2d(3 * downsample_factor ** 2, n_feats, kernel_size=1, padding=0),
                    nn.LeakyReLU(0.1, True),
                    *[ResBlock(inner_conv, n_feats, kernel_size=3) for _ in range(n_encoder_res)],
                    nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.1, True),
                    nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.1, True),
                    nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.1, True),
                    nn.AdaptiveAvgPool2d(1),
                    ]

        self.Enconder = nn.Sequential(*Enconder)
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, prior_dim),
            nn.LeakyReLU(0.1, True)
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(downsample_factor)

    def priori_losses(self, gd_value, k_value):
        """L1 distance between the prior of the masked image and of the ground truth."""
        return self.loss_func(gd_value, k_value)

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        # (N, 3*4*4, 256/4, 256/4)
        prior = self.Enconder(x).squeeze(-1).squeeze(-1)
        prior = self.mlp(prior)
        # (N, prior_dim)
        return prior


if __name__ == "__main__":
    img = torch.Tensor(4, 3, 256, 256)
    model = Prior_Extraction_Network()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    out = model(img)
    print(out.shape)
