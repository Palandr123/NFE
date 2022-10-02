import torch
import torch.nn as nn
from layers import ScaledLinear, ScaledConv2d, StyledConvBlock


class Generator(nn.Module):
    def __init__(self, z_dim=512, n_linear=5):
        super(Generator, self).__init__()
        layers = []
        for i in range(n_linear):
            layers.append(ScaledLinear(z_dim, z_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.style = nn.Sequential(*layers)
        self.progression = nn.ModuleList(
            [
              StyledConvBlock(512, 512, 3, 1, initial=True),
              StyledConvBlock(512, 256, 3, 1, upsample=True),
              StyledConvBlock(256, 128, 3, 1, upsample=True),
              StyledConvBlock(128, 64, 3, 1, upsample=True),
              StyledConvBlock(64, 32, 3, 1, upsample=True),
            ]
        )
        self.to_rgb = ScaledConv2d(32, 3, 1)

    def forward(self, x, noise=None, step=0):
        batch = x.size(0)
        if noise is None:
            noise = []
            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size, size, device=x[0].device))
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        styles = self.style(x)
        out = noise[0]
        for i, conv in enumerate(self.progression):
            out = self.progression[i](out, styles, noise[i])
        return self.to_rgb(out)
