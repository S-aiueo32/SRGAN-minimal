from math import log2

import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    def __init__(self, scale_factor=4):
        super(Generator, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.body = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.tail = nn.Sequential(
            *[UpsampleBLock(64, 2) for _ in range(int(log2(scale_factor)))],
            nn.Conv2d(64, 3, kernel_size=9, padding=4)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, patch_size=96):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            Flatten(),
            nn.Linear(512 * (patch_size // 2 ** 4) ** 2, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return self.net(x) + x


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super(UpsampleBLock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels *
                      (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.net(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
