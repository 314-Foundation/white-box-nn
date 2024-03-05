import random

import torch
from torch import nn


class Noise(nn.Module):
    def __init__(self, mean=0.0, scale=0.3, clip=True, p=0.5):
        super().__init__()
        self.mean = mean
        self.scale = scale
        self.clip = clip
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x

        noise = torch.randn_like(x) * self.scale + self.mean
        x = noise + x

        if self.clip:
            x = x.clip(0, 1)

        return x

    def __repr__(self):
        return (
            f"Noise(mean={self.mean}, scale={self.scale}, p={self.p}, clip={self.clip})"
        )
