import random

import torch
from torch import nn


class Noise(nn.Module):
    def __init__(self, eps, clip=True, p=0.5):
        super().__init__()
        self.eps = eps
        self.clip = clip
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x

        x = torch.randn_like(x) * self.eps + x

        if self.clip:
            x = x.clip(0, 1)

        return x

    def __repr__(self):
        return f"Noise(eps={self.eps}, p={self.p}, clip={self.clip})"
