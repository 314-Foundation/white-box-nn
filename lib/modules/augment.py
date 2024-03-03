import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from lib.helpers import hh


class Mixup(nn.Module):
    def __init__(self, eps=0.3, min_eps=None, p=0.5):
        super().__init__()
        self.eps = eps
        self.min_eps = min_eps or eps
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x

        eps = self.eps
        diff = max(self.eps - self.min_eps, 0)
        if diff > 0:
            diff = random.random() * diff
            eps = self.min_eps + diff
        return hh.mixup_data(x, eps=eps)[0]

    def __repr__(self):
        return f"Mixup(eps={self.eps})"


class Noise(nn.Module):
    def __init__(self, eps, clip=False, p=0.5):
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
        return f"Noise({self.eps}, clip={self.clip})"


class Rand(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x = 2 * (torch.rand_like(x) - 0.5) * self.eps + x
        return x.clip(0, 1)

    def __repr__(self):
        return f"Rand({self.eps})"


class Drop(nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        return torch.bernoulli(torch.ones_like(x) * (1 - self.prob)) * x

    def __repr__(self):
        return f"Drop(prob={self.prob})"


class Bern(nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        x = torch.bernoulli(torch.ones_like(x) * self.prob) + x
        return x.clip(0, 1)

    def __repr__(self):
        return f"Bern(prob={self.prob})"


class Mult(nn.Module):
    def __init__(self, mult=1.0):
        super().__init__()
        self.mult = mult

    def forward(self, x):
        return x * self.mult

    def __repr__(self):
        return f"Mult(mult={self.mult})"


class Zero(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)
