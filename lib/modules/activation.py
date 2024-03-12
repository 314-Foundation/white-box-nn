import torch
from torch import nn


class TopKActivation(nn.Module):
    def __init__(self, topk, dim=1):
        super().__init__()
        self.topk = topk
        self.dim = dim

    def forward(self, x):
        k = self.topk
        val, ind = x.topk(k, dim=self.dim)
        ret = torch.zeros_like(x)
        ret.scatter_(self.dim, ind, val)

        return ret

    def __repr__(self):
        return f"TopKActivation({self.topk}, {self.dim})"


class TopKNormActivation(nn.Module):
    def __init__(self, topk, dim=-1):
        super().__init__()
        self.topk = topk
        self.dim = dim

    def forward(self, x):
        # x - (..., out, in)
        abs_val, ind = x.abs().topk(self.topk, dim=self.dim)  # (..., out, self.topk)
        ret = torch.zeros_like(x)
        val = torch.gather(x, self.dim, ind)
        ret.scatter_(self.dim, ind, val)

        return ret

    def __repr__(self):
        return f"TopKNormActivation({self.topk}, {self.dim})"
