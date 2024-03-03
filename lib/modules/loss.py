from torch import nn


class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, x1, x2):
        return (1 - self.cos(x1.flatten(1), x2.flatten(1))).mean()
