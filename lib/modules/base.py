import torch
from torch import nn


class Module(nn.Module):
    def __init__(self):
        super().__init__()

        with torch.no_grad():
            self.init_weights()
            self.regularize_weights()

    def init_weights(self):
        pass

    def regularize_weights(self):
        pass
