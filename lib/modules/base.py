import torch
from torch import nn


class Module(nn.Module):
    regularize_weights_on_train_batch_end = True

    def after_init(self):
        with torch.no_grad():
            self.init_weights()
            self.regularize_weights()

    def init_weights(self):
        pass

    def regularize_weights(self):
        pass
