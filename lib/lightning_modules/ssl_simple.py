import random

import torch
from torch import nn

from ..helpers import hh
from .base import MultilabelClsModule


class SSLModule(MultilabelClsModule):
    def __init__(
        self,
        learning_rate=0.001,
        weight_decay=0.0,
        noise_eps=0.3,
        adv_prob=0.0,
        val_adv_prob=1.0,
        n_classes=10,
        cls_weight=0.1,
        ssl_weight=1.0,
        ignore_inner_loss=False,
        detach_head=True,
        augment=None,
        backbone=None,
        latent_dim=None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            noise_eps=noise_eps,
            adv_prob=adv_prob,
            val_adv_prob=val_adv_prob,
            n_classes=n_classes,
        )

        self.save_hyperparameters(ignore=("augment", "normalize", "base", "backbone"))

        self.cls_weight = cls_weight
        self.ssl_weight = ssl_weight
        self.ignore_inner_loss = ignore_inner_loss
        self.detach_head = detach_head

        self.weight_decay = weight_decay
        self.augment = augment or nn.Identity()
        self.backbone = backbone or nn.Identity()

        self.latent_dim = latent_dim

        self.cls_head = self._make_head()

    def get_repr(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x):
        x = self.get_repr(x)
        x = self.cls_head(x)
        return x

    def get_preds(self, x):
        return self.backbone.get_preds(x)

    def _make_head(self):
        return nn.Sequential(
            nn.Flatten(1),
            nn.LazyLinear(self.n_classes),
        )

    def ssl_target(self, clean, noisy):
        return clean

    def training_step(self, batch, batch_idx):
        assert self.training
        x, label = batch

        clean_x = x
        x = self.training_noise(x, label)
        ssl_target = self.ssl_target(clean_x, x)

        if self.ignore_inner_loss:
            repr = self.backbone(x)
            ssl_loss = 0.0
        else:
            repr, ssl_loss = self.backbone.forward_with_loss(
                x, ssl_target=ssl_target, label=label
            )

        if self.detach_head:
            repr = repr.detach()
        res = self.cls_head(repr)
        loss = self.loss(res, label)

        self.log_train(res, label, loss, ssl_loss)

        return self.cls_weight * loss + ssl_loss * self.ssl_weight

    def validation_step(self, batch, batch_idx):
        assert not self.training
        x, label = batch

        clean_x = x
        x = self.validation_noise(x, label)
        ssl_target = self.ssl_target(clean_x, x)

        repr, ssl_loss = self.backbone.forward_with_loss(
            x, ssl_target=ssl_target, label=label
        )

        res = self.cls_head(repr)
        loss = self.loss(res, label)

        self.log_validation(res, label, loss, ssl_loss)

    def test_step(self, batch, batch_idx):
        assert not self.training
        x, label = batch

        x = self.test_noise(x, label)
        res = self(x)
        loss = self.loss(res, label)
        ssl_loss = 0.0

        self.log_test(res, label, loss, ssl_loss)

    def training_noise(self, x, y):
        if random.random() > self.adv_prob:
            return self.augment(x)

        eps = self.noise_eps
        step_size = eps * 0.3
        x = hh.compute_advs(self, x, y, eps=eps, step_size=step_size, Nsteps=10)

        return x

    def validation_noise(self, x, y):
        if random.random() > self.val_adv_prob:
            return x

        eps = self.noise_eps
        step_size = eps * 0.3
        return hh.compute_advs(
            self, x, y, eps=eps, step_size=step_size, Nsteps=10, random_start=True
        )

    def test_noise(self, x, y):
        eps = self.noise_eps
        step_size = eps * 0.3
        return hh.compute_advs(
            self, x, y, eps=eps, step_size=step_size, Nsteps=20, random_start=True
        )

    def log_train(self, res, y, loss, ssl_loss):
        super().log_train(res, y, loss)
        self.log("train_ssl_loss", ssl_loss, prog_bar=True)

    def log_validation(self, res, y, loss, ssl_loss):
        super().log_validation(res, y, loss)
        self.log("val_ssl_loss", ssl_loss, prog_bar=True)

    def log_test(self, res, y, loss, ssl_loss):
        super().log_test(res, y, loss)
        self.log("test_ssl_loss", ssl_loss, prog_bar=True)

    def configure_optimizers(self):
        # https://lightning.ai/docs/pytorch/stable/common/optimization.html
        # parameters = list(self.backbone.parameters())
        # parameters += list(self.cls_head.parameters())
        # parameters += list(self.normalize.parameters())
        parameters = self.parameters()
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )  # 0.001
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate) # 0.1
        return optimizer

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.backbone.regularize_weights()
