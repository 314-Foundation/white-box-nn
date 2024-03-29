import random

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy, PrecisionRecallCurve

from ..helpers import hh


class ClsModule(LightningModule):
    def __init__(
        self,
        learning_rate=3e-3,
        weight_decay=0.0,
        noise_eps=0.3,
        noise_eps_step=0.1,
        adv_prob=0.0,
        val_adv_prob=1.0,
        n_classes=10,
        augment=None,
        backbone=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("augment", "backbone"))

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.noise_eps = noise_eps
        self.noise_eps_step = noise_eps_step
        self.adv_prob = adv_prob
        self.val_adv_prob = val_adv_prob if val_adv_prob is not None else adv_prob

        self.loss = nn.CrossEntropyLoss()
        self.robby_loss = self.loss
        self.n_classes = n_classes

        self.augment = augment or nn.Identity()
        self.backbone = backbone or nn.Identity()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=n_classes)

        self.reliability_curve = PrecisionRecallCurve(task="binary")

    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx):
        assert self.training
        x, label = batch

        x = self.training_noise(x, label)
        x = self(x)
        loss = self.loss(x, label)

        self.log_train(x, label, loss)

        return loss

    def validation_step(self, batch, batch_idx):
        assert not self.training
        x, label = batch

        x = self.validation_noise(x, label)
        x = self(x)
        loss = self.loss(x, label)

        self.log_validation(x, label, loss)

    def test_step(self, batch, batch_idx):
        assert not self.training
        x, label = batch

        x = self.test_noise(x, label)
        x = self(x)
        loss = self.loss(x, label)

        self.log_test(x, label, loss)

    def training_noise(self, x, y):
        if random.random() > self.adv_prob:
            return self.augment(x)

        eps, step_size = self.noise_eps, self.noise_eps_step
        x = hh.compute_advs(self, x, y, eps=eps, step_size=step_size, Nsteps=10)

        return x

    def validation_noise(self, x, y):
        if random.random() > self.val_adv_prob:
            return x

        eps, step_size = self.noise_eps, self.noise_eps_step
        return hh.compute_advs(
            self, x, y, eps=eps, step_size=step_size, Nsteps=10, random_start=True
        )

    def test_noise(self, x, y):
        eps, step_size = self.noise_eps, self.noise_eps_step
        return hh.compute_advs(
            self, x, y, eps=eps, step_size=step_size, Nsteps=40, random_start=True
        )

    def configure_optimizers(self):
        # https://lightning.ai/docs/pytorch/stable/common/optimization.html
        parameters = self.parameters()
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )  # 0.001
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate) # 0.1
        return optimizer

    def log_train(self, res, y, loss):
        self.log("train_loss", loss, prog_bar=True)
        self.train_accuracy(res, y)
        self.log("train_acc", self.train_accuracy, prog_bar=True)

    def log_validation(self, res, y, loss):
        self.log("val_adv_loss", loss, prog_bar=True)
        self.val_accuracy(res, y)
        self.log("val_adv_acc", self.val_accuracy, prog_bar=True)

    def log_test(self, res, y, loss):
        self.log("test_adv_loss", loss, prog_bar=True)
        self.test_accuracy(res, y)
        self.log("test_adv_acc", self.test_accuracy, prog_bar=True)

        confidence, preds = res.softmax(dim=1).max(dim=1)
        correct = (preds == y).long()

        self.reliability_curve(confidence, correct)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for m in self.modules():
            # NOTE - isinstance(m, Module) doesn't work well with jupyter autoreload
            if (
                hasattr(m, "regularize_weights_on_train_batch_end")
                and m.regularize_weights_on_train_batch_end
            ):
                m.regularize_weights()
