import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy

from ..helpers import hh
from ..modules.basic import Binarize, CosineLoss, FakeClip
from ..modules.conv import (
    PatchesFolder,
    PatchesToTokens,
    StateAndLabelToGray,
    TokenMixer,
)


class BaseModule(LightningModule):
    """Example LightningModule"""

    loss: nn.Module

    def __init__(
        self, learning_rate=0.001, noise_eps=0.0, adv_prob=0.0, val_adv_prob=None
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.noise_eps = noise_eps
        self.adv_prob = adv_prob
        self.val_adv_prob = val_adv_prob if val_adv_prob is not None else adv_prob
        # self.save_hyperparameters()

    def training_noise(self, x, y):
        return x, hh.add_adv_noise(
            self, x, y, eps=self.noise_eps, adv_prob=self.adv_prob
        )

    def validation_noise(self, x, y):
        return x, hh.add_adv_noise(self, x, y, eps=0.3, adv_prob=self.val_adv_prob)

    def test_noise(self, x, y):
        return x, hh.add_adv_noise(self, x, y, eps=0.3, adv_prob=0.0)

    def training_step(self, batch, batch_idx):
        assert self.training
        x, y = batch

        clean_x, x = self.training_noise(x, y)

        res = self(x)
        loss = self.loss(res, y)

        self.log_train(res, y, loss)

        return loss

    def validation_step(self, batch, batch_idx):
        assert not self.training
        x, y = batch

        clean_x, x = self.validation_noise(x, y)

        res = self(x)
        loss = self.loss(res, y)

        self.log_validation(res, y, loss)

    def test_step(self, batch, batch_idx):
        assert not self.training
        x, y = batch

        clean_x, x = self.test_noise(x, y)

        res = self(x)
        loss = self.loss(res, y)

        self.log_test(res, y, loss)

    def log_train(self, res, y, loss):
        self.log("train_loss", loss, prog_bar=True)

    def log_validation(self, res, y, loss):
        self.log("val_loss", loss, prog_bar=True)

    def log_test(self, res, y, loss):
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        # https://lightning.ai/docs/pytorch/stable/common/optimization.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # 0.001
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate) # 0.1
        return optimizer


class ClsModule(BaseModule):
    train_accuracy: Accuracy
    val_accuracy: Accuracy
    test_accuracy: Accuracy

    def compute_confidence(self, res, y):
        conf, preds = res.softmax(dim=1).max(dim=1)
        corr_conf = conf[preds == y]
        wrong_conf = conf[preds != y]

        return corr_conf, wrong_conf

    def log_confidence(self, res, y):
        corr_conf, wrong_conf = self.compute_confidence(res, y)
        wcm = wrong_conf.max() if wrong_conf.numel() > 0 else 0.0
        wca = wrong_conf.mean() if wrong_conf.numel() > 0 else 0.0
        ccm = corr_conf.max() if corr_conf.numel() > 0 else 0.0
        cca = corr_conf.mean() if corr_conf.numel() > 0 else 0.0

        self.log("wcm", wcm, prog_bar=True)
        self.log("wca", wca, prog_bar=True)
        self.log("ccm", ccm, prog_bar=True)
        self.log("cca", cca, prog_bar=True)

    def log_train(self, res, y, loss):
        super().log_train(res, y, loss)
        self.train_accuracy(res, y)
        self.log("train_acc", self.train_accuracy, prog_bar=True)

    def log_validation(self, res, y, loss):
        super().log_validation(res, y, loss)
        self.val_accuracy(res, y)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        self.log_confidence(res, y)

    def log_test(self, res, y, loss):
        super().log_test(res, y, loss)
        self.test_accuracy(res, y)
        self.log("test_acc", self.test_accuracy, prog_bar=True)


class MultilabelClsModule(ClsModule):
    def __init__(
        self,
        learning_rate=0.001,
        noise_eps=0.0,
        adv_prob=0.0,
        val_adv_prob=None,
        n_classes=10,
    ):
        super().__init__(
            learning_rate=learning_rate,
            noise_eps=noise_eps,
            adv_prob=adv_prob,
            val_adv_prob=val_adv_prob,
        )

        self.loss = nn.CrossEntropyLoss()
        self.robby_loss = self.loss
        self.n_classes = n_classes

        self.train_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=n_classes)


class SSLWithProbeModule(MultilabelClsModule):
    backbone: nn.Module
    cls_head: nn.Module
    cls_weight: float
    detach_head: bool

    def forward(self, x):
        x = self.backbone(x)
        x = self.cls_head(x)
        return x

    def training_step(self, batch, batch_idx):
        assert self.training
        x, target = batch
        label = self.target_to_label(target)

        clean_x, x = self.training_noise(x, target)

        repr, pred_loss = self.backbone(
            x,
            clean_x=clean_x,
            # n_updates=min(self.backbone.n_updates, self.current_epoch + 1),
        )
        if self.detach_head:
            repr = repr.detach()
        res = self.cls_head(repr)
        loss = self.loss(res, label)

        self.log_train(res, label, loss, pred_loss)

        return self.cls_weight * loss + pred_loss
        # return self.cls_weight * loss

    def validation_step(self, batch, batch_idx):
        assert not self.training
        x, target = batch
        label = self.target_to_label(target)

        clean_x, x = self.validation_noise(x, target)

        # pred_loss = 0.0
        # repr = self.backbone(
        repr, pred_loss = self.backbone(
            x,
            clean_x=clean_x,
            # clean_x=None,
            # n_updates=min(self.backbone.n_updates, self.current_epoch + 1),
        )
        res = self.cls_head(repr)
        loss = self.loss(res, label)

        self.log_validation(res, label, loss, pred_loss)

    def test_step(self, batch, batch_idx):
        assert not self.training
        x, target = batch
        label = self.target_to_label(target)

        clean_x, x = self.test_noise(x, label)

        repr, pred_loss = self.backbone(x, clean_x=clean_x)
        res = self.cls_head(repr)
        loss = self.loss(res, label)

        self.log_test(res, label, loss, pred_loss)

    def target_to_label(self, target):
        return target

    def log_train(self, res, y, loss, pred_loss):
        super().log_train(res, y, loss)
        self.log("train_pred_loss", pred_loss, prog_bar=True)

    def log_validation(self, res, y, loss, pred_loss):
        super().log_validation(res, y, loss)
        self.log("val_pred_loss", pred_loss, prog_bar=True)

    def log_test(self, res, y, loss, pred_loss):
        super().log_test(res, y, loss)
        self.log("test_pred_loss", pred_loss, prog_bar=True)


class BinaryDenoisingModule(ClsModule):
    def __init__(
        self, learning_rate=0.001, noise_eps=0.3, adv_prob=0.0, val_adv_prob=None
    ):
        super().__init__(
            learning_rate=learning_rate,
            noise_eps=noise_eps,
            adv_prob=adv_prob,
            val_adv_prob=val_adv_prob,
        )

        self.loss = nn.BCEWithLogitsLoss()
        self.robby_loss = self.loss

        self.binarize = Binarize()

        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")

    def training_step(self, batch, batch_idx):
        x, y = batch
        return super().training_step((x, x), batch_idx)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return super().validation_step((x, x), batch_idx)

    def test_step(self, batch, batch_idx):
        x, y = batch
        return super().test_step((x, x), batch_idx)

    def log_train(self, res, y, loss):
        super().log_train(res, self.binarize(y), loss)

    def log_validation(self, res, y, loss):
        super().log_validation(res, self.binarize(y), loss)

    def log_test(self, res, y, loss):
        super().log_test(res, self.binarize(y), loss)


class AmbivalentModule(BinaryDenoisingModule):
    def __init__(
        self,
        learning_rate=0.001,
        noise_eps=0.3,
        adv_prob=0.0,
        val_adv_prob=None,
        n_classes=10,
        dim=64,
        depth=5,
        kernel_size=5,
        patch_size=4,
        img_shape=(28, 28),
        n_iter=3,
    ):
        super().__init__(
            learning_rate=learning_rate,
            noise_eps=noise_eps,
            adv_prob=adv_prob,
            val_adv_prob=val_adv_prob,
        )
        assert n_iter > 0

        self.save_hyperparameters()
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.img_shape = img_shape
        self.n_iter = n_iter
        self.to_patches = PatchesFolder(self.patch_size, self.img_shape, down=True)
        self.patches_to_state = PatchesToState(
            dim=dim, depth=depth, kernel_size=kernel_size, patch_size=patch_size
        )
        self.denoising_neck = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )
        self.stc_to_gray = nn.Sequential(
            StateAndLabelToGray(dim, n_classes, patch_size),
            PatchesFolder(self.patch_size, self.img_shape, down=False),
        )
        # self.htanh = nn.Hardtanh(-1, 1)

    def mixup_batch(self, batch):
        x, y = batch
        b, c, h, w = x.shape
        assert b % 2 == 0

        x1, x2 = torch.split(x, [b // 2, b // 2], dim=0)
        x_amb = ((x1 + x2) / 2).repeat(2, 1, 1, 1)
        # y2 = torch.roll(y, b // 2)

        return x_amb, x, y

    def attach_context(self, x, y):
        y = (F.one_hot(y, self.n_classes).float() * 6).softmax(
            dim=1
        )  # tensor([0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.9782])
        y = y.reshape(-1, self.n_classes, 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
        return torch.cat([x, y], dim=1)

    def extract_state(self, x):
        x = self.to_patches(x)
        x = self.patches_to_state(x)
        return x

    def forward(self, x, y):
        st = self.extract_state(x)
        st = self.denoising_neck(st)
        stc = self.attach_context(st, y)
        gray = self.stc_to_gray(stc)

        return gray

    def loop_loss(self, x, xt, y):
        x = self(x, y)
        loss = self.loss(x, xt)
        for _ in range(self.n_iter - 1):
            x = self(x.detach().sigmoid(), y)
            loss += self.loss(x, xt)

        return loss / self.n_iter, x

    def training_step(self, batch, batch_idx):
        assert self.training
        x, xt, y = self.mixup_batch(batch)

        loss, res = self.loop_loss(x, xt, y)

        self.log_train(res, xt, loss)

        return loss

    def validation_step(self, batch, batch_idx):
        assert not self.training
        x, xt, y = self.mixup_batch(batch)

        loss, res = self.loop_loss(x, xt, y)

        self.log_validation(res, xt, loss)

    def test_step(self, batch, batch_idx):
        assert not self.training
        x, y = batch
        # x, xt, y = self.mixup_batch(batch)

        loss, res = self.loop_loss(x, x, y)

        self.log_test(res, x, loss)


class AmbivalentModuleWithState(BinaryDenoisingModule):
    def __init__(
        self,
        learning_rate=0.001,
        noise_eps=0.3,
        adv_prob=0.0,
        val_adv_prob=None,
        n_classes=10,
        dim=64,
        n_filters=128,
        depth=5,
        kernel_size=5,
        patch_size=4,
        img_shape=(28, 28),
    ):
        super().__init__(
            learning_rate=learning_rate,
            noise_eps=noise_eps,
            adv_prob=adv_prob,
            val_adv_prob=val_adv_prob,
        )

        self.save_hyperparameters()
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.img_shape = img_shape
        self.dim = dim
        self.to_patches = PatchesFolder(patch_size, img_shape, down=True)
        self.patches_to_vector = PatchesToVector(patch_size, n_filters)
        self.stc_to_gray = nn.Sequential(
            StateAndLabelToGray(dim, n_classes, patch_size),
            PatchesFolder(patch_size, img_shape, down=False),
        )
        self.input_and_state_mixer = nn.Sequential(
            nn.Conv2d(dim + n_filters, 2 * dim, 1),
            nn.GELU(),
            nn.Conv2d(2 * dim, dim, 1),
            nn.GELU(),
        )
        self.state_mixer = TokenMixer(dim, depth, kernel_size)
        # self.state_update = nn.Conv2d(dim, dim, 1)

        # nn.init.zeros_(self.state_update.weight)
        # nn.init.zeros_(self.state_update.bias)

        # self.htanh = nn.Hardtanh(-1, 1)

    def mixup_batch(self, batch):
        x, y = batch
        b, c, h, w = x.shape
        assert b % 2 == 0

        x1, x2 = torch.split(x, [b // 2, b // 2], dim=0)
        x_amb = ((x1 + x2) / 2).repeat(2, 1, 1, 1)
        # y2 = torch.roll(y, b // 2)

        return x_amb, x, y

    def attach_bias(self, x, y):
        y = (F.one_hot(y, self.n_classes).float() * 6).softmax(
            dim=1
        )  # tensor([0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.9782])
        y = y.reshape(-1, self.n_classes, 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
        return torch.cat([x, y], dim=1)

    def encode_input(self, x):
        x = self.to_patches(x)
        x = self.patches_to_vector(x)
        return x

    def init_state(self, x):
        return torch.zeros(
            (x.shape[0], self.dim, x.shape[2], x.shape[3]),
            device=x.device,
            dtype=x.dtype,
            requires_grad=False,
        )

    def forward(self, x, y):
        inp = self.encode_input(x)
        st = self.init_state(inp)
        st_inp = torch.cat([st, inp], dim=1)
        st = self.input_and_state_mixer(st_inp)
        st = self.state_mixer(st)
        stb = self.attach_bias(st, y)
        x = self.stc_to_gray(stb)

        return x

    def training_step(self, batch, batch_idx):
        assert self.training
        x, xt, y = self.mixup_batch(batch)

        res = self(x, y)
        loss = self.loss(res, xt)

        self.log_train(res, xt, loss)

        return loss

    def validation_step(self, batch, batch_idx):
        assert not self.training
        x, xt, y = self.mixup_batch(batch)

        res = self(x, y)
        loss = self.loss(res, xt)

        self.log_validation(res, xt, loss)

    def test_step(self, batch, batch_idx):
        assert not self.training
        x, y = batch
        # x, xt, y = self.mixup_batch(batch)

        res = self(x, y)
        loss = self.loss(res, x)

        self.log_test(res, x, loss)


class BobOld(MultilabelClsModule):
    def __init__(
        self,
        learning_rate=0.001,
        noise_eps=0.3,
        adv_prob=0.0,
        val_adv_prob=None,
        n_classes=10,
        dim=64,
        n_filters=64,
        depth=5,
        kernel_size=5,
        patch_size=4,
        img_shape=(28, 28),
        n_steps=10,
        fixed_selection=False,
        # batch_size=64,
    ):
        super().__init__(
            learning_rate=learning_rate,
            noise_eps=noise_eps,
            adv_prob=adv_prob,
            val_adv_prob=val_adv_prob,
        )

        self.save_hyperparameters()
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.img_shape = img_shape
        # self.state = None
        # self.batch_size = batch_size
        self.h, self.w = img_shape[0] // patch_size, img_shape[1] // patch_size
        self.dim = dim
        self.n_steps = n_steps

        self.extract_tokens = nn.Sequential(
            PatchesFolder(patch_size, img_shape),
            PatchesToTokens(n_filters, kernel_size, patch_size**2),
        )
        self.top_down = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )
        self.state_and_token_mixer = nn.Sequential(
            nn.Conv2d(dim + n_filters, dim, 1),
            nn.GELU(),
        )
        self.st_mixer = TokenMixer(dim, depth, kernel_size)
        self.state_to_label = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(dim, n_classes)
        )
        self.state_update = nn.Conv2d(dim, dim, 1)

        nn.init.zeros_(self.state_update.weight)
        nn.init.zeros_(self.state_update.bias)

    def reset_state(self, batch_size):
        self.state = torch.zeros(
            (batch_size, self.dim, self.h, self.w),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )

    def unroll_inp(self, x):
        return [(x * i) / self.n_steps for i in range(1, self.n_steps + 1)]

    def forward(self, x):
        tk = self.extract_tokens(x)
        st = self.state_and_token_mixer(
            torch.cat([self.top_down(self.state), tk], dim=1)
        )
        st = self.st_mixer(st)
        self.state = self.state + self.state_update(st)
        ret = self.state_to_label(self.state)

        return ret

    def training_step(self, batch, batch_idx):
        assert self.training

        x, y = batch
        self.reset_state(x.shape[0])
        xx = self.unroll_inp(x)
        res = [self(xi) for xi in xx][-1]

        loss = self.loss(res, y)

        self.log_train(res, y, loss)

        return loss

    def validation_step(self, batch, batch_idx):
        assert not self.training

        x, y = batch
        self.reset_state(x.shape[0])
        xx = self.unroll_inp(x)
        res = [self(xi) for xi in xx][-1]

        loss = self.loss(res, y)

        self.log_validation(res, y, loss)

    def test_step(self, batch, batch_idx):
        assert not self.training

        x, y = batch
        self.reset_state(x.shape[0])
        xx = self.unroll_inp(x)
        res = [self(xi) for xi in xx][-1]

        loss = self.loss(res, y)

        self.log_test(res, y, loss)


class BobWrapperOld(nn.Module):
    def __init__(self, bob):
        super().__init__()
        self.bob = bob

    def forward(self, x):
        self.bob.reset_state(x.shape[0])
        xx = self.bob.unroll_inp(x)
        results = [self.bob(xi) for xi in xx]
        return results[-1]


class DenoisingWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        # hp = model.hparams
        # super().__init__(
        #     learning_rate=hp.learning_rate,
        #     noise_eps=hp.noise_eps,
        #     adv_prob=hp.adv_prob,
        #     val_adv_prob=hp.val_adv_prob,
        # )
        self.model = model
        self.fake_clip = FakeClip()
        self.loss = nn.MSELoss()
        # self.robby_loss = nn.BCEWithLogitsLoss()
        self.robby_loss = CosineLoss()

    def forward(self, x):
        x = self.model(x, return_pred=True)
        x = self.model.backbone.preds_to_pred(x)
        x = self.fake_clip(x)
        # x = x.flatten(1)
        return x
