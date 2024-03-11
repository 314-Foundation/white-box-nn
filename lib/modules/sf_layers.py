import torch
import torch.nn.functional as F
from kornia.geometry.transform import rotate
from torch import einsum, nn

from lib.helpers import normalize_weight_
from lib.modules.base import Module


class SFLayer(Module):
    def __init__(
        self,
        sampler,
    ):
        super().__init__()
        self.sampler = sampler

        self.weight_cache = None

    def get_features(self):
        pass

    def _get_weight(self):
        ff = self.get_features()
        ww = self.sampler(ff)
        return ww

    def get_weight(self):
        if not self.training and self.weight_cache is not None:
            return self.weight_cache

        ww = self._get_weight()

        if self.training:
            self.weight_cache = None
        else:
            self.weight_cache = ww.detach().clone()

        return ww


class TwoStepFunction(Module):
    """Simplified integer-valued SFLayer for inputs in [0, 1]"""

    def __init__(
        self,
        init_scales_div=1,
        learnable=True,
    ):
        super().__init__()
        self.init_scales_div = init_scales_div
        self.learnable = learnable

        self.after_init()

    def init_weights(self):
        self.a0 = nn.Parameter(torch.tensor(0.5), requires_grad=self.learnable)
        self.a1 = nn.Parameter(torch.tensor(0.5), requires_grad=self.learnable)
        self.t0 = nn.Parameter(torch.tensor(0.25), requires_grad=self.learnable)
        self.t1 = nn.Parameter(torch.tensor(0.75), requires_grad=self.learnable)

        self.scales = nn.Parameter(
            (torch.ones((4,)) / self.init_scales_div), requires_grad=self.learnable
        )

    def regularize_weights(self):
        with torch.no_grad():
            self.a0.clip_(0.01, 0.99)
            self.a1.clip_(0.01, 0.99)
            self.t0.clip_(0.01, 0.49)
            self.t1.clip_(0.51, 0.99)
            self.scales.clip_(0.001, 2.0)

    def softsign(self, x, threshold, scale, internal):
        a = self.a1 if threshold > 0 else self.a0

        x = x - threshold
        x = x / (scale + x.abs())

        if internal:
            shift = threshold
            shift = shift / (scale + shift.abs())
            x = x + shift
            x = a * x / shift.abs()

            return x

        div = x.abs().max() / (1 - a)
        x = x / div
        x = x + threshold.sgn() * a

        return x

    def transform(self, x):
        # [0, 1] -> [-1, 1]
        return 2 * x - 1

    def forward(self, x):
        assert x.min() >= 0 and x.max() <= 1

        x = self.transform(x)
        t0, t1 = self.transform(self.t0), self.transform(self.t1)

        params = [
            (t0, self.scales[0], False),
            (t0, self.scales[1], True),
            (t1, self.scales[2], True),
            (t1, self.scales[3], False),
        ]
        xs = [self.softsign(x, *p) for p in params]

        masks = [
            (x < t0).float(),
            (t0 <= x).float() * (x < 0.0).float(),
            (0.0 <= x).float() * (x < t1).float(),
            (t1 <= x).float(),
        ]

        x = sum(m * xx for (m, xx) in zip(masks, xs))

        return x

    def __repr__(self):
        ndigits = 3
        t0 = round(self.t0.item(), ndigits=ndigits)
        t1 = round(self.t1.item(), ndigits=ndigits)
        a0 = round(self.a0.item(), ndigits=ndigits)
        a1 = round(self.a1.item(), ndigits=ndigits)
        scales = [round(s.item(), ndigits=ndigits) for s in self.scales]
        return f"TwoStepFunction({t0}, {t1}, {a0}, {a1}, scales={scales})"


class ConvLayer(SFLayer):
    def __init__(
        self,
        sampler,
        n_kernels,
        kernel_size=5,
        act=None,
        add_bias=False,
        rescale=False,
    ):
        super().__init__(sampler)

        self.in_channels = 1
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size

        self.act = act or nn.Identity()

        self.add_bias = add_bias
        self.rescale = rescale
        # self.bias = nn.Parameter(torch.zeros((1, self.n_kernels, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.scale = nn.Parameter(torch.ones((1,)))

        self.after_init()

    def init_weights(self):
        self.conv = nn.Conv2d(
            self.in_channels,
            self.n_kernels,
            kernel_size=self.kernel_size,
            padding="same",
            bias=False,
        )

        with torch.no_grad():
            w = self.conv.weight
            shape = w.shape[1:]
            ch, cw = shape[1] // 2, shape[2] // 2

            nn.init.zeros_(w)
            w[:, :, ch, cw] = 1.0
            w.add_(torch.randn_like(w) * 0.1)

    def regularize_weights(self):
        with torch.no_grad():
            w = self.conv.weight
            normalize_weight_(w.flatten(1))

    def get_features(self):
        return self.conv.weight

    def forward(self, x):
        # x - b c H W
        ww = self.get_weight()  # f p c h w
        ww = ww.flatten(0, 1)  # (f p) c h w
        scores = F.conv2d(
            x,
            ww,
            None,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )  # b (f p) H W
        # scores = scores.unflatten(1, (self.n_kernels, -1))  # b f p H W
        x, idx = scores.max(dim=1)  # b f H W
        x = x / self.kernel_size  # math.sqrt(kernel_dim) - to keep the scale of input

        if self.rescale:
            x = x * self.scale
        if self.add_bias:
            x = x + self.bias

        x = x[:, None]
        x = self.act(x)

        return x


class AffineLayer(SFLayer):
    def __init__(
        self,
        sampler,
        inp_shape,
        feature_shape,
        out_dim,
        act=None,
        add_bias=False,
        rescale=False,
    ):
        super().__init__(sampler)
        self.inp_shape = inp_shape
        self.inp_dim = torch.prod(torch.tensor(inp_shape))
        self.feature_shape = feature_shape
        self.feature_dim = torch.prod(torch.tensor(self.feature_shape))
        self.out_dim = out_dim

        self.act = act or nn.Identity()

        self.add_bias = add_bias
        self.rescale = rescale
        self.bias = nn.Parameter(torch.zeros((1, self.out_dim)))
        # self.scale = nn.Parameter(torch.ones((1,)))
        self.scale = nn.Parameter(torch.ones((1, self.out_dim)))

        self.feature_padding = (
            (self.inp_shape[1] - self.feature_shape[1]) // 2,
            (self.inp_shape[1] - self.feature_shape[1]) // 2,
            (self.inp_shape[2] - self.feature_shape[2]) // 2,
            (self.inp_shape[2] - self.feature_shape[2]) // 2,
        )

        self.after_init()

    def init_weights(self):
        self.features = nn.Parameter(torch.empty((self.out_dim, self.feature_dim)))
        f = self.features

        # f.normal_(0.0, std=1.0)
        # nn.init.uniform_(f, -1.0, 1.0)
        nn.init.kaiming_uniform_(f, a=2.2236)
        # normalize_weight_(f)

    def regularize_weights(self):
        with torch.no_grad():
            normalize_weight_(self.features)

    def get_features(self):
        return F.pad(
            self.features.unflatten(1, self.feature_shape), self.feature_padding
        )

    def forward(self, x, with_features=False):
        # x - b c h w
        ww = self.get_weight()  # f p c h w
        x, ww = x.flatten(1), ww.flatten(2)

        scores = einsum("b i, f p i -> b p f", x, ww)

        x, idx = scores.max(dim=1)  # b f

        if self.rescale:
            x = x * self.scale
        if self.add_bias:
            x = x + self.bias

        x = self.act(x)

        if not with_features:
            return x

        idx = idx % self.sampler.n_samples

        # NOTE - the following selects f features for every batch element
        features = ww[
            list(range(idx.shape[1])) * idx.shape[0], idx.flatten()
        ]  # (b f) i
        # features = features * x.flatten()[..., None]  # (b f) i
        features = features.unflatten(0, idx.shape)  # b f i
        features = features.unflatten(2, self.inp_shape)

        return x, features
