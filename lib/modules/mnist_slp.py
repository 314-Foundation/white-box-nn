import torch
import torch.nn.functional as F
from kornia.geometry.transform import affine, affine3d
from torch import einsum, nn

from lib.helpers import divide_chunks, sparsify_features
from lib.modules.activation import TopKActivation
from lib.modules.basic import CosineLoss, FakeClip


def prune_coordinate(t, idx, min=None, max=None):
    val = t[idx]
    if min is not None and val < min:
        t[idx] = min
    if max is not None and val > max:
        t[idx] = max


def prune_pose(pose, diff_min=0.25, diff_max=0.25):
    for p in pose:
        prune_coordinate(p, (0, 0), 1 - diff_min, 1 + diff_max)
        prune_coordinate(p, (0, 1), -diff_min, diff_max)
        prune_coordinate(p, (1, 1), 1 - diff_min, 1 + diff_max)
        prune_coordinate(p, (1, 0), -diff_min, diff_max)


def norm(w, p=2.0, dim=-1):
    return torch.norm(w, p=p, dim=dim, keepdim=True) + 1e-10


def normalize_weight(w, p=2.0, dim=-1):
    return w / norm(w, p=p, dim=dim)


def normalize_weight_(w, p=2.0, dim=-1):
    return w.div_(norm(w, p=p, dim=dim))


class Sampler(nn.Module):
    def __init__(
        self,
        n_samples=1,
        learnable=True,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.learnable = learnable

        with torch.no_grad():
            self.init_weights()
            self.regularize_weights()

    def init_weights(self):
        pass

    def regularize_weights(self):
        pass

    def forward(self, x, normalize=False, inverse=False):
        pass


class DummySampler(Sampler):
    def forward(self, x, normalize=False, inverse=False):
        return x[:, None]


class AffineSampler(Sampler):
    def __init__(
        self,
        dim_pose_group=1,
        pose_init_mult=0.01,
        **kwargs,
    ):
        self.dim_pose_group = dim_pose_group
        self.pose_init_mult = pose_init_mult
        super().__init__(**kwargs)

        # self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)

    def init_weights(self):
        poses = [torch.eye(2, 3) for _ in range(self.dim_pose_group * self.n_samples)]

        for w in poses:
            w.add_(torch.randn(2, 3) * self.pose_init_mult)
            loc = torch.cat([torch.zeros((2, 2)), torch.randn(2, 1) * 1.0], dim=-1)
            w.add_(loc)

        self.poses = nn.ParameterList(
            [
                nn.Parameter(
                    torch.stack(group, dim=0),
                    requires_grad=True,
                )
                for group in divide_chunks(poses, self.dim_pose_group)
            ]
        )

    def regularize_weights(self):
        for pose in self.poses:
            pass
            # prune_pose(pose)

    def forward(self, x, normalize=False, inverse=False):
        # x - f c h w
        n_repeats = x.shape[0] // self.dim_pose_group

        poses = [x]
        for pose in self.poses:
            poses.append(
                affine(
                    x,
                    (pose.repeat(n_repeats, 1, 1) if self.dim_pose_group > 1 else pose),
                    mode="bilinear",
                    padding_mode="zeros",
                )
            )

        # poses2 = [self.avg_pool(pose) for pose in poses]
        # poses = poses + poses2
        poses = [pose.flatten(1) for pose in poses]
        poses = torch.stack(poses, dim=-2)  # f p (c h w)

        if normalize:
            poses = normalize_weight(poses)

        return poses


class AffineLayer(nn.Module):
    def __init__(
        self,
        inp_shape,
        filter_shape,
        out_dim,
        n_affine_samples,
        pixel_sampler=None,
        act=None,
        act_down=None,
        clip_pred=None,
        init_sparsity=0.0,
        identify_up_and_down=True,
        add_bias=False,
        add_pred_bias=True,
        dim_pose_group=1,
    ):
        super().__init__()
        self.inp_shape = inp_shape
        self.inp_dim = torch.prod(torch.tensor(inp_shape))
        self.out_dim = out_dim

        self.pixel_sampler = pixel_sampler or DummySampler()
        self.filter_sampler = AffineSampler(
            n_samples=n_affine_samples, dim_pose_group=dim_pose_group
        )

        self.act = act or nn.Identity()
        self.act_down = act_down or nn.Identity()
        self.clip_pred = clip_pred
        self.init_sparsity = init_sparsity

        self.identify_up_and_down = identify_up_and_down
        self.add_bias = add_bias
        self.add_pred_bias = add_pred_bias

        self.bias = nn.Parameter(torch.zeros((1, self.out_dim)))
        self.pred_bias = nn.Parameter(torch.zeros((1, *self.inp_shape)))

        # NOTE - in general filter_dim << inp_dim should hold
        self.filter_shape = filter_shape
        self.filter_dim = torch.prod(torch.tensor(self.filter_shape))

        self.weight_cache = {"up": None, "down": None}

        # self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.topk = TopKActivation(1, dim=1)

        self.filter_padding = (
            (self.inp_shape[1] - self.filter_shape[1]) // 2,
            (self.inp_shape[1] - self.filter_shape[1]) // 2,
            (self.inp_shape[2] - self.filter_shape[2]) // 2,
            (self.inp_shape[2] - self.filter_shape[2]) // 2,
        )

        with torch.no_grad():
            self.init_weights()
            self.regularize_weights()

    def init_weights(self):
        self.up_filters = nn.Parameter(torch.empty((self.out_dim, self.filter_dim)))
        uf = self.up_filters

        # uf.normal_(0.0, std=1.0)
        # nn.init.uniform_(uf, -1.0, 1.0)
        nn.init.kaiming_uniform_(uf, a=2.2236)
        sparsify_features(uf, sparsity=self.init_sparsity)
        normalize_weight_(uf)

        if self.identify_up_and_down:
            self.down_filters = self.up_filters
        else:
            self.down_filters = nn.Parameter(uf.clone().detach())

    def regularize_weights(self):
        with torch.no_grad():
            self.pixel_sampler.regularize_weights()
            self.filter_sampler.regularize_weights()

            uf, df = self.up_filters, self.down_filters

            normalize_weight_(uf)
            # uf.unflatten(1, self.filter_shape)[:, 0].clip_(0.0)
            if not self.identify_up_and_down:
                normalize_weight_(df)

    def get_filters(self):
        return [
            F.pad(f.unflatten(1, self.filter_shape), self.filter_padding)
            for f in [self.up_filters, self.down_filters]
        ]

    def _get_weight(self, key):
        uf, df = self.get_filters()
        # f c h w -> f p i
        if key == "up":
            return self.filter_sampler(uf, normalize=True)
        elif key == "down":
            return self.filter_sampler(df, normalize=True)

        raise Exception(f"Wrong key: {key}")

    def get_weight(self, key="up"):
        if not self.training and self.weight_cache[key] is not None:
            return self.weight_cache[key]

        weight = self._get_weight(key)  # f p i

        if self.training:
            self.weight_cache[key] = None
        else:
            self.weight_cache[key] = weight.detach().clone()

        return weight

    def down(self, x, features):
        pred = features.sum(dim=1)
        pred = pred.unflatten(1, self.inp_shape)

        if self.add_pred_bias:
            pred = pred + self.pred_bias

        if self.clip_pred is not None:
            x = x.clip(*self.clip_pred)
            # x = FakeClip()(x)

        return pred

    def forward(self, x, with_features=False):
        x = self.pixel_sampler(x)  # b c h w -> b d c h w
        x = x.flatten(2)  # b d i

        uw = self.get_weight(key="up")  # f p i

        # scores = einsum("b i, f p i -> b p f", x, uw)
        scores = einsum("b d i, f p i -> b d p f", x, uw)
        scores = scores.flatten(1, 2)  # b (d p) f

        x, idx = scores.max(dim=1)  # b f

        if self.add_bias:
            x = x + self.bias

        x = self.act(x)

        if not with_features:
            return x

        xd = self.act_down(x) if self.training else x

        if self.identify_up_and_down:
            dw = uw
        else:
            dw = self.get_weight(key="down")  # f p i
        idx = idx % self.filter_sampler.n_samples

        # NOTE - the following selects f features for every batch element
        features = dw[
            list(range(idx.shape[1])) * idx.shape[0], idx.flatten()
        ]  # (b f) i
        features = features * xd.flatten()[..., None]  # (b f) i
        features = features.unflatten(0, idx.shape)  # b f i

        # return scx, features
        return x, features


class Denoiser(nn.Module):
    def regularize_weights(self):
        pass


class Transform(nn.Module):
    def forward(self, x):
        return (x - 0.5) * 2


class DenoiserAffineModule(nn.Module):
    def __init__(
        self,
        denoiser,
        affine_layer,
        loss_fn=None,
        transform=None,
        transform_target=True,
    ):
        super().__init__()
        # self.layers = nn.ModuleList(layers)
        self.denoiser = denoiser
        self.affine_layer = affine_layer

        self.loss_fn = loss_fn or CosineLoss()
        self.transform = transform or nn.Identity()
        self.transform_target = transform_target

    def forward(self, x, with_pred=False, with_features=False):
        x = self.transform(x)

        x = self.denoiser(x)
        pred = x

        if with_features:
            x, features = self.affine_layer(x, with_features=True)
            return x, features

        x = self.affine_layer(x, with_features=False)

        if with_pred:
            return x, pred

        return x

    def forward_with_preds(self, x):
        x, pred = self(x, with_pred=True)

        return x, [pred]

    def forward_with_loss(self, x, ssl_target, label):
        x, pred = self(x, with_pred=True)

        if self.transform_target:
            ssl_target = self.transform(ssl_target)

        loss = self.loss_fn(pred, ssl_target)

        return x, loss

    def get_preds(self, x):
        x, preds = self.forward_with_preds(x)

        return torch.stack(preds, dim=0)

    def regularize_weights(self):
        self.denoiser.regularize_weights()
        self.affine_layer.regularize_weights()
