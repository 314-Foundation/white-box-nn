import torch
import torch.nn.functional as F
from kornia.geometry.transform import affine, rotate
from torch import nn

from lib.helpers import divide_chunks, normalize_weight
from lib.modules.activation import TopKActivation
from lib.modules.base import Module


class SFSampler(Module):
    def __init__(self, n_samples=1):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, x):
        # b i -> b p i
        x = x[:, None]
        return x


class RotationSampler(SFSampler):
    def __init__(self, n_samples, max_rotation=360):
        super().__init__(n_samples)
        self.max_rotation = max_rotation

        self.after_init()

    def init_weights(self):
        self.poses = nn.Parameter(
            self.max_rotation * torch.arange(self.n_samples) / self.n_samples,
            requires_grad=False,
        )

    def forward(self, x, normalize=True):
        # x - f c h w

        shape = x.shape[1:]
        ch, cw = shape[1] // 2, shape[2] // 2
        center = torch.tensor([ch, cw]).float().to(x.device)

        poses = []
        for angle in self.poses:
            poses.append(rotate(x, angle, center=center))

        poses = torch.stack(poses, dim=1)  # f p c h w

        if normalize:
            poses = poses.flatten(2)
            poses = normalize_weight(poses)
            poses = poses.unflatten(2, shape)

        return poses


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


class AffineSampler(SFSampler):
    def __init__(
        self,
        n_samples,
        dim_pose_group=1,
        pose_init_mult=0.01,
    ):
        super().__init__(n_samples)
        self.dim_pose_group = dim_pose_group
        self.pose_init_mult = pose_init_mult

        self.after_init()

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
        pass
        # for pose in self.poses:
        #     prune_pose(pose)

    def forward(self, x, normalize=True):
        # x - f c h w

        n_repeats = x.shape[0] // self.dim_pose_group
        shape = x.shape[1:]

        poses = []
        for pose in self.poses:
            poses.append(
                affine(
                    x,
                    (pose.repeat(n_repeats, 1, 1) if self.dim_pose_group > 1 else pose),
                    mode="bilinear",
                    padding_mode="zeros",
                )
            )

        poses = torch.stack(poses, dim=1)  # f p c h w

        if normalize:
            poses = poses.flatten(2)
            poses = normalize_weight(poses)
            poses = poses.unflatten(2, shape)

        return poses


class PiecewiseRollSampler(SFSampler):
    def __init__(self, n_samples, mask_grad=True):
        super().__init__(n_samples)
        self.mask_grad = mask_grad
        self.topk = TopKActivation(1, dim=-1)

        self.after_init()

    def init_weights(self):
        self.poses = list(range(self.n_samples))

    def roll(self, x):
        poses = []
        for pose in self.poses:
            xx = torch.roll(x, shifts=pose, dims=-1)
            poses.append(xx)

        poses = torch.stack(poses, dim=1)  # f p i

        if self.mask_grad:
            poses = self.topk(poses)

        return poses

    def repeat(self, x):
        x = x[:, None]
        x = x.repeat((1, self.n_samples, 1))
        return x

    def forward(self, x, normalize=False):
        # x - f i
        f, i = x.shape
        assert i % f == 0
        split_size = i // f

        fs = torch.split(x, split_size, dim=1)  # [(f s), (f s), ...]
        fps = [self.roll(ff) for ff in fs]  # [(f p s), (f p s), ...]
        fps_id = [self.repeat(ff) for ff in fs]  # [(f p s), (f p s), ...]

        fis = []
        for j, fi in enumerate(fps):
            left = torch.cat(fps_id[:j], dim=-1) if j > 0 else None
            right = torch.cat(fps_id[(j + 1) :], dim=-1) if (j + 1) < len(fps) else None
            if left is not None:
                fi = torch.cat([left, fi], dim=-1)
            if right is not None:
                fi = torch.cat([fi, right], dim=-1)

            # fi - f p i
            fis.append(fi)

        fis = [fi[j] for j, fi in enumerate(fis)]  # [(p i), (p i), ...]
        poses = torch.stack(fis, dim=0)  # f p i

        if normalize:
            poses = normalize_weight(poses)

        return poses
