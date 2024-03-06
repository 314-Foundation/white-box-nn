import torch
import torch.nn.functional as F
from kornia.geometry.transform import affine, rotate
from torch import nn

from lib.helpers import divide_chunks, normalize_weight
from lib.modules.base import Module


class SFSampler(Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, x):
        # b i -> b p i
        x = x[:, None]
        return x


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


class RotationSampler(SFSampler):
    def __init__(self, n_samples):
        super().__init__(n_samples)

        self.after_init()

    def init_weights(self):
        self.poses = 360 * torch.arange(self.n_samples) / self.n_samples

    def forward(self, x, normalize=True):
        # x - f c h w

        shape = x.shape[1:]
        ch, cw = shape[1] // 2, shape[2] // 2
        center = torch.tensor([ch, cw]).float()

        poses = []
        for angle in self.poses:
            poses.append(rotate(x, angle, center=center))

        poses = torch.stack(poses, dim=1)  # f p c h w

        if normalize:
            poses = poses.flatten(2)
            poses = normalize_weight(poses)
            poses = poses.unflatten(2, shape)

        return poses
