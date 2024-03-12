from collections import OrderedDict

import torch
from torch import nn

from lib.modules.augment import Noise
from lib.modules.sf_layers import AffineLayer, ConvLayer, TwoPieceLayer, TwoStepFunction
from lib.modules.sf_samplers import AffineSampler, RotationSampler, TwoPieceRollSampler

IMG_SHAPE = (1, 28, 28)


def get_model(latent_dim=8, img_shape=IMG_SHAPE):

    pixel_layer = TwoStepFunction(10)

    conv_layer = ConvLayer(
        sampler=RotationSampler(32),
        n_kernels=2,
        kernel_size=5,
        act=nn.ReLU(),
        # add_bias=True,
        # rescale=True,
    )

    affine_layer = AffineLayer(
        sampler=AffineSampler(32, dim_pose_group=latent_dim),
        inp_shape=img_shape,
        feature_shape=(1, 20, 20),
        out_dim=latent_dim,
        act=nn.ReLU(),
        # add_bias=True,
        # rescale=True,
    )

    sparse = TwoPieceLayer(
        sampler=TwoPieceRollSampler(latent_dim // 2, mask_grad=True),
        inp_dim=latent_dim,
    )

    backbone = nn.Sequential(
        OrderedDict(
            [
                ("pixel", pixel_layer),
                ("conv", conv_layer),
                ("affine", affine_layer),
                ("sparse", sparse),
            ]
        )
    )

    return backbone


def visualize_processing_steps(model, x):
    x0 = model.pixel(x)
    x1 = model.conv(x0)
    x2, f = model.affine(x1, with_features=True)
    x3 = model.sparse(x2)

    fx = f + 0.1 * x1[:, None]

    per_feature = x2.shape[1] // 2
    ff0, ff1 = torch.split(x2, per_feature, dim=1)

    val0, idx0 = ff0.max(dim=1)
    f0 = fx[range(len(idx0)), idx0]

    val1, idx1 = ff1.max(dim=1)
    f1 = fx[range(len(idx1)), idx1 + per_feature]

    y = [
        (round(scores[0].item(), ndigits=2), round(scores[1].item(), ndigits=2))
        for scores in x3.softmax(dim=1)
    ]
    y += ["pixel" for _ in range(len(x3))]
    y += ["conv" for _ in range(len(x3))]
    y += [round(e.item(), ndigits=2) for e in val0]
    y += [round(e.item(), ndigits=2) for e in val1]

    ret = torch.cat([x, x0, x1, f0, f1], dim=0)

    return ret, y
