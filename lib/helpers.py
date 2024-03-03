import matplotlib.pyplot as plt
import numpy as np
import robbytorch as robby
import torch
from torchvision import transforms

# Note - you must have torchvision installed for this example


class Helper:
    RAND_ROTATE = transforms.RandomRotation(180)

    @staticmethod
    def rand_rotate(x):
        return Helper.RAND_ROTATE(x)

    @staticmethod
    def plot_example(X, y=None, n_row=3, n_col=16, cmap="bwr"):
        X = X.detach()
        X = np.transpose(X, (0, 2, 3, 1))
        if X.shape[-1] == 1:
            X = X.squeeze(-1)
            # X = np.repeat(X, 3, -1)

        num = min(len(X), n_row * n_col)

        if y is None:
            y = num * ["--"]
        # plot images
        fig, axes = plt.subplots(n_row, n_col, figsize=(max(4, 1.5 * n_col), 2 * n_row))
        for i in range(num):
            if n_row == 1 and n_col == 1:
                ax = axes
            elif n_row == 1 or n_col == 1:
                ax = axes[i]
            else:
                ax = axes[i // n_col, i % n_col]
            ax.imshow(X[i], cmap=cmap)
            ax.set_title("{}".format(y[i]))
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_conv(X):
        X = np.transpose(X, (1, 2, 3, 0))
        X = np.repeat(X, 3, -1)
        n_row = 1
        n_col = len(X)
        num = n_row * n_col
        # plot images
        fig, axes = plt.subplots(n_row, n_col, figsize=(5 * num, 5 * num))
        for i in range(num):
            ax = axes[i] if num > 1 else axes
            ax.imshow(X[i], cmap="gray")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def add_noise(x, scale=0.6):
        return (torch.randn_like(x) * scale + x).clip(0, 1)

    @staticmethod
    def forward_robby(module, dataitem, phase):
        targets = dataitem["target"]
        logits = module(dataitem["data"])
        loss = module.robby_loss(logits, targets)
        return {"loss": loss}

    @staticmethod
    def compute_advs(
        model,
        data,
        target,
        eps=0.3,
        step_size=0.1,
        Nsteps=20,
        constraint="inf",
        forward=None,
        minimize=False,
        random_start=False,
    ):
        if eps == 0:
            return data
        forward = (
            forward or getattr(model, "forward_robby", None) or Helper.forward_robby
        )
        with torch.enable_grad():
            with torch.inference_mode(False):
                return robby.input_transforms.PGD(
                    model,
                    {"data": data.clone(), "target": target.clone()},
                    forward,
                    constraint=constraint,
                    eps=eps,
                    step_size=step_size,
                    Nsteps=Nsteps,
                    use_tqdm=False,
                    minimize=minimize,
                    random_start=random_start,
                )

    @staticmethod
    def add_adv_noise(model, x, y, eps, adv_prob):
        if eps == 0.0:
            return x

        if torch.rand((1,)).item() < 1 - adv_prob:
            return Helper.add_noise(x, eps)

        if torch.rand((1,)).item() < 0.0:  # 0.2
            constraint = "2"
            eps = 1.5
            step_size = 0.5
            Nsteps = 10
        else:
            constraint = "inf"
            eps = 0.3
            # eps = 8 / 255
            step_size = 0.1
            # step_size = 2 / 255
            Nsteps = 10

        x = Helper.compute_advs(
            model=model,
            data=x,
            target=y,
            eps=eps,
            constraint=constraint,
            step_size=step_size,
            Nsteps=Nsteps,
            forward=Helper.forward_robby,
        )
        return x

    @staticmethod
    def mixup_batch(batch, eps=0.3):
        x, y = batch

        x = Helper.mixup_data(x, eps)[0]
        # y2 = torch.roll(y, b // 2)

        return x, y

    @staticmethod
    def mixup_data(x, eps=0.3):
        b = x.shape[0]

        x1, x2, r = torch.split(x, [b // 2, b // 2, b % 2], dim=0)
        x = torch.cat([(1 - eps) * x1 + eps * x2, eps * x1 + (1 - eps) * x2, r], dim=0)
        # x = torch.cat([x1 + eps * x2, eps * x1 + x2, r], dim=0)
        # x = x.clip(0, 1)
        # y2 = torch.roll(y, b // 2)
        revx = torch.cat([x2, x1], dim=0)

        return x, revx

    @staticmethod
    def to_show(pe, detach=True, separate=False):
        pe = pe.cpu()
        if detach:
            pe = pe.detach()

        if separate:
            shape = pe.shape[1:]
            x = pe.flatten(1)
            x = x - x.min(dim=-1)[0][..., None]
            x = x / x.max(dim=-1)[0][..., None]
            pe = x.unflatten(1, shape)
        else:
            pe = pe - pe.min()
            pe = pe / pe.max()

        return pe

    @staticmethod
    def to_show2(pe, detach=True, shift=0.5):
        pe = pe.cpu()
        if detach:
            pe = pe.detach()

        pe = pe + shift
        pe = pe.clip(0, 1)

        return pe


def sparsify_features(tensor, nonzero=1, sparsity=None):
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    features, dim = tensor.shape

    if sparsity is not None:
        num_zeros = int(sparsity * dim)
    else:
        num_zeros = dim - nonzero

    with torch.no_grad():
        for ft_idx in range(features):
            inp_indices = torch.randperm(dim)
            zero_indices = inp_indices[:num_zeros]
            tensor[ft_idx, zero_indices] = 0
    return tensor


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class OutDims:
    def __init__(self, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x):
        return (x + 2 * self.padding - self.kernel_size) // self.stride + 1


hh = Helper()
