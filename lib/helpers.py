import matplotlib.pyplot as plt
import numpy as np
import robbytorch as robby
import torch


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def _norm(w, p=2.0, dim=-1):
    return torch.norm(w, p=p, dim=dim, keepdim=True) + 1e-10


def normalize_weight(w, p=2.0, dim=-1):
    return w / _norm(w, p=p, dim=dim)


def normalize_weight_(w, p=2.0, dim=-1):
    return w.div_(_norm(w, p=p, dim=dim))


class Helper:

    @staticmethod
    def plot_example(X, y=None, n_row=3, n_col=16, cmap="bwr", no_y_labels=True):
        X = X.detach()
        X = np.transpose(X, (0, 2, 3, 1))
        if X.shape[-1] == 1:
            X = X.squeeze(-1)
            # X = np.repeat(X, 3, -1)

        num = min(len(X), n_row * n_col)

        if y is None:
            y = list(range(num))
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
            if no_y_labels:
                ax.set_yticklabels([])
            ax.set_title("{}".format(y[i]))
        plt.tight_layout()
        plt.show()

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


hh = Helper()
