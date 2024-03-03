import math
import warnings

from pytorch_lightning import LightningDataModule
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data import DataLoader, Subset  # , random_split
from torchvision import transforms

# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

from lib.datasets import SalientCIFAR


# For older versions of torch
def random_split(dataset, lengths, generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size=64,
        labels=None,
        shape=(28, 28),
        n_train_samples=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        self.transform = transforms.Compose(
            [
                transforms.RandomCrop(shape[0], padding=0),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )  # no normalization - for robust evaluation
        self.transform_test = transforms.Compose(
            [transforms.CenterCrop(shape), transforms.ToTensor()]
        )  # no normalization - for robust evaluation

        self.batch_size = batch_size
        self.labels = labels
        self.n_train_samples = n_train_samples

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist = self.get_mnist(self.data_dir, train=True, transform=self.transform)
            # self.mnist_train, self.mnist_val, _ = random_split(mnist, [0.2, 0.2, 0.6])
            # self.mnist_train, self.mnist_val, _ = random_split(mnist, [0.5, 0.1, 0.4])
            self.mnist_train, self.mnist_val = random_split(mnist, [0.9, 0.1])
            if self.n_train_samples:
                self.mnist_train.indices = self.mnist_train.indices[
                    : self.n_train_samples
                ]

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = self.get_mnist(
                self.data_dir, train=False, transform=self.transform_test
            )

    def get_mnist(self, data_dir, train, transform):
        if self.labels is None:
            target_transform = lambda x: x
        else:
            target_dict = {tgt: idx for idx, tgt in enumerate(self.labels)}
            target_transform = lambda x: target_dict[x]

        mnist = MNIST(
            data_dir,
            train=train,
            transform=transform,
            target_transform=target_transform,
        )
        if self.labels is None:
            return mnist

        indices = [
            idx for idx, target in enumerate(mnist.targets) if target in self.labels
        ]
        return Subset(mnist, indices)

    def train_dataloader(self, batch_size=None):
        return DataLoader(
            self.mnist_train, batch_size=(batch_size or self.batch_size), shuffle=True
        )

    def val_dataloader(self, batch_size=None):
        return DataLoader(
            self.mnist_val, batch_size=(batch_size or self.batch_size), shuffle=False
        )

    def test_dataloader(self, batch_size=None):
        return DataLoader(
            self.mnist_test, batch_size=(batch_size or self.batch_size), shuffle=False
        )


class CIFARDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size=64, labels=None):
        super().__init__()
        self.data_dir = data_dir
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )  # no normalization - for robust evaluation
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.batch_size = batch_size
        self.labels = labels

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar = self.get_cifar(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar, [0.9, 0.1])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = self.get_cifar(
                self.data_dir, train=False, transform=self.transform_test
            )

    def get_cifar(self, data_dir, train, transform):
        if self.labels is None:
            target_transform = lambda x: x
        else:
            target_dict = {tgt: idx for idx, tgt in enumerate(self.labels)}
            target_transform = lambda x: target_dict[x]

        cifar = CIFAR10(
            data_dir,
            train=train,
            transform=transform,
            target_transform=target_transform,
        )
        if self.labels is None:
            return cifar

        indices = [
            idx for idx, target in enumerate(cifar.targets) if target in self.labels
        ]
        return Subset(cifar, indices)

    def train_dataloader(self, batch_size=None):
        return DataLoader(
            self.cifar_train, batch_size=(batch_size or self.batch_size), shuffle=True
        )

    def val_dataloader(self, batch_size=None):
        return DataLoader(
            self.cifar_val, batch_size=(batch_size or self.batch_size), shuffle=False
        )

    def test_dataloader(self, batch_size=None):
        return DataLoader(
            self.cifar_test, batch_size=(batch_size or self.batch_size), shuffle=False
        )


class SalientCIFARDataModule(CIFARDataModule):
    def __init__(self, data_dir: str = "./data", batch_size=64, labels=None):
        super().__init__()
        self.data_dir = data_dir
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.transform = transforms.Compose(
            [
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
            ]
        )  # no normalization - for robust evaluation
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.batch_size = batch_size
        self.labels = labels

    def get_cifar(self, data_dir, train, transform):
        cifar = SalientCIFAR(data_dir, train=train, transform=transform)
        if self.labels is None:
            return cifar

        indices = [
            idx for idx, target in enumerate(cifar.targets) if target in self.labels
        ]
        return Subset(cifar, indices)


class GrayCIFARDataModule(CIFARDataModule):
    def __init__(self, data_dir: str = "./data", batch_size=64, labels=None):
        super().__init__()
        self.data_dir = data_dir
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                lambda x: x.mean(dim=0).unsqueeze(0),
            ]
        )  # no normalization - for robust evaluation
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                lambda x: x.mean(dim=0).unsqueeze(0),
            ]
        )

        self.batch_size = batch_size
        self.labels = labels


class FashionDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size=64, labels=None):
        super().__init__()
        self.data_dir = data_dir
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )  # no normalization - for robust evaluation

        self.batch_size = batch_size
        self.labels = labels

    def prepare_data(self):
        # download
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist = self.get_fashion(
                self.data_dir, train=True, transform=self.transform
            )
            # self.mnist_train, self.mnist_val, _ = random_split(mnist, [0.2, 0.2, 0.6])
            # self.mnist_train, self.mnist_val, _ = random_split(mnist, [0.5, 0.1, 0.4])
            self.mnist_train, self.mnist_val = random_split(mnist, [0.9, 0.1])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = self.get_fashion(
                self.data_dir, train=False, transform=self.transform
            )

    def get_fashion(self, data_dir, train, transform):
        mnist = FashionMNIST(data_dir, train=train, transform=transform)
        if self.labels is None:
            return mnist

        indices = [
            idx for idx, target in enumerate(mnist.targets) if target in self.labels
        ]
        return Subset(mnist, indices)

    def train_dataloader(self, batch_size=None):
        return DataLoader(
            self.mnist_train, batch_size=(batch_size or self.batch_size), shuffle=True
        )

    def val_dataloader(self, batch_size=None):
        return DataLoader(
            self.mnist_val, batch_size=(batch_size or self.batch_size), shuffle=False
        )

    def test_dataloader(self, batch_size=None):
        return DataLoader(
            self.mnist_test, batch_size=(batch_size or self.batch_size), shuffle=False
        )
