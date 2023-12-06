import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import tensorflow as tf


def get_data(n_features):
    return _get_train_data(n_features), _get_test_data(n_features)


def _get_train_data(n_features):
    # Train Dataset
    # -------------

    # Set train shuffle seed (for reproducibility)
    torch.manual_seed(42)

    # We will concentrate on the first 128 samples (per class). Total of 256 samples.
    n_samples = 128

    # Use pre-defined torchvision function to load MNIST train data
    train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    # Filter out labels (originally 0-9), leaving only labels 0 and 1
    idx = np.append(
        np.where(
            train_data.targets == 0
        )[0][:n_samples],
        np.where(
            train_data.targets == 1
        )[0][:n_samples]
    )

    # randomizes the data
    np.random.shuffle(idx)

    train_data.data = train_data.data[idx]
    train_data.targets = train_data.targets[idx].float()

    train_data = dim_reduction(train_data, n_features)

    return train_data


def _get_test_data(n_features):
    # Test Dataset
    # -------------

    # Set test shuffle seed (for reproducibility)
    torch.manual_seed(7)

    # 128 per class. Total of 256 samples.
    n_samples = 128

    # Use pre-defined torchvision function to load MNIST test data
    test_data = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    # Filter out labels (originally 0-9), leaving only labels 0 and 1
    idx = np.append(
        np.where(
            test_data.targets == 0
        )[0][:n_samples],
        np.where(
            test_data.targets == 1
        )[0][:n_samples]
    )

    # randomizes the data
    np.random.shuffle(idx)

    test_data.data = test_data.data[idx]
    test_data.targets = test_data.targets[idx].float()

    test_data = dim_reduction(test_data, n_features)

    return test_data


def dim_reduction(loader, dim_reduct):
    loader.data = loader.data[..., np.newaxis] / 255.0
    loader.data = tf.image.resize(loader.data, (256, 1)).numpy()
    loader.data = tf.squeeze(loader.data).numpy()
    loader.data = PCA(dim_reduct).fit_transform(loader.data)
    for i, x in enumerate(loader.data):
        loader.data[i] = (x - x.min()) * (np.pi / (x.max() - x.min()))

    loader.data = torch.tensor(loader.data, dtype=torch.float32)

    return loader
