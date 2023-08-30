from math import pi
from sklearn.datasets import make_moons, make_circles, make_blobs
from torch.utils.data import Dataset
import torch
import numpy as np

class MoonDataset(Dataset):
    def __init__(self, n_samples=100, noise=0.05):
        self.X, self.y = make_moons(n_samples=n_samples, noise=noise)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]).float(), torch.tensor(self.y[idx]).float()


class SpiralDataset(Dataset):
    def __init__(self, n_samples, noise=0.05, length=2*pi):
        dims = 2
        points_per_cls = int(n_samples/2)

        theta = np.sqrt(np.random.rand(points_per_cls))*length

        r_a = 2*theta + pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + noise * np.random.randn(points_per_cls, dims)

        r_b = -2*theta - pi
        data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
        x_b = data_b + noise * np.random.randn(points_per_cls, dims)

        res_a = np.append(x_a, np.zeros((points_per_cls, 1)), axis=1)
        res_b = np.append(x_b, np.ones((points_per_cls, 1)), axis=1)

        res = np.append(res_a, res_b, axis=0)
        np.random.shuffle(res)

        self.X = res[:, :dims]
        self.y = res[:, dims]

        self.x_a = x_a
        self.x_b = x_b

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]).float(), torch.tensor(self.y[idx]).float()


class BlobsDataset(Dataset):
    def __init__(self, n_samples, noise=0.15, centers=4):
        self.X, self.y = make_blobs(n_samples=n_samples, n_features=2, centers=centers,
                                    cluster_std=noise, center_box=(-4.0, 4.0), shuffle=True, random_state=None)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]).float(), torch.tensor(self.y[idx]).float()


class CirclesDataset(Dataset):
    def __init__(self, n_samples, noise=0.15):
        self.X, self.y = make_circles(
            n_samples=n_samples, noise=noise, shuffle=True, random_state=None, factor=0.8)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]).float(), torch.tensor(self.y[idx]).float()
