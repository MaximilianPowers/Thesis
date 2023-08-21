import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class SinusoidalRegression(Dataset):
    def __init__(self, n_samples=100, noise=0.05):
        self.y = 2*np.random.sample(n_samples)-1 + np.random.normal(0, noise, n_samples)
        self.y[self.y > 1] = 1
        self.y[self.y < -1] = -1
        X_1 = np.arcsin(self.y)
        X_2 = np.arccos(self.y)
        self.X = np.stack((X_1, X_2), axis=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]).float(), torch.tensor(self.y[idx]).float()
