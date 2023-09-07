import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.mnist_data = datasets.MNIST(
            root="./data",
            train=train,
            download=True,
            transform=transform
        )
        
    def __len__(self):
        return len(self.mnist_data)
    
    def __getitem__(self, idx):
        return self.mnist_data[idx]
    
