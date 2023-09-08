from torch.utils.data import Dataset
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

class MNISTDataset(Dataset):
    def __init__(self, train=True, root="data"):
        self.mnist_data = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=ToTensor()
        )
        self.X = self.mnist_data.data.detach().numpy()
        self.y = self.mnist_data.targets.detach().numpy()
    
    def __len__(self):
        return len(self.mnist_data)
    
    def __getitem__(self, idx):
        image, label = self.mnist_data[idx]
        return image, label