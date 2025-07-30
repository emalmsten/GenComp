from torchvision.datasets import MNIST
from torchvision import transforms
import torch

class CustomMNIST(MNIST):
    def __getitem__(self, index):
        # Fetch the data (image) and target (label)
        data, target = super().__getitem__(index)
        transform = transforms.ToTensor()
        data = transform(data)
        # Return data and set target to None
        return data, {}
