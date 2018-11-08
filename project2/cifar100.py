from torchvision.datasets import CIFAR100
import torch
imagenet_data = CIFAR100('data/', download=True)
data = torch.DoubleTensor(imagenet_data.train_data)
data = data.reshape(-1, 3)
print(data.mean(0) / 255, data.std(0) / 255)
