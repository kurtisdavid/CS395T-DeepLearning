import torch
import torch.nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
]) 

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
]) 
dataloader = datasets.CIFAR100
batch_size = 128

trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = dataloader(root='./data', train=False, download=False,
        transform=transform_train)
testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False)

print("SUCCESS")
