import torch
import torch.nn
import argparse
from torchvision import models, datasets, transforms
import torch.utils.data as data
from utils import *
from models import *
import matplotlib.pyplot as plt
import pickle
import sys


def get_args():
    # General system running and configuration options
    parser = argparse.ArgumentParser(description='main.py')

    parser.add_argument('--models',
                        type=str,
                        nargs='+',
                        default=[],
                        help='models to extract saliency maps from')
    parser.add_argument('--names',
                        type=str,
                        nargs='+',
                        default=[],
                        help='model names for titles')
    parser.add_argument('--bs',
                        type=int,
                        default=1,
                        help='number of eval images to get from')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory to load in cifar')

    args = parser.parse_args()
    return args

'''
setup dataloaders for training and testing sets for cifar10
'''
def setup_data(args):
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    dataloader = datasets.CIFAR10
    batch_size = args.bs

    testset = dataloader(root=args.data_dir, train=False, download=True,
            transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return testloader

def main():
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get a single batch and store in X,y
    testloader = setup_data(args)
    for X,y in testloader:
        X = X.to(device)
        y = y.to(device)
        break 

    model = resnet20().to(device)
    saliency_maps = []
    for model_path in args.models:
        model.load_state_dict(torch.load(model_path).state_dict())
        saliency_maps.append(compute_saliency_map(X,y, model,device))
    show_all_saliency_maps(saliency_maps)  
        

     

    
if __name__ == '__main__':
    main()   
