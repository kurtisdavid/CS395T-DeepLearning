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
    parser.add_argument('--save_file', type=str, default='./cam.jpg', help='what to save as')
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
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return testloader

def unnormalize(tensor):
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def main():
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get a single batch and store in X,y
    testloader = setup_data(args)
    for X,y in testloader:
        X = X.to(device)
        y = y.to(device)

        image = unnormalize(X)        
        break 

    CAM_maps = []
    finalconv_name = 'layer3'
    for model_path in args.models:
        model = resnet20().to(device)
        model.load_state_dict(torch.load(model_path).state_dict())
        
        features_blobs = []
        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())
 
        model._modules.get(finalconv_name).register_forward_hook(hook_feature)
        params = list(model.parameters())
        weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
        CAM_maps.append(CAM(image, y, model, features_blobs, weight_softmax, device))
    show_all_CAM(CAM_maps, image, args.save_file)  

    
if __name__ == '__main__':
    main()   
