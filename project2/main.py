import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import torch.utils.data as data
from utils import *
from models import *
#import matplotlib.pyplot as plt
import pickle
import sys
import random

'''
setup hyperparameter selection in args
'''
def get_args():
    # General system running and configuration options
    parser = argparse.ArgumentParser(description='main.py')

    # true or false args
    parser.add_argument('-no_tv', action='store_true', default=False, help='use TV regularization')
    parser.add_argument('-tv', action='store_true', default=False, help='use TV regularization')
    parser.add_argument('-tv3d', action='store_true', default=False, help='use 3D TV regularization')
    parser.add_argument('-tv4d', action='store_true', default=False, help='use 4D TV regularization')
    parser.add_argument('-l1', action='store_true', default=False, help='use L1 weight decay')
    parser.add_argument('-l2', action='store_true', default=False, help='use L2 weight decay')
    parser.add_argument('-weights', action='store_true', default=False, help='save weights')
    parser.add_argument('-no_log', action='store_true', default=False, help='logging experiments')

    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=200, help='num epochs to train for')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer to use')
    parser.add_argument('--lr', type=float, default=.001 if '--optim' not in sys.argv else 0.1)
    parser.add_argument('--train_bs', type=int, default=128, help='batch size')
    parser.add_argument('--eval_bs', type=int, default=256, help='batch size')
    parser.add_argument('--model', type=str, default='alexnet', help='model to test')
    parser.add_argument('--mask', nargs='+', type=int, default=None, help='layer mask for TV')
    parser.add_argument('--lambda_reg', type=float, default=1e-4, help='l1/l2 regularization weight')
    parser.add_argument('--lambda_TV', type=float, default=1, help='tv regularization weight')
    parser.add_argument('--lambda_mask', nargs='+', type=float, default=None, help='lambdas for each layer in mask')
    parser.add_argument('--model_file', type=str, default='./model.pt', help='where to save trained model')
    parser.add_argument('--log_file',
                        type=str,
                        default='./trial.pck',
                        help='path to save logged pickle results',
                        required='-no_log' not in sys.argv)
    parser.add_argument('--load_model_init',
                        type=str,
                        default=None,
                        help='use to start with same initial weights',
                        required='-no_tv' not in sys.argv)
    parser.add_argument('--save_model_init',
                        type=str,
                        default='./default.pt',
                        help='save init for a given trial name string',
                        required='-no_tv' in sys.argv)

    args = parser.parse_args()
    return args

'''
setup dataloaders for training and testing sets for cifar10
'''
def setup_data(args):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    dataloader = datasets.CIFAR10
    batch_size = args.train_bs
    test_bs = args.eval_bs

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = dataloader(root='./data', train=False, download=True,
            transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=test_bs, shuffle=False)

    return trainloader,testloader

'''
save kernel images
'''
def visualizeWeights(model, folder, title):
    params_c = list(list(model.features.children())[0].parameters())[0].detach().cpu().numpy()
    params_c = np.moveaxis(params_c,1,-1)
    params_c = (params_c - params_c.min())/(params_c.max() - params_c.min())

    fig = plt.figure(figsize=(24, 24))
    for i in range(1,65):
        plt.subplot(11,11,i)
        weights = params_c[i-1,:,:,:]
        plt.imshow(weights)
    plt.title(title)
    plt.savefig('./filters/' + title+'.png')

'''
evaluation
'''
def eval_model(model, testloader, criterion, device):
    model.eval()
    class_losses = []
    accs = []
    with torch.no_grad():
        for batch_input, batch_labels in testloader:
            batch_input = batch_input.to(device)
            batch_labels = batch_labels.to(device)

            batch_output = model(batch_input)
            batch_pred = torch.argmax(batch_output, dim = 1)
            acc = torch.mean(torch.eq(batch_pred,batch_labels).type(torch.float))
            class_loss = criterion(batch_output,batch_labels)

            accs.append(acc.item())
            class_losses.append(class_loss.item())
    return np.mean(accs),np.mean(class_losses)

'''
trainer
'''
def train_model(model, trainloader, testloader, args, tv_fn, device):
    losses = []
    TVs = []
    val_losses = []
    val_accs = []
    layer_tvs = [[] for i in range(len(args.mask))]

    EPOCHS = args.epochs
    lr = args.lr
    lambda_TV = args.lambda_TV
    lambda_reg = args.lambda_reg
    if not args.l1 and not args.l2:
        lambda_reg = 0
    criterion = nn.CrossEntropyLoss()
    if args.optim=='adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=lambda_reg)
    elif args.optim=='SGD':
        optim = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=lambda_reg)
    steps = [75,120]

    for e in range(EPOCHS):
        # validation
        val_acc, val_loss = eval_model(model,testloader,criterion,device)
        print("Epoch", "{:3d}".format(e), "| Test Acc:", "{:8.4f}".format(val_acc), "| Test Loss:", "{:8.4f}".format(val_loss), end=" ")
        sys.stdout.flush()
	val_accs.append(val_acc)
        val_losses.append(val_losses)
         # visualize weights
        if args.weights:
            visualizeWeights(model, 'WithTV_Epoch' + str(e))
        with torch.no_grad():
            init, layer_tv = tv_fn(model)
            TVs.append(init.item())
            assert len(layer_tv) == len(layer_tvs)
            for i in range(len(layer_tvs)):
                try:
                    layer_tvs[i].append(layer_tv[i].item())
                except:
                    layer_tvs[i].append(layer_tv[i])
        # training
        model.train()

        class_losses = []
        TV_losses = []

        # annealing learning rate
        if args.optim=='SGD' and e in steps:
            lr /= 10
            optim = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=0.9,
                                    weight_decay=lambda_reg if (args.l1 or args.l2) else 0)

        # go through batches
        for batch_input, batch_labels in trainloader:
            optim.zero_grad()
            batch_input = batch_input.to(device)
            batch_labels = batch_labels.to(device)
            batch_output = model(batch_input)

            # classification loss
            class_loss = criterion(batch_output,batch_labels)
            # total variation loss
            TV_loss, layer_tv = tv_fn(model)
            if not args.no_tv and args.lambda_mask is None:
                TV_loss = lambda_TV*TV_loss
                loss = class_loss + TV_loss
            elif not args.no_tv and len(layer_tv) == len(args.lambda_mask):
                TV_loss = sum([layer_tv[i]*args.lambda_mask[i]
                                          for i in range(len(args.lambda_mask))])
                loss = class_loss + TV_loss
            else:
                loss = class_loss
            loss.backward()
            # check for gradient problems
            # check_grad(model)
            optim.step()


            class_losses.append(class_loss.item())
            TV_losses.append(TV_loss.item())

        losses.append(np.mean(class_losses))
        print(" | Train loss:", "{:8.4f}".format(losses[-1]), "| Init TV:", "{:8.4f}".format(init), "| TV loss:", "{:8.4f}".format(np.mean(TV_losses)))
	sys.stdout.flush()

    results = {}
    # print(val_accs)
    results['layer_TVs'] = layer_tvs
    results['val_accs'] = val_accs
    results['val_losses'] = val_losses
    results['losses'] = losses
    results['TVs'] = TVs
    results['args'] = args

    if not args.no_log:
        with open(args.log_file, 'wb') as f:
            pickle.dump(results, f)

        torch.save(model, args.model_file)


def main():
    args = get_args()
    # up to 1 type of regularization allowed
    # assert (args.l1 and not args.l2 and not args.tv) or \
    #        (args.l2 and not args.l1 and not args.tv) or \
    #        (args.tv and not args.l1 and not args.l2) or \
    #        (not args.tv and not args.l1 and not args.l2)

    assert args.lambda_mask is None or len(args.lambda_mask)==len(args.mask)
    assert (not args.no_tv and (args.tv and not args.tv3d and not args.tv4d) or \
           (args.tv3d and not args.tv and not args.tv4d) or \
           (args.tv4d and not args.tv and not args.tv3d) ) or \
           (not args.tv and not args.tv3d and not args.tv4d)

    tv_state = (int(args.tv),int(args.tv3d),int(args.tv4d))
    tv_dict = {
        (1,0,0): TVMat,
        (0,1,0): TVMat3D,
        (0,0,1): TVMat4D
    }
    tv_loss = tv_dict[tv_state]


    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = setup_data(args)

    if args.model == 'alexnet':
        model = AlexNet(10).to(device)
        if args.mask == None:
            args.mask = [0,1,2,3,4]
        tv_fn = lambda model: TVLossMat(model, args.mask)
    elif args.model == 'resnet20':
        if args.mask == None:
            args.mask = [0,1,2,3]
        model = resnet20().to(device)
        tv_fn = lambda model: TVLossResNet(model, args.mask, tv_loss)
    else:
        raise Exception('Given model is invalid.')

    # transfer init weights to reduce compounding factors of stochasticity
    if not args.no_tv:
        model.load_state_dict(torch.load(args.load_model_init))
    else:
        torch.save(model.state_dict(),args.save_model_init)
    # train now
    train_model(model, trainloader, testloader, args, tv_fn, device)


if __name__ == "__main__":
    main()
