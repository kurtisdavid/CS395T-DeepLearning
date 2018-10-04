import argparse
import math
import models
import numpy as np
import pickle
import torch
import torch.nn as nn
from torchvision import transforms, datasets

def train(model, name):
    # image pre-processing
    data_transform = transforms.Compose([transforms.Grayscale(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5089547997389491],
                                     std=[1])])
    allImages = datasets.ImageFolder(root='./training',transform = data_transform)
    label_mapping = torch.FloatTensor([float(clazz) for clazz in allImages.classes])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    label_mapping_scaled = (label_mapping - label_mapping.min())/(label_mapping.max() - label_mapping.min())
    dataloader = torch.utils.data.DataLoader(allImages, batch_size = 64, shuffle=True)

    valImages = datasets.ImageFolder(root='./validation', transform = data_transform)
    label_mapping_v = torch.FloatTensor([float(clazz) for clazz in valImages.classes])
    label_mapping_scaled_v = (label_mapping_v - label_mapping.min())/(label_mapping.max() - label_mapping.min())

    val_dataloader = torch.utils.data.DataLoader(valImages, batch_size = 128, shuffle=False)


    # training
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr = 0.01, betas = (0.9,0.999))
    loss_metric = nn.L1Loss()
    n_epochs = 10
    iteration = 0

    train_losses = []
    eval_losses = []
    iterations = []

    for e in range(n_epochs):
        epoch_losses = []
        for batch_input, batch_labels in dataloader:
            if iteration % 25 == 0:
                print(iteration)

            # make sure to zero out gradient
            model.train()
            model.zero_grad()

            # move to gpu + get correct labels
            batch_input = batch_input.to(device)
            batch_labels = label_mapping_scaled[batch_labels].to(device)

            loss = loss_metric(model(batch_input), batch_labels)
            epoch_losses.append(loss.data)

            loss.backward()
            optim.step()

            # evaluation
            if iteration % 50 == 0:
                train_losses.append(loss.data)

                model.eval()
                losses = []
                for batch_input,batch_labels in val_dataloader:
                    batch_input = batch_input.to(device)
                    batch_labels = label_mapping_scaled_v[batch_labels].to(device)
                    res = model(batch_input)
                    loss = loss_metric(res, batch_labels)
                    losses.append(loss.data)

                eval_losses.append(np.mean(losses))
                iterations.append(iteration)

            iteration += 1

        print("Epoch %d: Training Loss: %0.3f" % (e,np.mean(epoch_losses)))

    with open(name + "_train_loss.txt", "wb") as f:
        pickle.dump(train_losses, f)

    with open(name + "_eval_loss.txt", "wb") as f:
        pickle.dump(eval_losses, f)

    with open(name + "_iterations.txt", "wb") as f:
        pickle.dump(iterations, f)

def main():
    args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument("--model", help="The model to train on")
    args = args.parse_args()

    if args.model == 'alexnet':
        model = models.AlexNet()
    elif args.model == 'resnet':
        model = models.ResNet(34, 1)
    else:
        raise ValueError("Did not provide a valid model")

    train(model, args.model)

if __name__ == "__main__":
    main()
