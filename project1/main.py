import argparse
import math
import models
import numpy as np
import pickle
import torch
import time
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision.models

def train(model, name):
    # image pre-processing
    train_data_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5089547997389491],
                                     std=[1])])
    allImages = datasets.ImageFolder(root='./training',transform = train_data_transform)
    label_mapping = torch.FloatTensor([float(clazz) for clazz in allImages.classes])
    label_mapping_scaled = (label_mapping - label_mapping.min())/(label_mapping.max() - label_mapping.min())
    dataloader = torch.utils.data.DataLoader(allImages, batch_size = 32, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    val_data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5089547997389491],
                                     std=[1])])
    valImages = datasets.ImageFolder(root='./validation', transform = val_data_transform)
    label_mapping_v = torch.FloatTensor([float(clazz) for clazz in valImages.classes])
    label_mapping_scaled_v = (label_mapping_v - label_mapping.min())/(label_mapping.max() - label_mapping.min())

    val_dataloader = torch.utils.data.DataLoader(valImages, batch_size = 256, shuffle=True)

    train_evalImages = datasets.ImageFolder(root='./training', transform = val_data_transform)
    train_eval_dataloader = torch.utils.data.DataLoader(train_evalImages, batch_size = 256
    , shuffle=True)


    # training
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr = 0.0001, betas = (0.9,0.999))
    loss_metric = nn.L1Loss()
    n_epochs = 51
    iteration = 0

    train_losses = []
    eval_losses = []
    iterations = []

    for e in range(n_epochs):
        epoch_losses = []
        print("Epoch", e)
        print("Evaluating train...")
        inference(model, device, train_eval_dataloader, label_mapping_scaled, loss_metric, train_losses)
        print("Evaluating test...")
        inference(model, device, val_dataloader, label_mapping_scaled_v, loss_metric, eval_losses)
        iterations.append(iteration)
        print("Iteration", iteration,"\t Train loss:", train_losses[-1], "Val loss:", str(eval_losses[-1]))
        if e == 50:
            break
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
            epoch_losses.append(loss.item())

            loss.backward()
            optim.step()
            # evaluation
            iteration += 1

        # print("Epoch %d: Training Loss: %0.3f" % (e,np.mean(epoch_losses)))


    torch.save(model, name + "2.pt")
    with open(name + "_train_loss2.txt", "wb") as f:
        pickle.dump(train_losses, f)

    with open(name + "_eval_loss2.txt", "wb") as f:
        pickle.dump(eval_losses, f)

    with open(name + "_iterations2.txt", "wb") as f:
        pickle.dump(iterations, f)

def inference(model, device, dataloader, label_mapping, loss_metric, losses):
    model.eval()
    losses_ = []
    with torch.no_grad():
        for i,(batch_input,batch_labels) in enumerate(dataloader):
            if i % 25 == 0:
                print("Evaluating " + str(i), end = "\r")
            batch_input = batch_input.to(device)
            batch_labels = label_mapping[batch_labels].to(device)
            res = model(batch_input)
            loss = loss_metric(res, batch_labels)
            losses_.append(loss.item())
    losses.append(np.mean(losses_))


def main():
    args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument("--model", help="The model to train on")
    args = args.parse_args()

    if args.model == 'alexnet':
        model = models.AlexNet()
        train(model, args.model)
    elif args.model == 'resnet':
        model = models.ResNet(101, 1)
        train(model, args.model)
    else:
        if args.model == 'all':
            # model = models.AlexNet()
            # train(model, 'AlexNet')
            # del model
            model = models.create_pretrained_alexnet(1)
            train(model,'AlexNetFinetuned')
            del model
            # model = models.ResNet(34,1)
            # train(model, 'ResNet34')
            # del model
            model = models.ResNet(50,1)
            train(model, 'ResNet50')
            del model
            model = models.ResNet(101,1)
            train(model, 'ResNet101')
            del model
        raise ValueError("Did not provide a valid model")

if __name__ == "__main__":
    main()
