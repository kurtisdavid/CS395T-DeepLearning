import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import *

class GeolocationDataset(Dataset):

    def __init__(self, data, root_dir, transform=None):
        self.locs = []
        self.imgs = []
        with open(data, 'r') as f:
            for line in f:
                split = line.split()
                self.imgs.append(split[0])
                self.locs.append([float(split[1]), float(split[2])])
        self.locs = np.array(self.locs, dtype=float)
        min_lat, max_lat = np.min(self.locs[:, 0]), np.max(self.locs[:, 0])
        min_lon, max_lon = np.min(self.locs[:, 1]), np.max(self.locs[:, 1])
        print("Latitude [{:f},{:f}], Longitude [{:f},{:f}]".format(
            min_lat, max_lat, min_lon, max_lon))
        self.locs[:, 0] = (self.locs[:, 0] - min_lat) / (max_lat - min_lat)
        self.locs[:, 1] = (self.locs[:, 1] - min_lon) / (max_lon - min_lon)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.imgs[idx])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return {'img': img, 'loc': torch.FloatTensor(self.locs[idx])}

mean_pixel = 0.4774289128851716
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[mean_pixel], std=[1])])
dataset = GeolocationDataset('data/geo/geo_train.txt', 'data/geo/train',
                             transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = VGG(150, 261, 2)
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr = 0.01, betas = (0.9,0.999))
loss_metric = nn.L1Loss()
n_epochs = 10
iteration = 0

for e in range(n_epochs):
    epoch_losses = []
    for _, batch in enumerate(dataloader):
        batch_input = batch['img']
        batch_labels = batch['loc']
        if iteration % 25 == 0:
            print(iteration)

        # make sure to zero out gradient
        model.train()
        model.zero_grad()

        # move to gpu + get correct labels
        batch_input = batch_input.to(device)
        batch_labels = batch_labels.to(device)

        loss = loss_metric(model(batch_input).reshape(-1, 2), batch_labels)
        del batch_input
        del batch_labels
        epoch_losses.append(loss.data)

        loss.backward()
        optim.step()
        iteration += 1
    torch.save(model, str(e) + '-1.pt')

    print("Epoch %d: Training Loss: %0.3f" % (e,np.mean(epoch_losses)))

'''
# Calculate mean pixel
running = 0
tot_images = 0
for _, batch in enumerate(dataloader):
    locs = batch['loc']
    imgs = np.squeeze(batch['img'].numpy())
    running += np.mean(imgs) * len(imgs)
    tot_images += len(imgs)
print(running / tot_images)
'''
