from os import path
import util
import numpy as np
import argparse
from util import *
import csv
from PIL import Image

import torch
from torchvision import transforms

def load(image_path, mean_pixel):
    #TODO:load image and process if you want to do any
    img = Image.open(image_path)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[mean_pixel],
                                         std=[1])])
    img = data_transform(img.convert('RGB'))
    return torch.FloatTensor(img.reshape(1, img.shape[0], img.shape[1], img.shape[2])).cuda()
    
class Predictor:
    DATASET_TYPE = 'yearbook'
    def __init__(self):
        self.model = None
    # baseline 1 which calculates the median of the train data and return each time
    def yearbook_baseline(self):
        # Load all training data
        train_list = listYearbook(train=True, valid=False)

        # Get all the labels
        years = np.array([float(y[1]) for y in train_list])
        med = np.median(years, axis=0)
        return [med]

    # Compute the median.
    # We do this in the projective space of the map instead of longitude/latitude,
    # as France is almost flat and euclidean distances in the projective space are
    # close enough to spherical distances.
    def streetview_baseline(self):
        # Load all training data
        train_list = listStreetView(train=True, valid=False)

        # Get all the labels
        coord = np.array([(float(y[1]), float(y[2])) for y in train_list])
        xy = coordinateToXY(coord)
        med = np.median(xy, axis=0, keepdims=True)
        med_coord = np.squeeze(XYToCoordinate(med))
        return med_coord

    def predict(self, image_path):


        #TODO: load model

        #TODO: predict model and return result either in geolocation format or yearbook format
        # depending on the dataset you are using
        if self.DATASET_TYPE == 'geolocation':
            img = load(image_path, 0.4774289128851716)
            if self.model is None:
                self.model = torch.load(path.join('..','model','geoloc.pt'))
                self.scale = torch.FloatTensor(np.array([14.326867, 9.71078])).cuda()
                self.min = torch.FloatTensor(np.array([-4.786422, 41.390225])).cuda()
            output = self.model(img) * self.scale + self.min
            return output.data
            #result = self.streetview_baseline() #for geolocation
        elif self.DATASET_TYPE == 'yearbook':
            img = load(image_path, 0.5089547997389491)
            if self.model is None:
                self.model = torch.load(path.join('..','model','yearbook.pt'))
            output = self.model(img) * 108 + 1905
            return output.data
        return result
        
    


