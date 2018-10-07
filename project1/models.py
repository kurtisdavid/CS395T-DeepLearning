import torch.nn as nn
import torchvision.models
import torch
import math

class ResNet(nn.Module):

    def __init__(self, n_layers, final_output):
        super(ResNet,self).__init__()
        self.conv_params = {'kernel_size': 3, 'padding': 1}
        self.width = 186
        self.height = 171
        self.layer_dict = {
            18: [2,2,2,2],
            34: [3,4,6,3],
            50: [3,4,6,3],
            101: [3,4,23,3]
        }
        self.bottleneck = True if n_layers in [50,101] else False
        self.layers = {}

        in_channels = 3
        out_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1, dilation = 1)
        )
        self.width = self.width / 2
        self.width = math.floor( (self.width - 1) / 2 + 1 )
        self.height = self.height / 2
        self.height = math.floor( (self.height - 1) / 2 + 1)
#         print("Height is now:", self.height, "Width is now:", self.width)


        in_channels = 64

        num_repeat = self.layer_dict[n_layers]
        for i in range(2,6):
            self.res_layer = i
            # [ [blocks], transition ]
            self.layers[self.res_layer] = [[], None]
            for j in range(num_repeat[i-2]):
                self.create_block(in_channels, out_channels, j)
                if j == 0:
                    self.add_transition(in_channels, out_channels * 4 if self.bottleneck else out_channels)
                if self.bottleneck:
                    in_channels = out_channels * 4
                else:
                    in_channels = out_channels
            out_channels = out_channels * 2
        self.relu = nn.ReLU(inplace=True)
        # global average pooling
        self.global_avg = nn.AvgPool2d(kernel_size = (self.width,self.height), stride = 1)
        # fully connected to final
        self.output = nn.Linear(in_channels,1)

    def create_block(self, in_channels, out_channels, block_num):
        if self.bottleneck:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, stride = 1, kernel_size = 1),
                nn.BatchNorm2d(num_features = out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, stride = 1, **self.conv_params),
                nn.BatchNorm2d(num_features = out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels*4, stride = 2 if (block_num == 0 and self.res_layer > 2) else 1, kernel_size = 1),
                nn.BatchNorm2d(num_features = out_channels*4),
            )
#             print("Added", "conv_bottleneck" + str(self.res_layer) + "_" + str(block_num), "input: " + str(in_channels), "output: " + str(out_channels*4))
        else:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, stride = 1, **self.conv_params),
                nn.BatchNorm2d(num_features = out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, stride = 2 if (block_num == 0 and self.res_layer > 2) else 1, **self.conv_params),
                nn.BatchNorm2d(num_features = out_channels)
            )
#             print("Added", "conv" + str(self.res_layer) + "_" + str(block_num), "input: " + str(in_channels), "output: " + str(out_channels))
        self.add_module("conv" + str(self.res_layer) + "_" + str(block_num), block)
        self.layers[self.res_layer][0].append(block)

        if block_num == 0 and self.res_layer > 2:
            self.height = math.floor( (self.height - 1) / 2 + 1)
            self.width = math.floor( (self.width - 1) / 2 + 1)
#             print("Height is now:", self.height, "Width is now:", self.width)

    def add_transition(self, in_channel, out_channel):
        transition = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, stride = 2 if self.res_layer > 2 else 1, kernel_size = 1),
            nn.BatchNorm2d(num_features = out_channel),
        )
#         transition = nn.AvgPool2d(kernel_size = 3, stride = 2 if self.res_layer > 2 else 1, padding = 1)
        self.add_module("transition"+ str(self.res_layer), transition)
        self.layers[self.res_layer][1] = transition

    def forward(self, X):
        # go through conv1
        X = self.conv1(X)
        # go through residuals
        for i in range(2,self.res_layer + 1):
            layers,transition = self.layers[i]
            for j,layer in enumerate(layers):
                identity = X
                if j == 0:
                    identity = transition(X)
                X = layer(X) + identity
                X = self.relu(X)
        X = self.global_avg(X)
        X = X.view(X.shape[0],-1)
        X = self.output(X)
        return X.view(-1)

def create_pretrained_alexnet(n_classes):
    model = AlexNet()
    pretrained = torchvision.models.alexnet(pretrained=True)
    # fine-tune conv layers, keep fc since we have different dimensions
    model.features = pretrained.features
    return model

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv_params = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        self.maxpool_params = {'kernel_size': 2, 'stride': 1, 'padding': 1, 'dilation': 1}

        self.width = 186
        self.height = 171

        def conv_size(x, kernel_size, stride=1, padding=0):
            return ((x - kernel_size + 2*padding)/stride + 1 )

        def maxpool_size(x, kernel_size, stride, padding=0, dilation=1):
            return math.floor((x + 2 * padding - dilation * (kernel_size - 1) - 1) / stride  + 1)

        # taken from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        dims = [self.width, self.height]
        dims = [conv_size(x, kernel_size=11, stride=4, padding=2) for x in dims]
        dims = [maxpool_size(x, kernel_size=3, stride=2) for x in dims]
        dims = [conv_size(x, kernel_size=5, padding=2) for x in dims]
        dims = [maxpool_size(x, kernel_size=3, stride=2) for x in dims]
        dims = [conv_size(x, kernel_size=3, padding=1) for x in dims]
        dims = [conv_size(x, kernel_size=3, padding=1) for x in dims]
        dims = [conv_size(x, kernel_size=3, padding=1) for x in dims]
        dims = [maxpool_size(x, kernel_size=3, stride=2) for x in dims]
        self.width, self.height = dims

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * self.width * self.height, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * self.width * self.height)
        x = self.classifier(x)
        return x.view(-1)
