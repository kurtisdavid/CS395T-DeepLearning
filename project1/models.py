class ResNet(nn.Module):
    def __init__(self, n_layers, final_output, bottleneck = False):
        super(ResNet,self).__init__()
        self.conv_params = {'kernel_size': 3, 'padding': 1}
        self.width = 186
        self.height = 171
        self.layer_dict = {
            18: [2,2,2,2],
            34: [3,4,6,3]
        }
        self.layers = {}

        in_channels = 1
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
        # print("Height is now:", self.height, "Width is now:", self.width)

        in_channels = 64

        num_repeat = self.layer_dict[n_layers]
        for i in range(2,6):
            self.res_layer = i
            # [ [blocks], transition ]
            self.layers[self.res_layer] = [[], None]
            for j in range(num_repeat[i-2]):
                self.create_block(in_channels, out_channels, j, bottleneck)
                if j == 0:
                    self.add_transition()
                in_channels = out_channels
            out_channels = out_channels * 2

        # global average pooling
        self.global_avg = nn.AvgPool2d(kernel_size = (self.width,self.height), stride = 1)
        # fully connected to final
        self.output = nn.Linear(in_channels,1)

    def create_block(self, in_channels, out_channels, block_num, bottleneck):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride = 1, **self.conv_params),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, stride = 2 if block_num == 0 else 1, **self.conv_params),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU()
        )
        self.add_module("conv" + str(self.res_layer) + "_" + str(block_num), block)
        self.layers[self.res_layer][0].append(block)
        # print("Added", "conv" + str(self.res_layer) + "_" + str(block_num), "input: " + str(in_channels), "output: " + str(out_channels))

        if block_num == 0:
            self.height = math.floor( (self.height - 1) / 2 + 1)
            self.width = math.floor( (self.width - 1) / 2 + 1)
            # print("Height is now:", self.height, "Width is now:", self.width)

    def add_transition(self):
        transition = nn.Sequential(
            nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        self.add_module("transition"+ str(self.res_layer), transition)
        self.layers[self.res_layer][1] = transition

    def forward(self, X):
        # go through conv1
        X = self.conv1(X)
        # go through residuals
        for i in range(2,self.res_layer + 1):
            layers,transition = self.layers[i]
            for j,layer in enumerate(layers):
                if j == 0:

                    pool = transition(X)
                    # dimension transition
                    if i > 2:
                        padding = (0,0,0,0,pool.shape[1]//2,pool.shape[1]//2,0,0)
                        pool = nn.functional.pad(pool,padding)
                    X = layer(X)
                    X = X + pool
                else:
                    X = layer(X)
        X = self.global_avg(X)
        X = X.view(X.shape[0],-1)
        X = self.output(X)
        return X.view(-1)


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
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
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


class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.conv_params = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        self.maxpool_params = {'kernel_size': 2, 'stride': 2, 'padding': 1, 'dilation': 1}

        self.layers = []

        self.in_channels = 1
        self.maxpool_out = -1
        self.in_features = -1
        self.width = 186
        self.height = 171

        # this assumes you can mix conv and maxpools, with all fc at the end
        def conv(out_channels):
            self.layers.append(nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, **self.conv_params),
                nn.BatchNorm2d(num_features = out_channels),
                nn.LeakyReLU(0.2)
            ))
            self.in_channels = out_channels
            # https://www.quora.com/How-can-I-calculate-the-size-of-output-of-convolutional-layer
            self.width = math.floor((self.width - self.conv_params['kernel_size'] + 2*self.conv_params['padding'])/self.conv_params['stride'] + 1 )
            self.height = math.floor((self.height - self.conv_params['kernel_size'] + 2*self.conv_params['padding'])/self.conv_params['stride'] + 1 )
            # print("applied conv: image is now ", self.width, " by ", self.height, " by ", self.in_channels)
            return self.layers[-1]

        def maxpool_size(x):
            kernel_size = self.maxpool_params['kernel_size']
            padding = self.maxpool_params['padding']
            stride = self.maxpool_params['stride']
            dilation = self.maxpool_params['dilation']
            return math.floor((x + 2 * padding - dilation * (kernel_size - 1) - 1) / stride  + 1)

        def maxpool():
            self.layers.append(nn.MaxPool2d(**self.maxpool_params))
            self.width = maxpool_size(self.width)
            self.height = maxpool_size(self.height)

            self.maxpool_out = self.width * self.height * self.in_channels
            print("applied maxpool: image is now ", self.width, " by ", self.height, " by ", self.in_channels)
            return self.layers[-1]

        def fc(out_features, first=False):
            in_features = self.maxpool_out if first else self.in_features

            self.layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.Sigmoid()
            ))
            self.in_features = out_features
            return self.layers[-1]

        # need them to be instance variables to be found by vgg.parameters() method
        self.c1 = conv(64)
        self.m1 = maxpool()
        self.c2 = conv(128)
        self.m2 = maxpool()
        self.conv_params['stride'] = 2
        self.c3 = conv(256)
        self.c4 = conv(256)
        self.m3 = maxpool()
        self.c5 = conv(128)
        self.conv_params['stride'] = 1
        self.m4 = maxpool()
        self.fc1 = fc(512, first=True)
        self.fc2 = fc(512)
        self.output = fc(1)

    def forward(self,X):
        for layer in self.layers[:-3]:
            X = layer(X)

        X = X.view(-1, self.maxpool_out)
        for layer in self.layers[-3:]:
            X = layer(X)

        return X.view(-1)
