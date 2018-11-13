import torch

def TVLoss(model, layer_mask=None):
    conv2D_idxs = [0, 3, 6, 8, 10]
    if layer_mask is None:
        layer_mask = [i for i in range(len(conv2D_idxs))]

    features = list(model.features.children())
    tv = 0

    pixels = 0
    for layer in layer_mask:
        weights = list(features[conv2D_idxs[layer]].parameters())[0]
        size = weights.size()

        pixels += size[0] * size[1] * (size[2] - 1) * (size[3] - 1)

        for kernel in range(size[0]):
            for channel in range(size[1]):
                y = weights[kernel][channel]

                for i in range(size[2] - 1):
                    for j in range(size[3] - 1):
                        tv += torch.sqrt((y[i + 1][j] - y[i][j])**2 + (y[i][j + 1] - y[i][j])**2)

    return tv / pixels

def TVLossMat(model, print_=False, layer_mask=None):
    conv2D_idxs = [0, 3, 6, 8, 10]
    if layer_mask is None:
        layer_mask = [i for i in range(len(conv2D_idxs))]
    features = list(model.features.children())
    tv = 0

    pixels = 0
    for layer in layer_mask:
        weights = list(features[conv2D_idxs[layer]].parameters())[0]
        size = weights.size()
#         print("size: ", size)

        pixels += size[0] * size[1] * (size[2] - 1) * (size[3] - 1)

        x = torch.zeros_like(weights)
        y = torch.zeros_like(weights)
        x[:, :, :-1, :-1] = (weights[:, :, 1:, :-1] - weights[:, :, :-1, :-1]) ** 2
        y[:, :, :-1, :-1] = (weights[:, :, :-1, 1:] - weights[:, :, :-1, :-1]) ** 2
        huh = torch.sqrt(x + y)
        ok = torch.sum(huh)
        tv += ok
        if print_:
            print("sqrt",huh)
            print("ok",ok)
            print("tv", tv)
            print("pixels", pixels)
    if print_:
        print("final_tv", tv)
        print("final_pixels", pixels)
    return tv / pixels

def TVLossMatResNet(model, layer_mask=None):

    blocks = [model.conv1, model.layer1,model.layer2,model.layer3,model.layer4]
    if layer_mask is None:
        layer_mask = [i for i in range(len(blocks))]

    # get tv values for convolutions in resnet block
    def TVBlock(block):
        block = list(block.children())
        pixels = 0
        tv = 0
        weights = list(block[0].parameters())[0]
        size = weights.size()
        pixels += size[0] * size[1] * (size[2] - 1) * (size[3] - 1)
        tv += TVMat(weights)

        weights = list(block[3].parameters())[0]
        size = weights.size()
        pixels += size[0] * size[1] * (size[2] - 1) * (size[3] - 1)
        tv += TVMat(weights)

        return tv, pixels

    tv = 0
    pixels = 0
    for layer in layer_mask:
        if layer == 0:
            weights = list(blocks[layer].parameters())[0]
            size = weights.size()
            pixels += size[0] * size[1] * (size[2] - 1) * (size[3] - 1)
            tv += TVMat(weights)
        else:
            basic_blocks = list(blocks[layer].children())
            for bl in basic_blocks:
                curr_tv, curr_pixels = TVBlock(bl)
                pixels += curr_pixels
                tv += curr_tv
    return tv/pixels

def TVMat(weights):
    x = torch.zeros_like(weights)
    y = torch.zeros_like(weights)
    x[:, :, :-1, :-1] = (weights[:, :, 1:, :-1] - weights[:, :, :-1, :-1]) ** 2
    y[:, :, :-1, :-1] = (weights[:, :, :-1, 1:] - weights[:, :, :-1, :-1]) ** 2
    return torch.sum(torch.sqrt(x+y))
