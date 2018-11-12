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

def TVLossMat(model, layer_mask=None):
    conv2D_idxs = [0, 3, 6, 8, 10]
    if layer_mask is None:
        layer_mask = [i for i in range(conv2D_idxs)]
    features = list(model.features.children())
    tv = 0

    pixels = 0
    for layer in layer_mask:
        weights = list(features[conv2D_idxs[layer]].parameters())[0]
        size = weights.size()
        print("size: ", size)

        pixels += size[0] * size[1] * (size[2] - 1) * (size[3] - 1)

        x = torch.zeros_like(weights)
        y = torch.zeros_like(weights)
        x[:, :, :-1, :-1] = (weights[:, :, 1:, :-1] - weights[:, :, :-1, :-1]) ** 2
        y[:, :, :-1, :-1] = (weights[:, :, :-1, 1:] - weights[:, :, :-1, :-1]) ** 2
        tv += torch.sum(torch.sqrt(x + y))
    
    print("tv", tv)
    print("pixels", pixels)
    return tv / pixels
